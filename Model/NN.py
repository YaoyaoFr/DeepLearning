from abc import ABCMeta

import os
import h5py
import numpy as np
import tensorflow as tf

from Log.log import Log
from Model.utils_model import EarlyStop, get_metrics
from Model.utils_model import check_dataset
from Schemes.xml_parse import parse_xml_file
from Layer.LayerConstruct import build_layer


class NeuralNetwork(object, metaclass=ABCMeta):
    log = None
    graph = None
    sess = None
    minimizer = None
    prediction = None
    global_step = None
    NN_type = 'Neural Network'

    data_placeholder = {
        'data': 'input_tensor',
        'label': 'output_tensor',
    }

    def __init__(self,
                 dir_path: str,
                 log: Log = None,
                 scheme: int or str = 1,
                 graph: tf.Graph = None,
                 spe_pas: dict = None, 
                 ):
        self.pa = {}
        self.tensors = {}
        self.results = {}
        self.op_layers = []
        self.input_placeholders = {}

        self.dir_path = dir_path
        self.project_path = os.path.join(dir_path, 'Program/Python/DeepLearning')
        self.set_graph(log=log, graph=graph)

        self.scheme = scheme
        self.load_parameters(scheme=scheme, spe_pas=spe_pas)

        with self.log.graph.as_default():
            # Build the input layers
            for placeholder_pa in self.pa['layers']['input']:
                layer = build_layer(arguments=placeholder_pa, parameters=self.pa['basic'])
                self.input_placeholders[placeholder_pa['scope']] = layer()

            # Build the operation layers
            for layer_pa in self.pa['layers']['layer']:
                layer = build_layer(arguments=layer_pa, parameters=self.pa['basic'])
                self.op_layers.append(layer)

    def load_parameters(self, 
                        scheme: str, 
                        spe_pas: dict = None, 
                        ):
        """
        Load parameters from configuration file (.xml)
        :param scheme: file path of configuration file
        :return:
        """
        if not spe_pas:
            pas = parse_xml_file(os.path.join(self.project_path, 'Schemes/{:s}.xml'.format(scheme)))
        else: 
            pas = spe_pas

            self.pa['early_stop'] = pas['parameters']['early_stop']
            self.pa['training'] = pas['parameters']['training']
            self.pa['basic'] = pas['parameters']['basic']
            self.pa['layers'] = pas['layers']

    def build_structure(self):
        parameters = list()

        input_tensors = None
        for layer in self.op_layers:
            layer(tensors=input_tensors,
                  placeholders=self.input_placeholders)
            layer_tensor = layer.tensors
            self.tensors[layer.pa['scope']] = layer_tensor
            if 'weight' in layer_tensor:
                parameters.append(layer_tensor['weight'])
            if 'bias' in layer_tensor:
                parameters.append(layer_tensor['bias'])
            input_tensors = layer_tensor

        output_tensor = input_tensors['output']

        print('Build {:s}.'.format(self.NN_type))

        self.build_optimizer(output_tensor=output_tensor)

        self.log.saver = tf.train.Saver()
        self.initialization()
        pass

    def build_optimizer(self,
                        output_tensor,
                        penalties: list = []):
        lr_place = self.input_placeholders['learning_rate']
        output_place = self.input_placeholders['output_tensor']

        # Cost function
        self.results = get_metrics(output_tensor=output_tensor,
                                   ground_truth=output_place)
        self.prediction = tf.nn.softmax(output_tensor)

        # Build loss function, which contains the cross entropy and regularizers.
        self.results['Cost'] = self.results['Cross Entropy']

        penalties.extend(['L1', 'L2'])
        for regularizer in penalties:
            loss_name = '{:s}_loss'.format(regularizer)
            loss_collection = tf.get_collection(loss_name)
            loss = tf.Variable(0.0, trainable=False)
            if len(loss_collection) > 0:
                loss = tf.add_n(loss_collection)
            self.results['{:s} Penalty'.format(regularizer)] = loss
            self.results['Cost'] += 0.5 * loss

        # build minimizer

        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            self.global_step = tf.Variable(initial_value=0, trainable=False, name='global_step')
            optimizer = tf.train.AdamOptimizer(lr_place, name='optimizer')
            with tf.variable_scope('graph_nn', reuse=tf.AUTO_REUSE):
                self.minimizer = optimizer.minimize(self.results['Cost'],
                                                    global_step=self.global_step,
                                                    name='minimizer',
                                                    )

        print('{:s} Optimizer initialized.'.format(self.NN_type))

    def training(self,
                 data: h5py.Group or dict,
                 run_time: int = 1,
                 fold_index: int = None,
                 restored_path: str = None,
                 show_info: bool = True,
                 ):
        with self.graph.as_default():
            self.build_structure()

        # data = self.load_data(data)

        early_stop = self.backpropagation(data=data)
        early_stop.show_results(run_time=run_time, fold_index=fold_index)
        early_stop.clear_models()

        return early_stop.results

    def backpropagation(self,
                        data: dict,
                        start_epoch: int = 0,
                        early_stop: EarlyStop = None,
                        ):
        self.log.write_graph()
        batch_size = self.pa['training']['train_batch_size']

        if early_stop is None:
            early_stop = EarlyStop(log=self.log,
                                   data=data,
                                   results=self.results,
                                   pas=self.pa['early_stop'])

        epoch = early_stop.epoch
        while epoch < early_stop.training_cycle:
            # Training
            self.backpropagation_epoch(data=data,
                                       batch_size=batch_size,
                                       learning_rate=early_stop.learning_rate,
                                       )

            # Evaluation
            results_epoch = self.predicting(data=data, epoch=epoch)

            early_stop.next(results=results_epoch)
            epoch = early_stop.epoch

        return early_stop

    def backpropagation_epoch(self,
                              data: dict,
                              batch_size: int,
                              learning_rate: float,
                              training: bool = True,
                              minimizer=None
                              ):

        # Shuffle
        tag = 'train'
        sample_size = check_dataset(data, tag=tag, data_placeholder=self.data_placeholder)
        random_index = np.random.permutation(sample_size)
        shuffled_data = {key: data[key][random_index]
                         for key in data if tag in key}

        # Start training
        for batch_index in np.arange(start=0, step=batch_size, stop=sample_size):
            batch_data = {key: shuffled_data[key][batch_index:batch_index + batch_size]
                          for key in shuffled_data}

            feed_dict = {self.input_placeholders[self.data_placeholder[key]]: batch_data['{:s} {:s}'.format(tag, key)]
                         for key in self.data_placeholder}

            feed_dict.update({self.input_placeholders['learning_rate']: learning_rate,
                              self.input_placeholders['training']: training})

            # Backpropagation
            if minimizer is None:
                minimizer = self.minimizer
            self.sess.run(fetches={'minimizer': minimizer},
                          feed_dict=feed_dict,
                          )

    def predicting(self, 
                   data: np.ndarray, 
                   epoch: int, 
                   ):
        
        results_epoch = {}
        for tag in ['train', 'valid', 'test']:
            if '{:s} data'.format(tag) not in data:
                continue
            results_epoch[tag] = self.feedforward(data=data,
                                                    epoch=epoch + 1,
                                                    tag=tag,
                                                    show_info=False,
                                                    )
        return results_epoch

    def feedforward(self,
                    epoch: int,
                    data: np.ndarray,
                    tag: str = 'valid',
                    if_save: bool = True,
                    show_info: bool = True,
                    get_tensors: bool = True,
                    ):

        feed_dict = {self.input_placeholders[self.data_placeholder[key]]: data['{:s} {:s}'.format(tag, key)]
                    for key in self.data_placeholder}
        feed_dict[self.input_placeholders['training']] = False

        fetches = {'results': self.results,
                   'global_step': self.global_step,
                   }

        if get_tensors:
            fetches['tensors'] = self.tensors

        # Feedforward
        result_batch = self.sess.run(fetches=fetches,
                                     feed_dict=feed_dict,
                                     )

        results = result_batch['results']

        self.log.write_log(res=results,
                           epoch=epoch,
                           log_type=tag,
                           if_save=if_save,
                           show_info=show_info)
        return results

    def initialization(self, init_op=None, name='', sess=None):
        if sess is None:
            sess = self.sess
        if init_op is None:
            init_op = tf.global_variables_initializer()
            name = 'all'

        sess.run(init_op)
        print('Parameters {:s} initialized.'.format(name))

    def set_graph(self, log=None, graph=None):
        if log:
            self.log = log
        else:
            log = Log(dir_path=self.dir_path)
            self.log = log

        self.graph = log.graph
        self.sess = log.sess

    def load_data(self,
                  fold: h5py.Group or dict):
        if isinstance(fold, dict):
            return fold

        data = {}
        for dataset in ['train', 'valid', 'test']:
            for label in self.data_placeholder:
                try:
                    key = '{:s} {:s}'.format(dataset, label)
                    data[key] = np.array(fold[key])
                except KeyError:
                    raise Warning('')

        return data

    @staticmethod
    def load_dataset(hdf5_file, 
                     scheme: str):
        dataset = {}

        scheme_group = hdf5_file['scheme {:s}'.format(scheme)]
        for fold_index in range(5):
            fold_dataset = {}
            fold_group = scheme_group['fold {:d}'.format(fold_index+1)]
            for tag in ['train', 'valid', 'test']:
                for data_type in ['data', 'label']:
                    str = '{:s} {:s}'.format(tag, data_type)
                    try:
                        fold_dataset[str] = np.array(fold_group[str])
                    except KeyError:
                        continue
            dataset['fold {:d}'.format(fold_index + 1)] = fold_dataset
        hdf5_file.close()

        return dataset
