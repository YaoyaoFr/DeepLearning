import collections
import os
from abc import ABCMeta

import h5py
import numpy as np
import tensorflow as tf
import scipy.io as sio

from Dataset.utils import hdf5_handler
from Layer.LayerConstruct import build_layer
from Log.log import Log
from Model.early_stop import EarlyStop
from Model.utils_model import check_dataset, get_metrics
from Schemes.xml_parse import parse_xml_file


class NeuralNetwork(object, metaclass=ABCMeta):
    """The base class of all neural network model

    Arguments:
        object {[type]} -- [description]

    Keyword Arguments:
        metaclass {[type]} -- [description] (default: {ABCMeta})

    Raises:
        Warning: [description]

    Returns:
        [type] -- [description]
    """
    log = None
    graph = None
    sess = None

    minimizer = None
    prediction = None
    global_step = None
    model_type = 'Neural Network'

    def __init__(self,
                 scheme: str,
                 dir_path: str,
                 log: Log = None,
                 spe_pas: dict = None,
                 ):
        self.pas = {}
        self.results = {}
        self.op_layers = []
        self.tensors = collections.OrderedDict()
        self.trainable_pas = collections.OrderedDict()
        self.input_placeholders = {}
        self.data_placeholder = {
            'data': 'input_tensor',
            'label': 'output_tensor',
        }

        self.scheme = scheme
        self.dir_path = dir_path
        self.project_path = os.path.join(
            dir_path, 'Program/Python/DeepLearning')

        self.set_graph(log=log)
        self.load_parameters(scheme=scheme, spe_pas=spe_pas)

        # Build the input layers
        for placeholder_pa in self.pas['layers']['input']:
            layer = build_layer(arguments=placeholder_pa,
                                parameters=self.pas['basic'])
            self.input_placeholders[placeholder_pa['scope']] = layer()

        # Build the operation layers
        for layer_pa in self.pas['layers']['layer']:
            layer = build_layer(arguments=layer_pa,
                                parameters=self.pas['basic'])
            self.op_layers.append(layer)

    def set_graph(self,
                  log: Log):
        """Set the log instance, session, and most important, the default graph for neural network model.

        Arguments:
            log {Log} -- [description]
        """
        self.log = log
        self.graph = log.graph
        self.sess = log.sess

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
            pas = parse_xml_file(os.path.join(
                self.project_path, 'Schemes/{:s}.xml'.format(scheme)))
        else:
            pas = spe_pas

        self.pas['early_stop'] = pas['parameters']['early_stop']
        self.pas['training'] = pas['parameters']['training']
        self.pas['basic'] = pas['parameters']['basic']
        self.pas['layers'] = pas['layers']

    def build_structure(self,
                        if_initialization: bool = True):
        """Build the structure according to parameters in xml file

        Keyword Arguments:
            if_initialization {bool} -- Whether initialize all the variables. (default: {True})
        """
        input_tensors = None
        for layer in self.op_layers:
            layer(tensors=input_tensors,
                  placeholders=self.input_placeholders)
            input_tensors = layer.tensors

            scope = layer.parameters['scope']
            self.tensors[scope] = input_tensors
            train_pas = {'{:s}/{:s}'.format(scope, p): layer.trainable_pas[p]
                         for p in layer.trainable_pas}
            self.trainable_pas.update(train_pas)

        print('Build {:s}.'.format(self.model_type))

        # Build the optimizers of neural network
        output_tensor = input_tensors['output']
        self.build_optimizer(output_tensor=output_tensor)

        # self.log.saver = tf.train.Saver(name='saver')
        if if_initialization:
            self.initialization()

        self.log.saver = tf.train.Saver(
            self.trainable_pas, max_to_keep=1000, name='saver')

    def initialization(self,
                       init_op: tf.Operation = None,
                       name: str = '',
                       ):
        """Initialization all the trainable variables

        Keyword Arguments:
            init_op {[type]} -- [description] (default: {None})
            name {str} -- [description] (default: {''})
            sess {[type]} -- [description] (default: {None})
        """
        if init_op is None:
            init_op = tf.global_variables_initializer()
            name = 'all'

        self.sess.run(init_op)
        print('Parameters {:s} initialized.'.format(name))

    def build_optimizer(self,
                        output_tensor: tf.Tensor,
                        penalties: list = None):
        """Build optimizers of the neural network.

        Arguments:
            output_tensor {tf.Tensor} -- Output or predicting of the neural network

        Keyword Arguments:
            penalties {list} -- Penalty items in the loss function. (default: {None})
        """
        lr_place = self.input_placeholders['learning_rate']
        output_place = self.input_placeholders['output_tensor']

        # Cost function
        self.results = get_metrics(output_tensor=output_tensor,
                                   ground_truth=output_place)
        self.prediction = tf.nn.softmax(output_tensor)

        # Build loss function, which contains the cross entropy and regularizers.
        self.results['Cost'] = self.results['Cross Entropy']

        if penalties is None:
            penalties = ['L1', 'L2']

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
            self.global_step = tf.Variable(
                initial_value=0, trainable=False, name='global_step')
            optimizer = tf.train.AdamOptimizer(lr_place, name='optimizer')
            with tf.variable_scope('graph_nn', reuse=tf.AUTO_REUSE):
                self.minimizer = optimizer.minimize(self.results['Cost'],
                                                    global_step=self.global_step,
                                                    name='minimizer',
                                                    )

        print('{:s} Optimizer initialized.'.format(self.model_type))

    def training(self,
                 data: dict,
                 run_time: int = 1,
                 fold_name: str = None,
                 if_show: bool = True,
                 ):
        """The main training process of the neural network, which includes: 1. build structure.
        2. write the architecture to file. 3. backpropagation 4. show results if permitted.
        5. save first, optimal and final model in the training process.

        Arguments:
            data {dict} -- [description]

        Keyword Arguments:
            run_time {int} --  (default: {1})
            fold_name {str} -- 'fold 1', 'fold 2', ... (default: {None})
            if_show {bool} --  (default: {True})

        Returns:
            [type] -- [description]
        """
        self.build_structure()
        self.log.write_graph()

        early_stop = self.backpropagation(data=data)

        if if_show:
            early_stop.show_results(run_time=run_time, fold_name=fold_name)

        early_stop.clear_models()
        return early_stop.results

    def backpropagation(self,
                        data: dict,
                        early_stop: EarlyStop = None,
                        ):
        """Backpropagation of neural network.

        Arguments:
            data {dict} -- Input dataset include train, valid and test data and label.

        Raises:
            Warning: [description]

        Returns:
            [type] -- [description]
        """
        batch_size = self.pas['training']['train_batch_size']

        if early_stop is None:
            early_stop = EarlyStop(log=self.log,
                                   data=data,
                                   results=self.results,
                                   pas=self.pas['early_stop'])

        epoch = early_stop.epoch
        training_cycle = early_stop.parameters['training_cycle']
        while epoch < training_cycle:
            # Training
            self.backpropagation_epoch(data=data,
                                       batch_size=batch_size,
                                       learning_rate=early_stop.parameters['learning_rate'],
                                       )

            # Evaluation
            results_epoch = self.predicting(data=data, epoch=epoch)

            epoch = early_stop.next(results=results_epoch)

        return early_stop

    def backpropagation_epoch(self,
                              data: dict,
                              batch_size: int,
                              learning_rate: float,
                              training: bool = True,
                              minimizer=None
                              ):
        """Feed data and labels, then run the optimizer with session

        Arguments:
            data {dict} -- dataset
            batch_size {int} -- batch size of training process
            learning_rate {float} -- learning rate of the optimization

        Keyword Arguments:
            training {bool} --  (default: {True})
            minimizer {[type]} --  (default: {None})
        """

        # Shuffle
        tag = 'train'
        sample_size = check_dataset(
            data, tag=tag, data_placeholder=self.data_placeholder)
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
                   restored_model: str = None,
                   if_save: bool = False,
                   ):
        """Predicting the result of train, valid and test dataset

        Arguments:
            data {np.ndarray} --
            epoch {int} --

        Keyword Arguments:
            if_save {bool} --  (default: {False})

        Returns:
            [type] -- [description]
        """
        if restored_model is not None:
            restored_path = os.path.join(self.log.dir_path,
                                         'optimal_model',
                                         'train_model_{:s}'.format(restored_model))
            self.log.restore_model(restored_path=restored_path)

        results_epoch = {}
        for tag in ['train', 'valid', 'test']:
            if '{:s} data'.format(tag) not in data:
                continue
            results_epoch[tag] = self.feedforward(data=data,
                                                  epoch=epoch + 1,
                                                  tag=tag,
                                                  if_save=if_save,
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
        """Feed the data into model and obtain the result

        Arguments:
            epoch {int} --
            data {np.ndarray} --

        Keyword Arguments:
            tag {str} -- 'train', 'valid' and 'test' (default: {'valid'})
            if_save {bool} --  (default: {True})
            show_info {bool} --  (default: {True})
            get_tensors {bool} -- (default: {True})

        Returns:
            [type] -- [description]
        """

        feed_dict = {self.input_placeholders[self.data_placeholder[key]]: data['{:s} {:s}'.format(tag, key)]
                     for key in self.data_placeholder}
        feed_dict[self.input_placeholders['training']] = False

        fetches = {'results': self.results,
                   'global_step': self.global_step,
                   }

        if get_tensors:
            fetches['tensors'] = self.tensors

        # Feedforward
        results = self.sess.run(fetches=fetches,
                                feed_dict=feed_dict,
                                )

        if get_tensors:
            tensors = results['tensors']

        results = results['results']

        self.log.write_log(res=results,
                           epoch=epoch,
                           log_type=tag,
                           if_save=if_save,
                           show_info=show_info)
        return results

    def get_parameters(self,
                       restored_model: str = None,
                       if_save: bool = False,
                       ):
        """Predicting the result of train, valid and test dataset

        Arguments:
            data {np.ndarray} --
            epoch {int} --

        Keyword Arguments:
            if_save {bool} --  (default: {False})

        Returns:
            [type] -- [description]
        """
        if restored_model is None:
            self.initialization()
            restored_model = 'random_initial'
        else:
            restored_path = os.path.join(self.log.dir_path,
                                         'optimal_model',
                                         'train_model_{:s}'.format(restored_model))
            self.log.restore_model(restored_path=restored_path)

        parameters = self.log.sess.run(self.trainable_pas)

        if if_save:
            save_path = os.path.join(self.log.dir_path,
                                     '{:s}_parameters.mat'.format(restored_model))
            sio.savemat(save_path, parameters)

        return parameters

    def load_data(self,
                  fold: h5py.Group or dict):
        """Loading data according to input placeholders

        Arguments:
            fold {h5py.Groupordict} -- [description]

        Raises:
            Warning: [description]

        Returns:
            [type] -- [description]
        """
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
    def load_dataset(scheme: str,
                     dataset: str,
                     hdf5_file_path: str):
        """Loading data in each fold from hdf5 file.

        Arguments:
            scheme {str} -- [Scheme name: 'CNNGLasso', 'BrainNetCNN', etc] (default: {None})
            hdf5_file_path {str} -- [The absolut path of hdf5 file] (default: {None})

        Returns:
            [type] -- [Dictionary of dataset. Key is 'fold 1', 'fold 2', etc]
        """

        data = collections.OrderedDict()
        hdf5 = hdf5_handler(hdf5_file_path)

        scheme_group = hdf5['{:s}/scheme {:s}'.format(dataset, scheme)]

        fold_list = list(scheme_group)
        fold_list.sort()
        for fold_name in fold_list:
            fold_dataset = {}
            for tag in scheme_group[fold_name]:
                fold_dataset[tag] = np.array(scheme_group[fold_name][tag])
            data[fold_name] = (fold_dataset)

        hdf5.close()
        return data
