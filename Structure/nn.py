import os
import sys
from Structure.utils_structure import get_metrics
from abc import ABCMeta, abstractmethod

import h5py
import numpy as np
import tensorflow as tf

from Log.log import Log
from Data.utils_prepare_data import vecter2onehot, extract_regions, select_top_significance_ROIs
from Structure.layers import get_layer_by_arguments
from Visualize.visualize import show_reconstruction
from Data.utils_prepare_data import create_dataset_hdf5
from Structure.parameters.xml_parse import parse_structure_parameters, parse_training_parameters


class AutoEncoder:
    model = dict()
    input_tensor = None
    encoder_tensor = None
    decoder_tensor = None
    mean_square_error = None
    learning_rate = None
    global_step = None
    minimizer = None

    def __init__(self, parameters, log):
        self.parameters = parameters
        self.log = log
        self.encoder = get_layer_by_arguments(arguments=self.parameters['encoder'])
        self.decoder = get_layer_by_arguments(arguments=self.parameters['decoder'])

    def build_encoder(self, input_tensor):
        with self.log.graph.as_default():
            # Build encoder
            self.input_tensor = input_tensor
            self.encoder_tensor = self.encoder(self.input_tensor)

    def build_decoder(self, input_tensor):
        with self.log.graph.as_default():
            # Build decoder
            output_shape = tf.shape(self.input_tensor)
            self.decoder_tensor = self.decoder(input_tensor=input_tensor, output_shape=output_shape)

    # def set_optimizer(self):
    #     if self.encoder_tensor is None or self.decoder_tensor is None:
    #         print('The encoder or decoder is None!')
    #         return
    #
    #     # mean_square_error function
    #     self.mean_square_error = tf.reduce_mean(tf.square(tf.subtract(self.input_tensor, self.decoder_tensor)))
    #
    #     # training
    #     self.lr_place = self.inputs['lr_place']
    #     self.global_step = tf.Variable(0, trainable=False)
    #     optimizer = tf.train.AdamOptimizer(self.lr_place)
    #     self.minimizer = optimizer.minimize(loss=self.mean_square_error,
    #                                         global_step=self.global_step)
    #
    #     print('Autoencoder Structure Initialized.')

    # def get_encoder(self, data, sess):
    #     encoder_data = sess.run(fetches=[self.encoder_tensor],
    #                             feed_dict={
    #                                 self.input_tensor: data
    #                             })
    #     return encoder_data


class NeuralNetwork(object, metaclass=ABCMeta):
    graph = None
    log = None
    sess = None

    structure = None
    optimizer = {}

    def __init__(self):
        pass

    @abstractmethod
    def build_structure(self, *args, **kwargs):
        raise NotImplementedError

    @abstractmethod
    def feedforward(self, *args, **kwargs):
        raise NotImplementedError

    @abstractmethod
    def backpropagation_epoch(self, *args, **kwargs):
        raise NotImplementedError

    @abstractmethod
    def backpropagation(self, *args, **kwargs):
        raise NotImplementedError

    @abstractmethod
    def pre_train_fold(self, *args, **kwargs):
        raise NotImplementedError

    @abstractmethod
    def fine_tune_fold(self, *args, **kwargs):
        raise NotImplementedError

    def build_optimizer(self, output_tensor, output_place, lr_place=tf.Tensor, if_acc: bool = False):
        results = dict()

        # Cost function
        metrics = get_metrics(output_tensor=output_tensor, output_place=output_place, if_acc=if_acc)
        results.update(metrics)

        square_error = tf.square(tf.subtract(x=output_place,
                                             y=output_tensor))
        square_errors = tf.reduce_mean(square_error, np.arange(1, len(square_error.get_shape())), name='square_errors')
        l2_loss = tf.reduce_sum([tf.reduce_sum(tf.square(tensor)) for tensor in self.structure['parameters']]) / 2
        results['L2 Penalty'] = l2_loss

        # build minimizer
        results['learning rate'] = lr_place
        optimizer = tf.train.AdamOptimizer(lr_place, name='optimizer')
        init_op_all = tf.all_variables()
        global_step = tf.Variable(initial_value=0, trainable=False, name='global_step')
        minimizer = optimizer.minimize(loss=results['MSE'] + 1e-3 * l2_loss,
                                       global_step=global_step,
                                       name='minimizer')

        # Initialize minimizer
        init_op_minimizer = tf.variables_initializer(set(tf.all_variables()) -
                                                     set(init_op_all))
        self.initialization(init_op_minimizer, name='minimizer')
        print('SAE Optimizer initialized.')
        self.optimizer.update(
            {'results': results,
             'output_place': output_place,
             'lr_place': lr_place,
             'minimizer': minimizer,
             'global_step': global_step,
             'square_errors': square_errors,
             })

        if if_acc:
            # Accuracy
            prediction = tf.argmax(output_tensor, 1)
            label = tf.argmax(output_place, 1)
            correct_prediction = tf.cast(x=tf.equal(prediction,
                                                    tf.argmax(output_place, 1)),
                                         dtype=tf.float32)
            self.optimizer['accuracies'] = correct_prediction

            prediction = tf.concat((tf.expand_dims(prediction, -1), tf.expand_dims(label, -1)), axis=-1)
            self.optimizer['prediction'] = prediction

    def initialization(self, init_op=None, name='', sess=None):
        if sess is None:
            sess = self.sess
        if init_op is None:
            init_op = self.init_op

        sess.run(init_op)
        print('Parameters {:s} initialized.'.format(name))

    def set_graph(self, log=None, graph=None, sub_folder_name=None):
        if log:
            self.log = log
        else:
            if graph is None:
                self.log = Log()

        if graph:
            self.graph = graph

        self.sess = self.log.sess

    def write_graph(self):
        self.log.write_graph()


class StackedConvolutionAutoEncoder(NeuralNetwork):
    autoencoders = list()
    inputs = dict()
    tensors = dict()
    parameters = list()
    results = dict()

    def __init__(self, log=None, graph=None, scheme: int = 1):
        self.set_graph(log=log, graph=graph)
        structure_xml_path = 'Structure/parameters/Scheme {:d}.xml'.format(scheme)
        self.stru_pa = parse_structure_parameters(structure_xml_path)['autoencoders']
        train_pa = parse_training_parameters('Structure/parameters/Training.xml')['autoencoders']
        self.pre_train_pa = train_pa['pre_train']
        self.fine_tune_pa = train_pa['fine_tune']

        with self.log.graph.as_default():
            init_op_all = tf.all_variables()
            for ae_pa in self.stru_pa['autoencoder']:
                self.autoencoders.append(AutoEncoder(parameters=ae_pa, log=self.log))
            for placeholder in self.stru_pa['input']:
                self.inputs[placeholder['scope']] = get_layer_by_arguments(arguments=placeholder)()

            self.log.saver = tf.train.Saver(tf.global_variables(), max_to_keep=1000, name='saver')
            self.init_op = tf.variables_initializer(set(tf.all_variables()) -
                                                    set(init_op_all))
            self.initialization(self.init_op, name='SCAE weights')

    def build_structure(self, train_index: list = None, optimizer: bool = True):
        if train_index is None:
            train_index = range(len(self.autoencoders))

        tensors = dict()
        parameters = list()

        with self.log.graph.as_default():
            feedforward_place = self.inputs['input']
            tensor = feedforward_place
            for feedforward_index in range(train_index[0]):
                self.autoencoders[feedforward_index].build_encoder(input_tensor=tensor)
                tensor = self.autoencoders[feedforward_index].encoder_tensor
            feedforward_tensor = tensor

            backpro_place = tf.placeholder(dtype=tf.float32,
                                           shape=feedforward_tensor.get_shape().as_list(),
                                           name='input_place')
            tensor = backpro_place
            for backward_index in train_index:
                autoencoder = self.autoencoders[backward_index]
                autoencoder.build_encoder(input_tensor=tensor)
                ae_tensors = autoencoder.encoder.tensors
                tensors[autoencoder.encoder.scope] = ae_tensors
                if 'weight' in ae_tensors:
                    parameters.append(ae_tensors['weight'])
                # if 'bias' in ae_tensors:
                #     parameters.append(ae_tensors['bias'])

                tensor = self.autoencoders[backward_index].encoder_tensor

            encoder_tensor = tensor

            for backward_index in reversed(train_index):
                autoencoder = self.autoencoders[backward_index]
                autoencoder.build_decoder(input_tensor=tensor)
                ae_tensors = autoencoder.decoder.tensors
                tensors[autoencoder.decoder.scope] = ae_tensors
                tensor = self.autoencoders[backward_index].decoder_tensor
                if 'weight' in ae_tensors:
                    parameters.append(ae_tensors['weight'])
                # if 'bias' in ae_tensors:
                #     parameters.append(ae_tensors['bias'])

            output_tensor = tensor
            self.structure = {'feedforward_place': feedforward_place,
                              'feedforward_tensor': feedforward_tensor,
                              'backpro_place': backpro_place,
                              'input_place': backpro_place,
                              'encoder_tensor': encoder_tensor,
                              'output_tensor': output_tensor,
                              'tensors': tensors,
                              'parameters': parameters
                              }
            print('Build Autoencoders')

            if optimizer:
                self.build_optimizer(output_tensor=self.structure['output_tensor'],
                                     output_place=self.structure['backpro_place'],
                                     lr_place=self.inputs['learning_rate'],
                                     )

    def build_classifier(self, subfolder_name: str = None, scheme: int = 4, tag: str = 'pre_train'):
        self.build_structure(optimizer=False)

        input_tensor = None
        if tag == 'fine_tune':
            input_tensor = self.output_tensor

        classifier = DeepNeuralNetwork(scheme=scheme,
                                       graph=self.graph,
                                       input_tensor=input_tensor,
                                       subfolder_name=subfolder_name,
                                       )
        return classifier

    def feedforward(self,
                    data: np.ndarray,
                    epoch: int = 0,
                    tag: str = 'Train',
                    if_print: bool = True,
                    if_save: bool = True,
                    ):
        data_size = np.size(data, 0)
        batch_size = self.pre_train_pa['train_batch_size']
        learning_rate = self.pre_train_pa['learning_rate']

        encoders = list()
        reconstructions = list()
        mses = list()

        steps = (data_size - 1) // batch_size + 1
        for step in range(steps):
            data_batch = data[step * batch_size: (step + 1) * batch_size]

            # Feedforward
            data_batch = self.sess.run(fetches=self.structure['feedforward_tensor'],
                                       feed_dict={
                                           self.structure['feedforward_place']: data_batch
                                       })

            results_batch, tensors_batch, recon_batch, mses_batch, encoder_batch, = \
                self.sess.run(fetches=[self.optimizer['results'],
                                       self.structure['tensors'],
                                       self.structure['output_tensor'],
                                       self.optimizer['square_errors'],
                                       self.structure['encoder_tensor'],
                                       ],
                              feed_dict={
                                  self.structure['backpro_place']: data_batch,
                                  self.optimizer['lr_place']: learning_rate,
                              })
            encoders.append(encoder_batch)
            reconstructions.append(recon_batch)
            mses.extend(mses_batch)

            if if_print:
                msg = '\rProcessing {:3d} of {:3d}  MSE: {:5e}'.format(step + 1, steps, np.mean(mses_batch))
                sys.stdout.write(msg)
        if if_print:
            results = {'MSE': np.mean(mses)}
            self.log.write_log(res=results,
                               epoch=epoch,
                               log_type=tag,
                               if_print=if_print,
                               if_save=if_save,
                               new_line=True,
                               )

        encoder = np.concatenate(encoders, 0)
        reconstruction = np.concatenate(reconstructions, 0)
        return data, encoder, reconstruction, mses

    def backpropagation_epoch(self, data, epoch, pas):
        mses = list()

        # Shuffle
        train_data_size = np.size(data, axis=0)
        random_index = np.random.permutation(train_data_size)
        train_data = data[random_index]

        # Start training
        batch_size = pas['train_batch_size']
        learning_rate = pas['learning_rate'] * pas['decay_rate'] ** np.floor(epoch / pas['decay_step'])
        train_steps = (train_data_size - 1) // batch_size + 1
        for train_step in range(train_steps):
            train_data_batch = train_data[train_step * batch_size: (train_step + 1) * batch_size]

            # Feedforward
            train_data_batch = self.sess.run(fetches=self.structure['feedforward_tensor'],
                                             feed_dict={
                                                 self.structure['feedforward_place']: train_data_batch
                                             })

            # Backpropagation
            results_batch, _, mses_batch, global_step, = \
                self.sess.run(fetches=[self.optimizer['results'],
                                       self.optimizer['minimizer'],
                                       self.optimizer['square_errors'],
                                       self.optimizer['global_step'],
                                       ],
                              feed_dict={
                                  self.structure['backpro_place']: train_data_batch,
                                  self.optimizer['lr_place']: learning_rate,
                              })

            mses.extend(mses_batch)

            message = '{:4d}/{:d}\t'.format(train_step + 1, train_steps)
            self.log.write_log(res=results_batch, epoch=global_step, pre_fix=message)

        print()
        results = {'MSE': np.mean(mses_batch)}
        self.log.write_log(res=results,
                           epoch=epoch,
                           )

    def backpropagation(self,
                        data: h5py.Group or dict,
                        train_pa: dict,
                        show_flag: bool = False,
                        start_epoch: int = 0,
                        ) -> str:
        """

        :param data: Dictionary which has key: train data, valid data, test data
        :param show_flag:
        :param start_epoch:
        :param train_pa
        :return:
        """
        self.write_graph()
        save_path = None

        training_parameters = train_pa
        for epoch in np.arange(start=start_epoch, stop=training_parameters['training_cycle']):
            if show_flag:
                show_reconstruction(data=data['valid data'],
                                    model=self,
                                    title='Epoch {:d}'.format(epoch + 1),
                                    subject_num=1,
                                    )

            print('Epoch: {:d}'.format(epoch + 1))

            # Train
            self.backpropagation_epoch(data=data['train data'],
                                       epoch=epoch,
                                       pas=training_parameters,
                                       )

            if (epoch + 1) % training_parameters['test_cycle'] == 0:
                # Valid
                if 'valid data' in data:
                    self.feedforward(data=data['valid data'],
                                     epoch=epoch,
                                     tag='Valid')

                # Test
                if 'test data' in data:
                    self.feedforward(data=data['test data'],
                                     epoch=epoch,
                                     tag='Test')

            # Save
            if (epoch + 1) % training_parameters['save_cycle'] == 0:
                save_path = self.log.save_model(epoch=epoch + 1)

        return save_path

    def pre_train_fold(self,
                       fold: h5py.Group,
                       train_indexes: list = None,
                       start_index: list = None,
                       ) -> str:
        if not isinstance(fold, h5py.Group):
            raise TypeError('The fold must be type of h5py.Group.')

        save_path = None

        if train_indexes is None:
            train_indexes = [[0, 1], [2, 3]]

        data = {'train data': np.array(fold['pre train data'])}

        for train_index in train_indexes:
            if start_index is not None and train_index != start_index:
                continue

            self.build_structure(train_index=train_index)
            start_epoch = self.log.restore()

            # set subfolder name such as 'fold 1/pre_train_SCAE/0-1'
            subfolder_name = '{:s}/pre_train_SCAE/{:s}'.format(
                fold.name.split('/')[-1], '-'.join([str(i) for i in train_index])
            )
            self.log.set_filepath_by_subfolder(subfolder_name=subfolder_name)
            show_flag = True if 0 in train_index else False
            save_path = self.backpropagation(data=data,
                                             start_epoch=start_epoch,
                                             show_flag=False,
                                             train_pa=self.pre_train_pa)

            restore_path = None
            start_index = None

        return save_path

    def fine_tune_fold(self, fold: h5py.Group) -> str:
        if not isinstance(fold, h5py.Group):
            raise TypeError('The fold must be type of h5py.Group.')

        data = {'train data': np.array(fold['train data']),
                'valid data': np.array(fold['valid data']),
                'test data': np.array(fold['test data']),
                }

        self.build_structure()
        start_epoch = self.log.restore()

        # set subfolder name
        fold_name = fold.name.split('/')[-1]
        index_str = 'fine_tune_SCAE'
        subfolder_name = '{:s}/{:s}'.format(fold_name, index_str)
        self.log.set_filepath_by_subfolder(subfolder_name=subfolder_name)

        save_path = self.backpropagation(data=data,
                                         start_epoch=start_epoch,
                                         train_pa=self.fine_tune_pa)
        return save_path

    def encode_folds(self, folds, save_dir: str = None, save_path: str = None):
        for fold_idx in np.arange(start=1, stop=6):
            if save_dir:
                save_path = os.path.join(save_dir, '{:s}/fine_tune_SCAE/model/train.model_300'.format(fold_idx))
            self.encode_fold(fold=folds['fold {:d}'.format(fold_idx)], save_path=save_path)

    def encode_fold(self, fold: h5py.Group, save_path: str = None, num_split: int = None):
        self.build_structure()
        self.log.restore()

        for tag in ['train', 'valid', 'test']:
            data_tag = '{:s} data'.format(tag)

            try:
                data_tmp = np.array(fold[data_tag])
            except KeyError as e:
                print(e)
                continue

            _, encoder, reconstruction, mses = self.feedforward(data=data_tmp, if_save=False)

            # reshape
            batch_size = np.shape(encoder)[0]
            new_shape = [batch_size // num_split, num_split, -1] if num_split else [batch_size, -1]
            encoder = np.reshape(encoder, newshape=new_shape)

            # save to hdf5 file
            create_dataset_hdf5(group=fold,
                                name='{:s} output'.format(data_tag),
                                data=reconstruction,
                                )
            create_dataset_hdf5(group=fold,
                                name='{:s} encoder'.format(data_tag),
                                data=encoder,
                                )

    def train_folds(self, folds: h5py.Group,
                    pre_train: bool = True,
                    fine_tune: bool = True,
                    train_indexes: list = None,
                    restored_path: str = None,
                    fold_indexes: list = None,
                    ):
        if fold_indexes is None:
            fold_indexes = range(5)

        for fold_index in fold_indexes:
            fold = folds[list(folds.keys())[fold_index]]
            if pre_train:
                self.pre_train_fold(fold=fold,
                                    train_indexes=train_indexes)
            if fine_tune:
                save_path = self.fine_tune_fold(fold=fold,
                                                restored_path=restored_path,
                                                pre_train=pre_train)
        return save_path


class DeepNeuralNetwork(NeuralNetwork):
    log = None
    graph = None
    sess = None

    input_layers = {}
    op_layers = []
    inputs = {}

    def __init__(self,
                 graph: tf.Graph = None,
                 log: Log = None,
                 subfolder_name: str = None,
                 scheme: int = 1,
                 ):
        self.set_graph(log=log, graph=graph, sub_folder_name=subfolder_name)
        structure_xml_path = 'Structure/parameters/Scheme {:d}.xml'.format(scheme)
        self.str_pa = parse_structure_parameters(structure_xml_path)['classifier']
        train_pa = parse_training_parameters('Structure/parameters/Training.xml')['classifier']
        self.pre_train_pa = train_pa['pre_train']
        self.fine_tune_pa = train_pa['fine_tune']
        with self.log.graph.as_default():
            init_op_all = tf.all_variables()
            for placeholder in self.str_pa['input']:
                layer = get_layer_by_arguments(arguments=placeholder)
                self.input_layers[placeholder['scope']] = layer
                self.inputs[placeholder['scope']] = layer()

            for layer_pa in self.str_pa['layers']:
                layer = get_layer_by_arguments(arguments=layer_pa)
                self.op_layers.append(layer)

            self.log.saver = tf.train.Saver(tf.global_variables(), max_to_keep=1000, name='saver')
            self.init_op = tf.variables_initializer(set(tf.all_variables()) -
                                                    set(init_op_all))
            self.initialization(self.init_op, name='DNN weights')

    def build_structure(self, input_tensor: tf.Tensor = None, input_place: tf.Tensor = None):
        if input_tensor is not None:
            if input_place is None:
                raise TypeError('Input placeholder should not be None when input tensor is not None.')
            tensor = input_tensor
        else:
            input_place = self.inputs['input']
            tensor = input_place

        tensors = dict()
        parameters = list()
        for layer in self.op_layers:
            tensor = layer(tensor)
            layer_tensor = layer.tensors
            tensors[layer.scope] = layer_tensor
            if 'weight' in layer_tensor:
                parameters.append(layer_tensor['weight'])
            # if 'bias' in layer_tensor:
            #     parameters.append(layer_tensor['bias'])

        output_tensor = tensor

        self.structure = {'input_place': input_place,
                          'output_tensor': output_tensor,
                          'tensors': tensors,
                          'parameters': parameters,
                          }
        print('Build Deep Neural Network.')
        self.build_optimizer(output_tensor=output_tensor,
                             output_place=self.inputs['output'],
                             lr_place=self.inputs['learning_rate'],
                             if_acc=True,
                             )

    def backpropagation_epoch(self, data: np.ndarray,
                              label: np.ndarray,
                              train_pa: dict,
                              epoch: int,
                              regions: int or list = None):
        # Shuffle
        data_label = list(zip(data, label))
        np.random.shuffle(data_label)
        data, label = zip(*data_label)
        data = np.array(data)
        label = np.array(label)

        # Start training
        batch_size = train_pa['train_batch_size']
        learning_rate = train_pa['learning_rate']
        learning_rate = learning_rate * train_pa['decay_rate'] ** np.floor(epoch / train_pa['decay_step'])

        mses = list()
        accs = list()
        train_data_size = np.size(data, axis=0)
        train_steps = (train_data_size - 1) // batch_size + 1
        for train_step in range(train_steps):
            train_data_batch = data[train_step * batch_size: (train_step + 1) * batch_size]
            train_label_batch = label[train_step * batch_size: (train_step + 1) * batch_size]

            feed_dict = {}
            input_placeholder = self.structure['input_place']
            if isinstance(input_placeholder, list):
                train_data_batch_regions = extract_regions(data=train_data_batch, regions=regions)
                if len(train_data_batch_regions) != len(input_placeholder):
                    raise TypeError('Length doesn\'t match!')
                for train_data_batch_region, input_place in zip(train_data_batch_regions, input_placeholder):
                    feed_dict[input_place] = train_data_batch_region
            else:
                feed_dict[input_placeholder] = train_data_batch

            feed_dict[self.optimizer['output_place']] = train_label_batch
            feed_dict[self.optimizer['lr_place']] = learning_rate

            # Backpropagation
            results_batch, _, predict, mses_batch, global_step, accs_batch, = \
                self.sess.run(fetches=[self.optimizer['results'],
                                       self.optimizer['minimizer'],
                                       self.optimizer['prediction'],
                                       self.optimizer['square_errors'],
                                       self.optimizer['global_step'],
                                       self.optimizer['accuracies'],
                                       ],
                              feed_dict=feed_dict,
                              )

            mses.extend(mses_batch)
            accs.extend(accs_batch)
            message = '{:4d}/{:d}\t'.format(train_step + 1, train_steps)
            self.log.write_log(res=results_batch, epoch=global_step, pre_fix=message)
        results = {'MSE': np.mean(mses),
                   'Accuracy': np.mean(accs),
                   }
        self.log.write_log(res=results, epoch=epoch, new_line=True)

    def feedforward(self,
                    data: np.ndarray,
                    label: np.ndarray,
                    epoch: int,
                    tag: str = 'Valid',
                    regions: list = None):
        batch_size = self.pre_train_pa['train_batch_size']
        learning_rate = self.pre_train_pa['learning_rate'] * self.pre_train_pa['decay_rate'] ** np.floor(
            epoch / self.pre_train_pa['decay_step'])

        mses = list()
        accs = list()
        train_data_size = np.size(data, axis=0)
        train_steps = (train_data_size - 1) // batch_size + 1
        for train_step in range(train_steps):
            train_data_batch = data[train_step * batch_size: (train_step + 1) * batch_size]
            train_label_batch = label[train_step * batch_size: (train_step + 1) * batch_size]

            feed_dict = {}
            input_placeholder = self.structure['input_place']
            if isinstance(input_placeholder, list):
                train_data_batch_regions = extract_regions(data=train_data_batch, regions=regions)
                if len(train_data_batch_regions) != len(input_placeholder):
                    raise TypeError('Length doesn\'t match!')

                for train_data_batch_region, input_place in zip(train_data_batch_regions, input_placeholder):
                    feed_dict[input_place] = train_data_batch_region
            else:
                feed_dict[input_placeholder] = train_data_batch

            feed_dict[self.optimizer['output_place']] = train_label_batch
            feed_dict[self.optimizer['lr_place']] = learning_rate

            # Backpropagation
            results_batch, predict, mses_batch, accs_batch, tensors, = \
                self.sess.run(fetches=[self.optimizer['results'],
                                       self.optimizer['prediction'],
                                       self.optimizer['square_errors'],
                                       self.optimizer['accuracies'],
                                       self.structure['tensors'],
                                       ],
                              feed_dict=feed_dict,
                              )

            mses.extend(mses_batch)
            accs.extend(accs_batch)
        results = {'MSE': np.mean(mses),
                   'Accuracy': np.mean(accs),
                   }
        self.log.write_log(res=results, epoch=epoch, log_type=tag, new_line=True)

    def backpropagation(self, data: dict,
                        train_pa: dict,
                        start_epoch: int = 0,
                        regions: int or list = None,
                        ):
        self.write_graph()
        save_path = None
        if regions is None:
            dataset = self.input_layers['input'].dataset
            regions = select_top_significance_ROIs(datasets=[dataset])[dataset]

        for epoch in np.arange(start=start_epoch, stop=train_pa['training_cycle']):
            self.backpropagation_epoch(data=data['train data'],
                                       label=data['train label'],
                                       train_pa=train_pa,
                                       epoch=epoch + 1,
                                       regions=regions,
                                       )

            if (epoch + 1) % train_pa['test_cycle'] == 0:
                # Valid
                self.feedforward(data=data['valid data'],
                                 label=data['valid label'],
                                 epoch=epoch + 1,
                                 tag='Valid',
                                 regions=regions,
                                 )

                # Test
                self.feedforward(data=data['test data'],
                                 label=data['test label'],
                                 epoch=epoch + 1,
                                 tag='Test',
                                 regions=regions,
                                 )

            if (epoch + 1) % train_pa['save_cycle'] == 0:
                save_path = self.log.save_model(epoch=epoch + 1)

        return save_path

    def pre_train_fold(self,
                       fold: h5py.Group,
                       regions: int or list = None, ):
        if not isinstance(fold, h5py.Group):
            raise TypeError('The fold must be type of h5py.Group.')

        self.build_structure()
        start_epoch = self.log.restore()

        data = {'train data': np.array(fold['train data']),
                'train label': vecter2onehot(np.array(fold['train label'])),
                'valid data': np.array(fold['valid data']),
                'valid label': vecter2onehot(np.array(fold['valid label'])),
                'test data': np.array(fold['test data']),
                'test label': vecter2onehot(np.array(fold['test label'])),
                }

        # set subfolder name
        fold_name = fold.name.split('/')[-1]
        index_str = 'pre_train_Classifier'
        subfolder_name = '{:s}/{:s}'.format(fold_name, index_str)
        self.log.set_filepath_by_subfolder(subfolder_name=subfolder_name)

        save_path = self.backpropagation(data=data,
                                         start_epoch=start_epoch,
                                         train_pa=self.pre_train_pa,
                                         regions=regions,
                                         )
        return save_path

    def fine_tune_fold(self,
                       fold: h5py.Group,
                       input_tensor: tf.Tensor,
                       input_place: tf.Tensor,
                       ):
        if not isinstance(fold, h5py.Group):
            raise TypeError('The fold must be type of h5py.Group.')

        self.build_structure(input_tensor=input_tensor, input_place=input_place)
        start_epoch = self.log.restore()

        data = {'train data': np.array(fold['train data']),
                'train label': vecter2onehot(np.array(fold['train label'])),
                'valid data': np.array(fold['valid data']),
                'valid label': vecter2onehot(np.array(fold['valid label'])),
                'test data': np.array(fold['test data']),
                'test label': vecter2onehot(np.array(fold['test label'])),
                }

        # set subfolder name
        fold_name = fold.name.split('/')[-1]
        index_str = 'fine_tune_Classifier'
        subfolder_name = '{:s}/{:s}'.format(fold_name, index_str)
        self.log.set_filepath_by_subfolder(subfolder_name=subfolder_name)

        save_path = self.backpropagation(data=data,
                                         start_epoch=start_epoch,
                                         train_pa=self.fine_tune_pa,
                                         )
        return save_path
