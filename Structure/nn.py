import os
import sys

from Structure.DeepNerualNetwork.AutoEncoder import AutoEncoder
from Structure.DeepNerualNetwork.DeepNeuralNetwork import DeepNeuralNetwork
from Structure.DeepNerualNetwork.NeuralNetwork import NeuralNetwork

import h5py
import numpy as np
import tensorflow as tf

from Structure.Layer.LayerConstruct import get_layer_by_arguments
from Analyse.visualize import show_reconstruction
from data.utils_prepare_data import create_dataset_hdf5


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
        self.train_pa['pre_train'] = train_pa['pre_train']
        self.train_pa['fine_tune'] = train_pa['fine_tune']

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
        batch_size = self.train_pa['pre_train']['train_batch_size']
        learning_rate = self.train_pa['pre_train']['learning_rate']

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
                               show_info=if_print,
                               if_save=if_save,
                               new_line=True,
                               )
            print()

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

    def train_fold(self,
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
                                             train_pa=self.train_pa['pre_train'])

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
                                         train_pa=self.train_pa['fine_tune'])
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
                self.train_fold(fold=fold,
                                train_indexes=train_indexes)
            if fine_tune:
                save_path = self.fine_tune_fold(fold=fold,
                                                restored_path=restored_path,
                                                pre_train=pre_train)
        return save_path


