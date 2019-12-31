import os

import h5py
import numpy as np
import scipy.io as sio
import tensorflow as tf

from Analyse.visualize import save_or_exhibit
from Dataset.utils import vector2onehot
from Log.log import Log
from Model.NN import NeuralNetwork
from Model.utils_model import EarlyStop, get_metrics, get_weights, save_weights


class CNNGraphicalLasso(NeuralNetwork):

    def __init__(self,
                 dir_path: str,
                 log: Log = None,
                 scheme: int or str = 1,
                 graph: tf.Graph = None,
                 spe_pas: dict = None, 
                 ):
        self.log = None
        self.graph = None
        self.sess = None
        self.op_layers = []
        self.structure = {}
        self.input_placeholders = {}
        self.minimizer_SICE = None

        tf.get_default_graph().clear_collection('SICE_loss')
        NeuralNetwork.__init__(self,
                               log=log,
                               scheme=scheme,
                               graph=graph,
                               dir_path=dir_path,
                               spe_pas=spe_pas, 
                               )
        self.optimizer = {}
        self.NN_type = 'GraphNN'

    def build_optimizer(self, output_tensor,
                        penalties: list = []):
        lr_place = self.input_placeholders['learning_rate']
        output_place = self.input_placeholders['output_tensor']
        # Cost function
        self.results = get_metrics(output_tensor=output_tensor,
                                   ground_truth=output_place,
                                   )
        self.prediction = tf.nn.softmax(output_tensor)

        # Build loss function, which contains the cross entropy and regularizers.
        self.results['Cost'] = self.results['Cross Entropy']

        penalties.extend(['L1', 'L2', 'SICE'])
        coefficients = [1, 0.5, self.pa['basic']['SICE_lambda']]
        for regularizer, coef in zip(penalties, coefficients):
            loss_name = '{:s}_loss'.format(regularizer)
            loss_collection = tf.get_collection(loss_name)
            loss = tf.Variable(0.0, trainable=False)
            if len(loss_collection) > 0:
                loss = tf.add_n(loss_collection)
            self.results['{:s} Penalty'.format(regularizer)] = loss
            self.results['Cost'] += coef * loss

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
                if 'SICE Penalty' in self.results:
                    self.minimizer_SICE = optimizer.minimize(self.results['SICE Penalty'],
                                                             global_step=self.global_step,
                                                             name='minimizer_SICE',
                                                             )
                else:
                    self.minimizer_SICE = None

    def training(self,
                 data: h5py.Group or dict,
                 run_time: int = 1,
                 fold_index: int = None,
                 restored_path: str = None,
                 show_info: bool = True,
                 ):
        with self.graph.as_default():
            self.build_structure()

        data = self.load_data(data)

        self.pre_training(data=data)
        early_stop = self.backpropagation(data=data)
        self.discriminant_power_analyse(run_time=run_time, fold_index=fold_index)
        early_stop.show_results(run_time=run_time, fold_index=fold_index)
        early_stop.clear_models()

        return early_stop.results

    def pre_training(self,
                     data: dict):
        batch_size = self.pa['training']['train_batch_size']
        pre_learning_rate = self.pa['training']['pre_learning_rate']
        pre_training_cycle = self.pa['training']['pre_training_cycle']

        supervised_data = {'train data': data['train data'],
                           'train label': data['train label']}
        unsupervised_data = {'train data': data['valid data'],
                             'train label': data['valid label']}

        for i in range(pre_training_cycle):
            print('Pre training epoch: {:d}'.format(i + 1))
            self.backpropagation_epoch(
                data=supervised_data,
                batch_size=batch_size,
                learning_rate=pre_learning_rate,
                training=True,
                minimizer=self.minimizer_SICE,
            )
            self.backpropagation_epoch(
                data=unsupervised_data,
                batch_size=batch_size,
                learning_rate=pre_learning_rate,
                training=False,
                minimizer=self.minimizer_SICE,
            )

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
            if self.pa['training']['SICE_training']:            
                supervised_data = {'train data': data['train data'],
                                'train label': data['train label']}
                unsupervised_data = {'train data': data['valid data'],
                                    'train label': data['valid label']}
                self.backpropagation_epoch(
                    data=supervised_data,
                    batch_size=batch_size,
                    learning_rate=early_stop.learning_rate,
                    training=True,
                    minimizer=self.minimizer_SICE,
                )
                self.backpropagation_epoch(
                    data=unsupervised_data,
                    batch_size=batch_size,
                    learning_rate=early_stop.learning_rate,
                    training=False,
                    minimizer=self.minimizer_SICE,
                )

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

    def training_(self,
                  data: h5py.Group or dict,
                  run_time: int = 1,
                  fold_index: int = 1,
                  restored_path: str = None,
                  show_info: bool = True,
                  ):
        with self.graph.as_default():
            self.build_structure()

        data = self.load_data(data)

        results = self.backpropagation(data=data,
                                       fold_index=fold_index,
                                       run_time=run_time,
                                       start_epoch=start_epoch,
                                       show_info=show_info,
                                       )

        return results

    def backpropagation_(self,
                         data: dict,
                         fold_index: int = 1,
                         run_time: int = 1,
                         start_epoch: int = 0,
                         regions: int or list = None,
                         show_info: bool = True,
                         ):
        self.log.write_graph()
        train_pa = self.pa['training']
        train_pa['SICE_training'] = self.pa['basic']['SICE_training']
        early_stop_pa = self.pa['early_stop']

        early_stop_pa.update({'epoch': start_epoch,
                              'decay_count': 0,
                              'tolerant_count': 0,
                              'training_cycle': train_pa['training_cycle'],
                              'learning_rate': early_stop_pa['learning_rate']
                              if 'learning_rate' in early_stop_pa else early_stop_pa['learning_rates'][0],
                              'stage': 0,
                              })

        data_unsupervise = np.concatenate((data['valid data'], data['test data']), axis=0)
        label_unsupervise = np.concatenate((data['valid label'], data['test label']), axis=0)
        extra_data_unsupervise = np.concatenate((data['valid covariance'], data['test covariance']), axis=0)
        extra_feed_unsupervise = {self.input_placeholders['sample_covariance']: extra_data_unsupervise}

        self.save_GLasso_weights_to_figure(run_time=run_time,
                                           prefix='CNNWithGLasso initialize',
                                           # if_show=True,
                                           # if_save=True,
                                           )

        for i in range(train_pa['pretraining_cycle']):
            print('Pre training epoch: {:d}'.format(i + 1))
            self.backpropagation_epoch(
                epoch=i + 1,
                data=data,
                learning_rate=early_stop_pa['pre_learning_rate'],
                train_pa=train_pa,
                training=True,
                save_info=False,
                show_info=False,
                get_tensors=False,
                minimizer=self.optimizer['minimizer_SICE'],
            )
            self.backpropagation_epoch(
                data=data,
                learning_rate=early_stop_pa['pre_learning_rate'],
                train_pa=train_pa,
                training=False,
                save_info=False,
                show_info=False,
                get_tensors=False,
                minimizer=self.optimizer['minimizer_SICE'],
                epoch=i + 1,
            )

            # if (i + 1) % 20 == 0:
            #     self.analyse_weight(run_time=run_time,
            #                         fold_index=fold_index,
            #                         prefix='CNNWithGLasso pretraining',
            #                         epoch=i + 1,
            #                         save=True,
            #                         )
        if train_pa['pretraining_cycle'] > 0:
            self.save_GLasso_weights_to_figure(run_time=run_time,
                                               prefix='CNNWithGLasso pre trained',
                                               # if_save=True,
                                               if_show=True,
                                               )

        extra_feed_train = {self.input_placeholders['sample_covariance']: data['train covariance']}
        extra_feed_valid = {self.input_placeholders['sample_covariance']: data['valid covariance']}
        extra_feed_test = {self.input_placeholders['sample_covariance']: data['test covariance']}
        epoch = early_stop_pa['epoch']
        while epoch < train_pa['training_cycle']:
            if train_pa['SICE_training']:
                self.backpropagation_epoch(
                    data=data,
                    learning_rate=early_stop_pa['learning_rate_SICE'] if 'learning_rate_SICE' in early_stop_pa else
                    early_stop_pa['learning_rate'],
                    train_pa=train_pa,
                    training=True,
                    save_info=False,
                    show_info=False,
                    minimizer=self.optimizer['minimizer_SICE'],
                    epoch=epoch + 1,
                    feed_dict_extra=extra_feed_train,
                )
                self.backpropagation_epoch(
                    data=data,
                    learning_rate=early_stop_pa['learning_rate'],
                    train_pa=train_pa,
                    training=True,
                    save_info=False,
                    show_info=False,
                    minimizer=self.optimizer['minimizer_SICE'],
                    feed_dict_extra=extra_feed_unsupervise,
                    epoch=epoch + 1
                )

            results_train = self.backpropagation_epoch(data=data,
                                                       learning_rate=early_stop_pa['learning_rate'],
                                                       train_pa=train_pa,
                                                       epoch=epoch + 1,
                                                       get_tensors=True,
                                                       show_info=False,
                                                       feed_dict_extra=extra_feed_train,
                                                       )

            if (epoch + 1) % train_pa['test_cycle'] == 0:
                # Valid
                results_valid = self.feedforward(data=data,
                                                 epoch=epoch + 1,
                                                 tag='Valid',
                                                 get_tensors=False,
                                                 show_info=False,
                                                 )
                # Test
                results_test = self.feedforward(data=data,
                                                epoch=epoch + 1,
                                                tag='Test',
                                                get_tensors=False,
                                                show_info=False,
                                                )

                if show_info:
                    print(
                        'Epoch: {:d}\t'
                        # 'Norm: {:f}\t'
                        # 'SICE Penalty: {:5e}\r\n\t\t'
                        'Train cost: {:f}\taccuracy: {:f}\t'
                        'Valid cost: {:f}\taccuracy: {:f}\t'
                        'Test cost: {:f}\taccuracy: {:f}'.format(
                            epoch + 1,
                            # np.mean(result['tensors']['E2NGLasso1']['Norm 1']),
                            # np.mean(result['results']['SICE Penalty']),
                            results_train['Cross Entropy'],
                            results_train['Accuracy'],
                            results_valid['Cross Entropy'],
                            results_valid['Accuracy'],
                            results_test['Cross Entropy'],
                            results_test['Accuracy']))

                early_stop_pa = early_stop(log=self.log,
                                           epoch=epoch,
                                           pa=early_stop_pa,
                                           results_train=results_train,
                                           results_valid=results_valid,
                                           results_test=results_test,
                                           )
                epoch = early_stop_pa['epoch']
            # if (epoch + 1) % 20 == 0:
            #     self.analyse_weight(run_time=run_time,
            #                         fold_index=fold_index,
            #                         prefix='CNNWithGLasso training',
            #                         epoch=epoch + 1,
            #                         save=True,
            #                         )

        self.save_GLasso_weights_to_figure(run_time=run_time,
                                           weight_names=['weight_SICE', 'weight_SICE_bn', 'weight', 'weight_multiply'],
                                           prefix='CNNWithGLasso trained',
                                           # if_save=True,
                                           if_show=True,
                                           )
        save_weights(self.op_layers, self.sess)
        # self.discriminant_power_analyse(run_time=run_time, fold_index=fold_index)

        return early_stop_pa['results'], early_stop_pa['max_epoch']

    def discriminant_power_analyse(self,
                                   run_time: int = 1,
                                   fold_index: int = 1):
        layers = self.op_layers
        weight_dict = {}
        for layer in layers:
            weights = {}
            for tensors_name in layer.tensors:
                if 'weight' in tensors_name:
                    weights[tensors_name] = layer.tensors[tensors_name]
            weight_dict[layer.pa['scope']] = weights
        weights = self.sess.run(weight_dict)

        F = np.array([1, -1])
        for layer_scope in ['hidden2', 'hidden1']:
            weight = weights[layer_scope]['weight']
            F = np.matmul(weight, F)

        # N2G
        weight = weights['N2G1']['weight']
        F = np.multiply(weight, F)
        F = np.squeeze(np.sum(np.absolute(F), axis=-1))

        # E2N
        weight = weights['E2N1']['weight_multiply']
        F = np.sum(np.abs(np.multiply(np.squeeze(weight), F)), axis=-1)

        weight_names = ['weight', 'weight_SICE', 'weight_SICE_bn', 'weight_multiply']
        results = {weight_name: np.squeeze(weights['E2N1'][weight_name]) for weight_name in weight_names}
        results['F'] = F

        # save_dir_path = 'F:/OneDriveOffL/Data/Result/Net'
        save_path = os.path.join(self.log.dir_path, 'parameters.mat')

        sio.savemat(save_path, results)

    def save_GLasso_weights_to_figure(self,
                                      run_time: int = 0,
                                      epoch: int = None,
                                      weight_names: list = None,
                                      if_show: bool = False,
                                      if_save: bool = False,
                                      if_absolute: bool = False,
                                      prefix: str = None,
                                      ):
        """
        Save or exhibit the weights in node-to-edge layer
        :param run_time: Run time
        :param epoch: Epoch
        :param weight_names: The names of the weights need to be save or exhibit
        :param if_save: The flag of whether saving the figure to file
        :param if_show: The flag of whether showing the weights in platform
        :param if_absolute: The flag of whether save or exhibit absolute of the weights
        :param prefix: The prefix of saving file
        :return:
        """
        if not (if_save or if_show):
            return

        save_dir_path = 'F:/OneDriveOffL/Data/Result/Net'

        weights = get_weights(self.op_layers, self.sess)['E2N1']
        if weight_names is None:
            weight_names = ['weight_SICE_bn', 'weight_SICE']

        for weight_name in weight_names:
            try:
                weight = weights[weight_name]
            except KeyError:
                print('Warning: {:} is not in the list {:}'.format(weight_name, weight_names))
                continue

            save_path = os.path.join(save_dir_path, 'time {:d}'.format(run_time))

            prefix_tmp = '{:s} {:}{:}'.format(prefix,
                                              weight_name.replace('_', ' '),
                                              ' epoch {:d}'.format(epoch + 1) if epoch is not None else '',
                                              )

            save_or_exhibit(weight=weight,
                            prefix=prefix_tmp,
                            save_path=save_path,
                            if_show=if_show,
                            if_save=if_save,
                            if_absolute=if_absolute,
                            )
