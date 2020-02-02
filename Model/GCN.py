import os

import h5py
import numpy as np
import scipy.io as sio
import tensorflow as tf

from Analyse.visualize import save_or_exhibit
from Log.log import Log
from Model.early_stop import EarlyStop
from Model.NN import NeuralNetwork
from Model.utils_model import get_metrics, get_weights, save_weights


class GraphConvolutionNetwork(NeuralNetwork):
    """Convolutional neural network with graphical Lasso.

    Arguments:
        NeuralNetwork {[type]} -- [description]

    Returns:
        [type] -- [description]
    """

    model_type = 'Convolutional neural network with graphical Lasso'

    def __init__(self,
                 dir_path: str,
                 log: Log = None,
                 scheme: int or str = 1,
                 spe_pas: dict = None,
                 ):
        NeuralNetwork.__init__(self,
                               log=log,
                               scheme=scheme,
                               dir_path=dir_path,
                               spe_pas=spe_pas,
                               )

        self.minimizer_SICE = None
        self.data_placeholder = {'covariance': 'covariance_tensor',
                                 'label': 'output_tensor'}

    def build_optimizer(self, output_tensor,
                        penalties: list = None):
        """Build optimizer of this model, which includes Cross Entropy loss and SICE loss.

        Arguments:
            output_tensor {[type]} -- Output or predicting of the neural network

        Keyword Arguments:
            penalties {list} -- [description] (default: {None})
        """
        lr_place = self.input_placeholders['learning_rate']
        output_place = self.input_placeholders['output_tensor']
        # Cost function
        self.results = get_metrics(output_tensor=output_tensor,
                                   ground_truth=output_place,
                                   )
        self.prediction = tf.nn.softmax(output_tensor)

        # Build loss function, which contains the cross entropy and regularizers.

        if penalties is None:
            penalties = ['L2', 'SICE']
        coefficients = [0.5, self.pas['basic']['SICE_lambda']]

        for regularizer, coef in zip(penalties, coefficients):
            loss_name = '{:s}_loss'.format(regularizer)
            loss_collection = tf.get_collection(loss_name)
            loss = tf.Variable(0.0, trainable=False)
            if len(loss_collection) > 0:
                loss = tf.add_n(loss_collection)
            self.results['{:s} Penalty'.format(regularizer)] = loss * coef
        self.results['Cost'] = self.results['Cross Entropy'] + \
            self.results['L2 Penalty']

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
                if 'SICE Penalty' in self.results:
                    self.minimizer_SICE = optimizer.minimize(self.results['SICE Penalty'],
                                                             global_step=self.global_step,
                                                             name='minimizer_SICE',
                                                             )

    def training(self,
                 data: dict,
                 run_time: int = 1,
                 fold_name: str = None,
                 if_show: bool = True,
                 ):
        """The main training process of the neural network, which includes: 1. build structure. 
        2. write the architecture to file. 3. Pretraining. 4. backpropagation 5. show results if permitted. 
        6. save first, optimal and final model in the training process. 

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

        data = self.load_data(data)

        if self.pas['basic']['SICE_training']:
            self.pre_training(data=data)

        early_stop = self.backpropagation(data=data)
        self.discriminant_power_analyse()

        if if_show:
            early_stop.show_results(run_time=run_time, fold_name=fold_name)

        early_stop.clear_models()
        return early_stop.results

    def pre_training(self,
                     data: dict):
        """Pretraining for learning sparse inverse covariance matrices.

        Arguments:
            data {dict} -- 
        """
        batch_size = self.pas['training']['train_batch_size']
        pre_learning_rate = self.pas['training']['pre_learning_rate']
        pre_training_cycle = self.pas['training']['pre_training_cycle']

        supervised_data = {'train label': data['train label'],
                           'train covariance': data['train covariance']}
        unsupervised_data = {'train covariance': data['valid covariance'],
                             'train label': data['valid label']}

        for epoch in range(pre_training_cycle):
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
            results = self.predicting(data=data, epoch=epoch+1)
            print('Train: {:.5f}\tValid: {:.5f}\tTest: {:.5f}'.format(
                results['train']['SICE Penalty'],
                results['valid']['SICE Penalty'],
                results['test']['SICE Penalty']
            ))

    def backpropagation(self,
                        data: dict,
                        ):
        """Backpropagation of neural network.

        Arguments:
            data {dict} -- Input dataset include train, valid and test data and label.

        Raises:
            Warning: [description]

        Returns:
            [type] -- [description]
        """
        self.log.write_graph()
        batch_size = self.pas['training']['train_batch_size']

        early_stop = EarlyStop(log=self.log,
                               data=data,
                               results=self.results,
                               pas=self.pas['early_stop'])

        epoch = early_stop.epoch
        training_cycle = early_stop.parameters['training_cycle']
        while epoch < training_cycle:
            learning_rate = early_stop.parameters['learning_rate']
            if self.pas['basic']['SICE_training']:
                supervised_data = {'train covariance': data['train covariance'],
                                   'train label': data['train label']}
                unsupervised_data = {'train covariance': np.concatenate((data['valid covariance'],
                                                                         data['test covariance'])),
                                     'train label': np.concatenate((data['valid label'],
                                                                    data['test label'])),
                                     }
                self.backpropagation_epoch(
                    data=supervised_data,
                    batch_size=batch_size,
                    learning_rate=learning_rate,
                    training=True,
                    minimizer=self.minimizer_SICE,
                )
                self.backpropagation_epoch(
                    data=unsupervised_data,
                    batch_size=batch_size,
                    learning_rate=learning_rate,
                    training=True,
                    minimizer=self.minimizer_SICE,
                )

            # Training
            self.backpropagation_epoch(data=data,
                                       batch_size=batch_size,
                                       learning_rate=learning_rate,
                                       )

            # Evaluation
            results_epoch = self.predicting(data=data, epoch=epoch)

            early_stop.next(results=results_epoch)
            epoch = early_stop.epoch

        return early_stop

    def discriminant_power_analyse(self):
        """Analyse important edges for discrimination.
        """

        weights = self.log.sess.run(self.trainable_pas)

        z = np.array([1, -1])
        for layer_scope in ['hidden2', 'hidden1']:
            weight = weights['{:s}/weight'.format(layer_scope)]
            z = np.matmul(weight, z)

        # N2G
        weight = weights['N2G1/weight']
        z = np.multiply(weight, z)
        z = np.squeeze(np.sum(np.absolute(z), axis=-1))

        # E2N
        weight = weights['E2N1/weight_multiply']
        z = np.sum(np.abs(np.multiply(np.squeeze(weight), z)), axis=-1)

        weight_names = ['weight', 'weight_SICE',
                        'weight_SICE_bn', 'weight_multiply']
        results = {weight_name: np.squeeze(
            weights['E2N1'][weight_name]) for weight_name in weight_names}
        results['F'] = z

        # save_dir_path = 'F:/OneDriveOffL/Data/Result/Net'
        save_path = os.path.join(self.log.dir_path, 'parameters.mat')

        sio.savemat(save_path, results)

    def save_weights_to_figure(self,
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
                print('Warning: {:} is not in the list {:}'.format(
                    weight_name, weight_names))
                continue

            save_path = os.path.join(
                save_dir_path, 'time {:d}'.format(run_time))

            prefix_tmp = '{:s} {:}{:}'.format(prefix,
                                              weight_name.replace('_', ' '),
                                              ' epoch {:d}'.format(
                                                  epoch + 1) if epoch is not None else '',
                                              )

            save_or_exhibit(weight=weight,
                            prefix=prefix_tmp,
                            save_path=save_path,
                            if_show=if_show,
                            if_save=if_save,
                            if_absolute=if_absolute,
                            )

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

        trainable_pas = self.trainable_pas
        trainable_pas['E2N1/weight_multiply'] = self.tensors['E2N1']['weight_multiply']
        parameters = self.log.sess.run(self.trainable_pas)

        if if_save:
            save_path = os.path.join(self.log.dir_path,
                                     '{:s}_parameters.mat'.format(restored_model))
            sio.savemat(save_path, parameters)

        return parameters
