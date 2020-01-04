import os
import h5py
import tensorflow as tf

from Log.log import Log
from Schemes.xml_parse import parse_xml_file
from Model.utils_model import EarlyStop, get_metrics, upper_triangle
from Model.NN import NeuralNetwork


class StackedAutoEncoders(NeuralNetwork):

    def __init__(self,
                 dir_path: str,
                 log: Log = None,
                 scheme: int or str = 1,
                 spe_pas: dict = None, 
                 ):
        NeuralNetwork.__init__(self,
                               log=log,
                               dir_path=dir_path,
                               scheme=scheme,
                               spe_pas=spe_pas,
                               )
        self.auto_encoders = [layer for layer in self.op_layers if 'AE' in layer.pa['scope']]

        self.NN_type = 'DAE'

    def load_parameters(self,
                        scheme: str,
                        spe_pas: dict = None):
        """
        Load parameters from configuration file (.xml)
        :param scheme: file path of configuration file
        :return:
        """
        if not spe_pas:
            pas = parse_xml_file(os.path.join(self.project_path, 'Schemes/{:s}.xml'.format(scheme)))

        autoencoder_index = 0
        while True:
            str = 'early_stop{:d}'.format(autoencoder_index)
            if str in pas['parameters']:
                self.pas[str] = pas['parameters'][str]
                autoencoder_index += 1
            else:
                break
        self.pas['training'] = pas['parameters']['training']
        self.pas['basic'] = pas['parameters']['basic']
        self.pas['layers'] = pas['layers']

    def build_structure(self):

        self.structure = {}
        layer_tensor = None

        for layer in self.op_layers:
            layer(tensors=layer_tensor,
                  placeholders=self.input_placeholders)
            layer_tensor = layer.tensors

            self.tensors[layer.pa['scope']] = layer_tensor

        print('Build stacked autoencoder.')

        self.initialization()
        self.log.saver = tf.train.Saver(tf.global_variables(), max_to_keep=1000, name='saver')

    def build_optimizer(self,
                        init_op,
                        autoencoder_index: int = -1,
                        penalties: list = []):
        assert autoencoder_index <= len(self.auto_encoders), \
            'The index of autoencoder must less equal than {:d} but get {:d}.'.format(len(self.auto_encoders),
                                                                                      autoencoder_index)

        lr_place = self.input_placeholders['learning_rate']
        output_place = self.input_placeholders['output_tensor']

        if autoencoder_index == -1:
            output_tensor = self.op_layers[-1].tensors['output']
            variables = tf.trainable_variables()
            task = 'prediction'
        else:
            output_tensor = self.auto_encoders[autoencoder_index].tensors['reconstruction']
            output_place = self.auto_encoders[autoencoder_index].tensors['input']
            variables = self.auto_encoders[autoencoder_index].variables
            task = 'regression'

        # Cost function
        self.results = get_metrics(ground_truth=output_place,
                                   output_tensor=output_tensor,
                                   task=task)

        # Build loss function, which contains the cross entropy and regularizers.
        if task == 'prediction':
            self.results['Cost'] = tf.reduce_mean(self.results['Cross Entropy'])
        elif task == 'regression':
            self.results['Cost'] = self.results['MSE']

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
            optimizer = tf.train.AdamOptimizer(lr_place, name='optimizer')
            with tf.variable_scope('graph_nn', reuse=tf.AUTO_REUSE):
                self.global_step = tf.Variable(initial_value=0, trainable=False, name='global_step')
                self.minimizer = optimizer.minimize(self.results['Cost'],
                                                    global_step=self.global_step,
                                                    var_list=variables,
                                                    name='minimizer',
                                                    )
                init_op = tf.variables_initializer(set(tf.global_variables()) - init_op)
                self.initialization(init_op=init_op)

        print('{:s} Optimizer initialized.'.format(self.NN_type))

    def training(self,
                 data: h5py.Group or dict,
                 run_time: int = 1,
                 fold_name: str = None,
                 restored_path: str = None,
                 show_info: bool = True):

        data = self.load_data(data)
        data = upper_triangle(data)

        with self.graph.as_default():
            self.build_structure()
            init_op = set(tf.global_variables())

            for autoencoder_index in range(len(self.auto_encoders)):
                self.build_optimizer(autoencoder_index=autoencoder_index, init_op=init_op)
                early_stop = EarlyStop(log=self.log,
                                       data=data,
                                       results=self.results,
                                       pas=self.pas['early_stop{:d}'.format(autoencoder_index + 1)])
                self.backpropagation(data=data,
                                     early_stop=early_stop
                                     )

            self.build_optimizer(init_op=init_op)
            early_stop = EarlyStop(log=self.log,
                                   data=data,
                                   results=self.results,
                                   pas=self.pas['early_stop0'])
            early_stop = self.backpropagation(data=data,
                                              early_stop=early_stop)

        early_stop.show_results(run_time=run_time, fold_name=fold_name)
        return early_stop.results
