import tensorflow as tf

from Layer.BasicLayer import FullyConnected
from Layer.LayerObject import LayerObject


class AutoEncoder(LayerObject):
    """
    Autoencoder layer.
    """
    required_pa = ['kernel_shape']

    def __init__(self,
                 arguments: dict,
                 parameters: dict = None,
                 ):
        LayerObject.__init__(self)
        self.optional_pa.update({'bias': True,
                                 'batch_normalization': False,
                                 'activation': None,
                                 'L2_lambda': 5e-3,
                                 'load_weight': False,
                                 'beta': 1,
                                 'rho': 0.1,
                                 'scope': 'AE',
                                 })
        self.set_parameters(arguments=arguments,
                            parameters=parameters)

        self.input_tensor = tf.placeholder(
            dtype=tf.float32, shape=[None, self.parameters['kernel_shape'][0]])
        self.output_tensor = tf.placeholder(
            dtype=tf.float32, shape=[None, self.parameters['kernel_shape'][0]])

        # Encoding layer
        arguments['scope'] = self.parameters['scope'] + '/enc'
        self.encode_layer = FullyConnected(arguments=arguments,
                                           parameters=parameters)
        self.trainable_pas.update({'enc/{:s}'.format(parameter):
                                   self.encode_layer.trainable_pas[parameter]
                                   for parameter in self.encode_layer.trainable_pas})

        # Decoding layer
        arguments['scope'] = self.parameters['scope'] + 'dec'
        if 'placeholders' in arguments:
            arguments.pop('placeholders')
        decoding_kernel_shape = [
            self.parameters['kernel_shape'][1], self.parameters['kernel_shape'][0]]
        arguments['kernel_shape'] = decoding_kernel_shape
        self.decode_layer = FullyConnected(arguments=arguments,
                                           parameters=parameters)
        self.trainable_pas.update({'dec/{:s}'.format(parameter):
                                   self.encode_layer.trainable_pas[parameter]
                                   for parameter in self.encode_layer.trainable_pas})

    def build(self, *args, **kwargs):
        if 'input_tensor' in kwargs:
            input_tensor = kwargs['input_tensor']
            tensors = None
            placeholders = {'input_tensor': input_tensor}
        elif 'output' in kwargs:
            input_tensor = kwargs['output']
            tensors = {'output': input_tensor}
            placeholders = {}

        placeholders['training'] = kwargs['training']

        self.tensors['input'] = input_tensor

        self.encode_layer(tensors=tensors,
                          placeholders=placeholders)
        tensors = self.encode_layer.tensors
        self.tensors['output'] = tensors['output']

        # Sparsity penalty
        if 'rho' in self.parameters and 'beta' in self.parameters:
            rho_ = tf.reduce_mean(tensors['output'], axis=0)
            rho = self.parameters['rho']
            sparsity_penalty = tf.reduce_sum(
                rho * tf.log(tf.div(rho, rho_) + (1-rho) * tf.log(tf.div(1-rho, 1-rho_))))
            tf.add_to_collection(
                'Sparsity_loss', self.parameters['beta'] * sparsity_penalty)

        self.decode_layer(tensors=tensors,
                          placeholders=placeholders)
        self.tensors['reconstruction'] = self.decode_layer.tensors['output']
