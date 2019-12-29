import tensorflow as tf

from Layer.BasicLayer import FullyConnected
from Layer.LayerObject import LayerObject


class AutoEncoder(LayerObject):
    """

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
                                 'scope': 'AE',
                                 })
        self.tensors = {}
        self.pa = self.set_parameters(arguments=arguments,
                                      parameters=parameters)

        self.input_tensor = tf.placeholder(
            dtype=tf.float32, shape=[None, self.pa['kernel_shape'][0]])
        self.output_tensor = tf.placeholder(
            dtype=tf.float32, shape=[None, self.pa['kernel_shape'][0]])

        # Encoding layer
        arguments['scope'] = self.pa['scope'] + '/enc'
        self.encode_layer = FullyConnected(arguments=arguments,
                                           parameters=parameters)

        # Decoding layer
        arguments['scope'] = self.pa['scope'] + 'dec'
        if 'placeholders' in arguments:
            arguments.pop('placeholders')
        decoding_kernel_shape = [
            self.pa['kernel_shape'][1], self.pa['kernel_shape'][0]]
        arguments['kernel_shape'] = decoding_kernel_shape
        self.decode_layer = FullyConnected(arguments=arguments,
                                           parameters=parameters)

        # Building trainable variables
        self.variables = []
        for layer in [self.encode_layer, self.decode_layer]:
            self.variables.append(layer.tensors['weight'])
            if 'bias' in layer.tensors:
                self.variables.append(layer.tensors['bias'])

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
        if 'rho' in self.pa and 'beta' in self.pa:
            rho_ = tf.reduce_mean(tensors['output'], axis=0)
            rho = self.pa['rho']
            sparsity_penalty = tf.reduce_sum(
                rho * tf.log(tf.div(rho, rho_) + (1-rho) * tf.log(tf.div(1-rho, 1-rho_))))
            tf.add_to_collection('Sparsity_loss', self.pa['beta'] * sparsity_penalty)

        self.decode_layer(tensors=tensors,
                          placeholders=placeholders)
        self.tensors['reconstruction'] = self.decode_layer.tensors['output']
