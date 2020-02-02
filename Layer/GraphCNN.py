import tensorflow as tf

from Layer.LayerObject import LayerObject
from Model.utils_model import load_initial_value


class GraphCNN(LayerObject):
    """

    """
    required_pa = ['kernel_shape']

    def __init__(self,
                 arguments,
                 parameters: dict = None,
                 ):
        LayerObject.__init__(self)

        self.optional_pa.update({'bias': True,
                                 'strides': [1, 1, 1, 1],
                                 'batch_normalization': False,
                                 'activation': None,
                                 'scope': 'GCN',
                                 })
        self.tensors = {}

        self.parameters = self.set_parameters(arguments=arguments,
                                      parameters=parameters)

        if self.parameters['load_weight']:
            initializer = load_initial_value(type='weight',
                                             name=self.parameters['scope'])
        else:
            initializer = self.get_initial_weight(kernel_shape=self.parameters['kernel_shape'])

        # build kernel
        self.weight = tf.Variable(initial_value=initializer,
                                  name=self.parameters['scope'] + '/kernel',
                                  )
        self.tensors['weight'] = self.weight

        L2 = tf.contrib.layers.l2_regularizer(self.parameters['L2_lambda'])(self.weight)
        tf.add_to_collection('L2_loss', L2)

        # build bias
        num_output_channels = self.parameters['kernel_shape'][-1]
        if self.parameters['bias']:
            if self.parameters['load_bias']:
                initializer = load_initial_value(type='bias',
                                                 name=self.parameters['scope'])
            else:
                initializer = tf.constant(0.0, shape=[num_output_channels])

            self.bias = tf.Variable(initial_value=initializer,
                                    name=self.parameters['scope'] + '/bias',
                                    )
            self.tensors['bias'] = self.bias

    def build(self, input_tensor, correlation_tensor, training=True):
        self.tensors['input'] = input_tensor

        weight = self.tensors['weight']
        # weight *= correlation_tensor

        output = tf.matmul(weight, input_tensor)
        # output = tf.matmul(correlation_tensor, input_tensor)
        self.tensors['output_conv'] = output

        # bias
        if self.parameters['bias']:
            output = output + self.bias
            self.tensors['output_bias'] = output


        # activation
        if self.parameters['activation']:
            output = self.parameters['activation'](output)
            self.tensors['output_activation'] = output

        if self.parameters['expand_dim']:
            output = tf.expand_dims(output, axis=-1)

        # batch_normalization
        if self.parameters['batch_normalization']:
            output = self.batch_normalization(tensor=output,
                                              scope=self.parameters['scope'] + '/bn',
                                              training=training)
            self.tensors.update(output)
            output = self.tensors['output_bn']

        self.tensors['output'] = output
        return output

    def call(self, input_tensor, correlation_tensor, training=True):
        return self.build(input_tensor, correlation_tensor=correlation_tensor, training=training)

