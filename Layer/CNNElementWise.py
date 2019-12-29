import numpy as np
import tensorflow as tf

from Layer.LayerObject import LayerObject
from Model.utils_model import load_initial_value


class EdgeToEdgeElementWise(LayerObject):
    """

    """
    required_pa = ['kernel_shape']

    def __init__(self,
                 arguments: dict,
                 parameters: dict = None,
                 ):
        LayerObject.__init__(self)
        self.optional_pa.update({'L2_lambda': 5e-3,
                                 'bias': True,
                                 'batch_normalization': False,
                                 'activation': None,
                                 'strides': [1, 1, 1, 1],
                                 'conv_fun': tf.nn.conv2d,
                                 'padding': 'SAME',
                                 'scope': 'E2EEW',
                                 })
        self.tensors = {}

        self.pa = self.set_parameters(arguments=arguments,
                                      parameters=parameters)
        # Weight initializer
        initializer = self.get_initial_weight(kernel_shape=self.pa['kernel_shape'])
        # Load weight
        if self.pa['load_weight']:
            initializer = load_initial_value(type='weight',
                                             name=self.pa['scope'])

        # build kernel
        self.weight = tf.Variable(initial_value=initializer,
                                  name=self.pa['scope'] + '/kernel',
                                  )
        self.tensors['weight'] = self.weight

        L2 = tf.contrib.layers.l1_regularizer(self.pa['L2_lambda'])(self.weight)
        tf.add_to_collection('L2_loss', L2)

        # build bias
        num_output_channels = self.pa['kernel_shape'][-1]

        if self.pa['bias']:
            if self.pa['load_bias']:
                initializer = load_initial_value(type='bias',
                                                 name=self.pa['scope'])
            else:
                initializer = tf.constant(0.0, shape=[num_output_channels])
            self.bias = tf.Variable(initial_value=initializer,
                                    name=self.pa['scope'] + '/bias',
                                    )
            self.tensors['bias'] = self.bias

    def build(self, **kwargs):
        if 'input' in kwargs:
            input_tensor = kwargs['input']
        elif 'output' in kwargs:
            input_tensor = kwargs['output']
        training = kwargs['training']

        self.tensors['input'] = input_tensor
        shape = input_tensor.shape.as_list()
        if len(shape) == 3:
            input_tensor = tf.expand_dims(input_tensor, axis=-1)

        # convolution
        input_slices = tf.split(input_tensor, axis=1, num_or_size_splits=shape[1])
        weight_slices = tf.split(self.weight, axis=0, num_or_size_splits=self.pa['kernel_shape'][0])

        output = []
        for input_slice, weight_slice in zip(input_slices, weight_slices):
            feature_map = self.pa['conv_fun'](input_slice,
                                              weight_slice,
                                              strides=self.pa['strides'],
                                              padding=self.pa['padding'],
                                              )
            output.append(feature_map)
        output = tf.concat(output, axis=1)

        self.tensors['output_conv'] = output

        # bias
        if self.pa['bias']:
            output = output + self.bias
            self.tensors['output_bias'] = output

        # batch_normalization
        if self.pa['batch_normalization']:
            output = self.batch_normalization(tensor=output,
                                              scope=self.pa['scope'] + '/bn',
                                              training=training)
            self.tensors.update(output)
            output = self.tensors['output_bn']

        # activation
        if self.pa['activation']:
            output = self.pa['activation'](output)
            self.tensors['output_activation'] = output

        self.tensors['output'] = output
        return output

class EdgeToNodeElementWise(LayerObject):
    """

    """
    required_pa = ['kernel_shape']

    def __init__(self,
                 arguments: dict,
                 parameters: dict = None,
                 ):
        LayerObject.__init__(self)
        self.optional_pa.update({'L2_lambda': 5e-3,
                                 'bias': True,
                                 'batch_normalization': False,
                                 'activation': None,
                                 'strides': [1, 1, 1, 1],
                                 'conv_fun': tf.nn.conv2d,
                                 'padding': 'VALID',
                                 'scope': 'E2NEW',
                                 })
        self.tensors = {}

        self.pa = self.set_parameters(arguments=arguments,
                                      parameters=parameters)
        # Weight initializer
        initializer = self.get_initial_weight(kernel_shape=self.pa['kernel_shape'])
        # Load weight
        if self.pa['load_weight']:
            initializer = load_initial_value(type='weight',
                                             name=self.pa['scope'])

        # build kernel
        self.weight = tf.Variable(initial_value=initializer,
                                  name=self.pa['scope'] + '/kernel',
                                  )
        self.tensors['weight'] = self.weight

        L2 = tf.contrib.layers.l1_regularizer(self.pa['L2_lambda'])(self.weight)
        tf.add_to_collection('L2_loss', L2)

        # build bias
        num_output_channels = self.pa['kernel_shape'][-1]

        if self.pa['bias']:
            if self.pa['load_bias']:
                initializer = load_initial_value(type='bias',
                                                 name=self.pa['scope'])
            else:
                initializer = tf.constant(0.0, shape=[num_output_channels])
            self.bias = tf.Variable(initial_value=initializer,
                                    name=self.pa['scope'] + '/bias',
                                    )
            self.tensors['bias'] = self.bias

    def build(self, **kwargs):
        if 'input_tensor' in kwargs:
            input_tensor = kwargs['input_tensor']
        elif 'output' in kwargs:
            input_tensor = kwargs['output']
        training = kwargs['training']

        shape = input_tensor.shape.as_list()
        if len(shape) == 3:
            input_tensor = tf.expand_dims(input_tensor, axis=-1)

        # convolution
        input_slices = tf.split(input_tensor, axis=1, num_or_size_splits=shape[1])
        weight_slices = tf.split(self.weight, axis=0, num_or_size_splits=self.pa['kernel_shape'][0])

        output = []
        for input_slice, weight_slice in zip(input_slices, weight_slices):
            feature_map = self.pa['conv_fun'](input_slice,
                                              weight_slice,
                                              strides=self.pa['strides'],
                                              padding=self.pa['padding'],
                                              )
            output.append(feature_map)
        output = tf.concat(output, axis=1)

        self.tensors['output_conv'] = output

        # bias
        if self.pa['bias']:
            output = output + self.bias
            self.tensors['output_bias'] = output

        # batch_normalization
        if self.pa['batch_normalization']:
            output = self.batch_normalization(tensor=output,
                                              scope=self.pa['scope'] + '/bn',
                                              training=training)
            self.tensors.update(output)
            output = self.tensors['output_bn']

        # activation
        if self.pa['activation']:
            output = self.pa['activation'](output)
            self.tensors['output_activation'] = output

        self.tensors['output'] = output
        return output