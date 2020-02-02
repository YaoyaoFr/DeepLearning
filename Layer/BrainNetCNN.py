import numpy as np
import tensorflow as tf

from Layer.LayerObject import LayerObject
from Model.utils_model import load_initial_value


class EdgeToEdge(LayerObject):
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
                                 'padding': 'SAME',
                                 'activation': None,
                                 'scope': 'E2E',
                                 'conv_fun': tf.nn.conv2d,
                                 })
        self.set_parameters(arguments=arguments,
                            parameters=parameters)

        if self.parameters['load_weight']:
            initializer = load_initial_value(type='weight',
                                             name=self.parameters['scope'])
        else:
            initializer = self.get_initial_weight(
                kernel_shape=self.parameters['kernel_shape'])

        # build kernel
        self.weight = tf.Variable(initial_value=initializer,
                                  name=self.parameters['scope'] + '/kernel',
                                  )
        self.trainable_pas['weight'] = self.weight

        L2 = tf.contrib.layers.l2_regularizer(
            self.parameters['L2_lambda'])(self.weight)
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
            self.trainable_pas['bias'] = self.bias

    def build(self, input_tensor, output_shape=None, training=True):
        self.tensors['input'] = input_tensor

        # convolution
        output_row = self.parameters['conv_fun'](input_tensor,
                                                 self.weight,
                                                 strides=self.parameters['strides'],
                                                 padding=self.parameters['padding'],
                                                 )
        output_col = tf.transpose(output_row, [0, 2, 1, 3])

        ROI_num = self.parameters['kernel_shape'][1]
        tile_row = tf.tile(output_row, [1, 1, ROI_num, 1])
        tile_col = tf.tile(output_col, [1, ROI_num, 1, 1])

        output = tf.add(tile_row, tile_col)
        self.tensors['output_conv'] = output

        # bias
        if self.parameters['bias']:
            output = output + self.bias
            self.tensors['output_bias'] = output

        # batch_normalization
        if self.parameters['batch_normalization']:
            output = self.batch_normalization(tensor=output,
                                              scope=self.parameters['scope'] + '/bn',
                                              training=training)
            self.tensors.update(output)
            output = self.tensors['output_bn']

        # activation
        if self.parameters['activation']:
            output = self.parameters['activation'](output)
            self.tensors['output_activation'] = output

        self.tensors['output'] = output
        return output


class EdgeToNode(LayerObject):
    """
    Edge to node layer.
    """
    required_pa = ['kernel_shape']

    def __init__(self,
                 arguments: dict,
                 parameters: dict = None,
                 ):
        LayerObject.__init__(self)
        self.optional_pa.update({'bias': True,
                                 'strides': [1, 1, 1, 1],
                                 'batch_normalization': False,
                                 'padding': 'VALID',
                                 'activation': None,
                                 'scope': 'E2N',
                                 'conv_fun': tf.nn.conv2d,
                                 'L2_lambda': 5e-3,
                                 })
        self.set_parameters(arguments=arguments,
                            parameters=parameters)

        #Build kernels
        initializer = self.get_initial_weight(
            kernel_shape=self.parameters['kernel_shape'])
        self.weight_row = tf.Variable(initial_value=initializer,
                                      name=self.parameters['scope'] +
                                      '/kernel',
                                      )
        self.trainable_pas['weight_row'] = self.weight_row
        l2_loss_row = tf.contrib.layers.l2_regularizer(
            self.parameters['L2_lambda'])(self.weight_row)
        tf.add_to_collection('L2_loss', l2_loss_row)

        initializer = self.get_initial_weight(
            kernel_shape=self.parameters['kernel_shape'])
        self.weight_col = tf.transpose(tf.Variable(initial_value=initializer,
                                                   name=self.parameters['scope'] +
                                                   '/kernel',
                                                   ),
                                       perm=[1, 0, 2, 3])
        self.tensors['weight_col'] = self.weight_col
        l2_loss_col = tf.contrib.layers.l2_regularizer(
            self.parameters['L2_lambda'])(self.weight_col)
        tf.add_to_collection('L2_loss', l2_loss_col)

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
            self.trainable_pas['bias'] = self.bias

    def build(self, *args, **kwargs):
        input_tensor = kwargs['input_tensor'] if 'input_tensor' in kwargs else kwargs['output']

        self.tensors['input'] = input_tensor
        if len(input_tensor.shape.as_list()) == 3:
            input_tensor = tf.expand_dims(input_tensor, axis=-1)

        training = kwargs['training']

        # convolution
        output_col = self.parameters['conv_fun'](input_tensor,
                                                 self.weight_col,
                                                 strides=self.parameters['strides'],
                                                 padding=self.parameters['padding'],
                                                 )
        output_row = self.parameters['conv_fun'](input_tensor,
                                                 self.weight_row,
                                                 strides=self.parameters['strides'],
                                                 padding=self.parameters['padding'],
                                                 )
        output = (output_row + tf.transpose(output_col, perm=[0, 2, 1, 3])) / 2

        self.tensors['output_conv'] = output

        # bias
        if self.parameters['bias']:
            output = output + self.bias
            self.tensors['output_bias'] = output

        # batch_normalization
        if self.parameters['batch_normalization']:
            output = self.batch_normalization(tensor=output,
                                              scope=self.parameters['scope'] + '/bn',
                                              training=training)
            self.tensors.update(output)
            output = self.tensors['output_bn']

        # activation
        if self.parameters['activation']:
            output = self.parameters['activation'](output)
            self.tensors['output_activation'] = output

        self.tensors['output'] = output
        return output

    def get_initial_weight(self,
                           kernel_shape: list,
                           mode: str = 'fan_in',
                           distribution: str = 'norm',
                           ):
        rank = len(kernel_shape)
        assert rank in {
            2, 3, 4, 5}, 'The rank of kernel expected in {2, 3, 4, 5} but go {:d}'.format(rank)
        assert 'activation' in self.parameters, 'The activation function must be given. '

        if rank == 2:
            fan_in = kernel_shape[0] + kernel_shape[1]
            fan_out = kernel_shape[1]
        else:
            receptive_field_size = kernel_shape[1] * kernel_shape[1]
            fan_in = kernel_shape[-2] * receptive_field_size
            fan_out = kernel_shape[-1] * receptive_field_size

        if mode == 'fan_in':
            scale = 1 / max(1., fan_in)
        elif mode == 'fan_out':
            scale = 1 / max(1., fan_out)
        else:
            scale = 1 / max(1., float(fan_in + fan_out) / 2)

        if distribution == 'norm':
            stddev = np.sqrt(2. * scale)
            initial_value = tf.truncated_normal(shape=kernel_shape,
                                                stddev=stddev)
        elif distribution == 'uniform':
            limit = np.sqrt(3. * scale)
            initial_value = tf.random_uniform(shape=kernel_shape,
                                              minval=-limit,
                                              maxval=limit)

        return initial_value


class NodeToGraph(LayerObject):
    """
    Node to graph layer.
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
                                 'padding': 'VALID',
                                 'activation': None,
                                 'scope': 'N2G',
                                 'conv_fun': tf.nn.conv2d,
                                 'L2_lambda': 5e-3,
                                 'load_weight': False,
                                 })
        self.tensors = {}

        self.parameters = self.set_parameters(arguments=arguments,
                                              parameters=parameters)

        # Weight initializer
        initializer = self.get_initial_weight(
            kernel_shape=self.parameters['kernel_shape'])
        # Load weight
        if self.parameters['load_weight']:
            initializer = load_initial_value(type='weight',
                                             name=self.parameters['scope'])

        # build kernel
        
        self.weight = tf.Variable(initial_value=initializer,
                                  name=self.parameters['scope'] + '/kernel',
                                  )
        self.trainable_pas['weight'] = self.weight

        l2_loss = tf.contrib.layers.l2_regularizer(
            self.parameters['L2_lambda'])(self.weight)
        tf.add_to_collection('L2_loss', l2_loss)

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
            self.trainable_pas['bias'] = self.bias

    def build(self, *args, **kwargs):
        input_tensor = kwargs['input_tensor'] if 'input_tensor' in kwargs else kwargs['output']
        self.tensors['input'] = input_tensor
        training = kwargs['training']

        # convolution
        output = self.parameters['conv_fun'](input_tensor,
                                             self.weight,
                                             strides=self.parameters['strides'],
                                             padding=self.parameters['padding'],
                                             )

        self.tensors['output_conv'] = output

        # bias
        if self.parameters['bias']:
            output = output + self.bias
            self.tensors['output_bias'] = output

        # batch_normalization
        if self.parameters['batch_normalization']:
            output = self.batch_normalization(tensor=output,
                                              scope=self.parameters['scope'] + '/bn',
                                              training=training)
            self.tensors.update(output)
            output = self.tensors['output_bn']

        # activation
        if self.parameters['activation']:
            output = self.parameters['activation'](output)
            self.tensors['output_activation'] = output

        self.tensors['output'] = output
        return output
