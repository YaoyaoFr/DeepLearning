import tensorflow as tf

from Layer.LayerObject import LayerObject
from Model.utils_model import get_initial_weight, load_initial_value


class Convolution(LayerObject):
    """
    Convolutional layer.
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
                                 'scope': 'conv',
                                 'conv_fun': tf.nn.conv2d,
                                 'data_format': 'NHWC',
                                 })
        self.set_parameters(arguments=arguments, parameters=parameters)

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

    def build(self, **kwargs):
        """Convolution process
        
        Returns:
            [type] -- [description]
        """
        input_tensor = kwargs.get('output')
        self.tensors['input'] = input_tensor

        training = kwargs.get('training')

        # Convolution
        output = self.parameters['conv_fun'](input_tensor,
                                             self.weight,
                                             strides=self.parameters['strides'],
                                             padding=self.parameters['padding'])

        self.tensors['output_conv'] = output

        # bias
        if self.parameters['bias']:
            output = tf.nn.bias_add(value=output,
                                    bias=self.bias,
                                    data_format=self.parameters['data_format'])
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


class Convolutions(LayerObject):
    """

    """
    required_pa = ['kernel_shape']

    def __init__(self,
                 arguments,
                 parameters=None,
                 ):
        LayerObject.__init__(self)
        self.tensors = dict()
        self.optional_pa.update({'bias': True,
                                 'strides': [1, 1, 1, 1],
                                 'batch_normalization': False,
                                 'padding': 'VALID',
                                 'activation': None,
                                 'scope': 'convs',
                                 'conv_fun': tf.layers.conv2d,
                                 }
                                )
        self.parameters = self.set_parameters(arguments=arguments)

        if parameters is not None:
            if 'ROIs' in parameters:
                self.parameters['num'] = len(parameters['ROIs'])
            elif 'input_num' in parameters:
                self.parameters['num'] = parameters['input_num']

        # Weight initializer
        if self.parameters['activation']:
            activation = self.parameters['activation']._tf_api_names[0]
        else:
            activation = ''
        initializer = get_initial_weight(kernel_shape=self.parameters['kernel_shape'],
                                         activation=activation)
        # Load weight
        if self.parameters['load_weight']:
            initializer = load_initial_value(type='weight',
                                             name=self.parameters['scope'])

        # build kernel
        self.kernel = [tf.Variable(initial_value=initializer,
                                   name=self.parameters['scope'] +
                                   '/kernel_{:d}'.format(index + 1),
                                   dtype=tf.float32
                                   )
                       for index in range(self.parameters['num'])]
        self.tensors['weight'] = self.kernel

        # build bias
        num_output_channels = self.parameters['kernel_shape'][-1]

        if self.parameters['bias']:
            if self.parameters['load_bias']:
                initializer = load_initial_value(type='bias',
                                                 name=self.parameters['scope'])
            else:
                initializer = tf.constant(0.0, shape=[num_output_channels])
            self.bias = [tf.Variable(initial_value=initializer,
                                     name=self.parameters['scope'] + '/bias',
                                     ) for _ in range(self.parameters['num'])]
            self.tensors['bias'] = self.bias

    def build(self, input_tensors, output_shape=None, training=True):
        self.tensors['input'] = input_tensors

        # convolution
        self.parameters['conv_fun'] = self.parameters['conv_fun'](input_shape=input_tensors.shape,
                                                                  filter_shape=self.weight.shape,
                                                                  data_format=self.parameters['data_format'],
                                                                  padding=self.parameters['padding'],
                                                                  )

        outputs = [self.parameters['conv_fun'](inp=input_tensor,
                                               filter=kernel)
                   for input_tensor, kernel in zip(input_tensors, self.kernel)]
        self.tensors['output_conv'] = outputs

        # bias
        if self.parameters['bias']:
            outputs = [outputs[index] + self.bias[index]
                       for index in range(len(outputs))]
            self.tensors['output_bias'] = outputs

        # batch_normalization
        if self.parameters['batch_normalization']:
            output_bn = []
            aft_mean = []
            aft_var = []
            for index, output in enumerate(outputs):
                output = self.batch_normalization(tensor=output,
                                                  scope='{:s}/bn_{:d}'.format(self.parameters['scope'],
                                                                              index + 1),
                                                  training=training,
                                                  )
                output_bn.append(output['output_bn'])
                aft_mean.append(output['aft_mean'])
                aft_var.append(output['aft_var'])
            self.tensors['output_bn'] = output_bn
            self.tensors['aft_mean'] = aft_mean
            self.tensors['aft_var'] = aft_var

        # activation
        if self.parameters['activation']:
            outputs = [self.parameters['activation']
                       (output) for output in outputs]
        self.tensors['output_activation'] = outputs

        self.tensors['output'] = outputs
        return outputs

    def call(self, input_tensor, output_shape=None, training=True):
        return self.build(input_tensor, output_shape=output_shape, training=training)


class DeConvolution(LayerObject):
    """

    """

    def __init__(self, arguments):
        LayerObject.__init__(self)
        self.tensors = dict()
        self.kernel = None
        self.bias = True
        self.parametersdding = 'SAME'
        self.activation = tf.nn.relu

        self.kernel_shape = arguments['kernel_shape']
        self.scope = arguments['scope']
        self.conv_fun = arguments['conv_fun']
        if 'kernel' in arguments:
            self.kernel = arguments['kernel']
        if 'bias' in arguments:
            self.bias = arguments['bias']
        if 'strides' in arguments:
            self.strides = arguments['strides']
        if 'padding' in arguments:
            self.parametersdding = arguments['padding']
        if 'activation' in arguments:
            self.activation = arguments['activation']

        # build kernel
        if self.kernel:
            assert self.kernel.get_shape() == self.kernel_shape
        else:
            if self.initializer is None:
                if self.activation._tf_api_names[0] == 'nn.relu':
                    self.initializer = tf.contrib.layers.variance_scaling_initializer()
                else:
                    self.initializer = tf.contrib.layers.xavier_initializer()
            self.kernel = tf.Variable(initial_value=self.initializer(self.kernel_shape),
                                      name=self.scope + '/kernel',
                                      trainable=True,
                                      )
        self.tensors['weight'] = self.kernel

        # build bias
        num_output_channels, num_input_channels = self.kernel_shape[-2:]
        if self.bias:
            self.bias = tf.Variable(initial_value=tf.constant(0.1, shape=[num_output_channels]),
                                    name=self.scope + '/bias')
            self.tensors['bias'] = self.bias

    def build(self, input_tensor, output_shape, training=None):
        self.tensors['input'] = input_tensor

        # convolution
        output = self.conv_fun(value=input_tensor,
                               filter=self.kernel,
                               output_shape=output_shape,
                               strides=self.strides,
                               padding=self.parametersdding)
        self.tensors['output_conv'] = output

        # bias
        if self.bias:
            output = output + self.bias
            self.tensors['output_bias'] = output

        # activation
        if self.activation:
            output = self.activation(output)
            self.tensors['output_activation'] = output

        self.tensors['output'] = output

        return output

    def call(self, input_tensor, output_shape, training=None):
        return self.build(input_tensor, output_shape=output_shape, training=training)


class DepthwiseConvolution(LayerObject):
    """

        """

    def __init__(self,
                 arguments,
                 ):
        LayerObject.__init__(self)
        self.tensors = dict()
        self.kernel = None
        self.bias = True
        self.strides = [1, 1, 1, 1, 1]
        self.batch_normalization = False

        self.kernel_shape = arguments['kernel_shape']
        if 'kernel' in arguments:
            self.kernel = arguments['kernel']
        if 'bias' in arguments:
            self.bias = arguments['bias']
        if 'strides' in arguments:
            self.strides = arguments['strides']
        if 'batch_normalization' in arguments:
            self.batch_normalization = arguments
        self.parametersdding = arguments['padding']
        self.activation = arguments['activation']
        self.scope = arguments['scope']

        # build kernel
        if self.kernel:
            assert self.kernel.get_shape() == self.kernel_shape
        else:
            if self.initializer is None:
                if self.activation._tf_api_names[0] == 'nn.relu':
                    self.initializer = tf.contrib.layers.variance_scaling_initializer()
                else:
                    self.initializer = tf.contrib.layers.xavier_initializer()
            self.kernel = tf.Variable(initial_value=self.initializer(self.kernel_shape),
                                      name=self.scope + '/kernel',
                                      trainable=True,
                                      )
        self.tensors['weight'] = self.kernel

        # build bias
        _, _, _, num_input_channels, num_output_channels = self.kernel.get_shape()
        if self.bias:
            self.bias = tf.Variable(initial_value=tf.constant(0.1, shape=[num_input_channels * num_output_channels]),
                                    name=self.scope + '/bias',
                                    trainable=True,
                                    )
            self.tensors['bias'] = self.bias

    def build(self, input_tensor, output_shape=None, training=True):
        """
        Build the convolution process
        :param input_tensor: A Tensor with shape [batch_size, height, width, depth, num_input_channels]
        :param output_shape: The tensor shape of output tensor
        :return: The output tensor with shape
                    [batch_size, height_out, width_out, depth_out, num_input_channels * num_output_channels]
        """
        self.tensors['input'] = input_tensor

        # convolution
        _, _, _, num_input_channels, num_output_channels = self.kernel.get_shape()
        input_tensor_slices = tf.split(
            value=input_tensor, num_or_size_splits=num_input_channels, axis=-1)
        kernel_slices = tf.split(
            value=self.kernel, num_or_size_splits=num_input_channels, axis=-2)
        output_tensor_slices = [tf.nn.conv3d(input=input_tensor_slice,
                                             filter=kernel_slice,
                                             strides=self.strides,
                                             padding=self.parametersdding
                                             )
                                for input_tensor_slice, kernel_slice in zip(input_tensor_slices, kernel_slices)]
        output = tf.concat(values=output_tensor_slices, axis=-1)
        self.tensors['output_conv'] = output

        # bias
        if self.bias:
            output = output + self.bias
            self.tensors['output_bias'] = output

        # batch_normalization
        if self.batch_normalization:
            output = self.batch_normalization(tensor=output,
                                              scope=self.scope + '/bn',
                                              training=training)
            self.tensors.update(output)
            output = self.tensors['output_bn']

        # activation
        if self.activation:
            output = self.activation(output)
            self.tensors['output_activation'] = output

        self.tensors['output'] = output

        return output

    def call(self, input_tensor, output_shape=None, training=True):
        return self.build(input_tensor, output_shape=output_shape, training=training)


class DepthwiseDeConvolution(LayerObject):
    """

    """

    def __init__(self,
                 arguments,
                 ):
        LayerObject.__init__(self)
        self.tensors = dict()
        self.kernel = None
        self.bias = True
        self.strides = [1, 1, 1, 1, 1]
        self.batch_normalization = False

        self.kernel_shape = arguments['kernel_shape']
        self.conv_fun = arguments['conv_fun']
        if 'kernel' in arguments:
            self.kernel = arguments['kernel']
        if 'bias' in arguments:
            self.bias = arguments['bias']
        if 'strides' in arguments:
            self.strides = arguments['strides']
        if 'batch_normalization' in arguments:
            self.batch_normalization = arguments
        self.parametersdding = arguments['padding']
        self.activation = arguments['activation']
        self.scope = arguments['scope']

        # build kernel
        if self.kernel:
            assert self.kernel.get_shape() == self.kernel_shape
        else:
            if self.initializer is None:
                if self.activation._tf_api_names[0] == 'nn.relu':
                    self.initializer = tf.contrib.layers.variance_scaling_initializer()
                else:
                    self.initializer = tf.contrib.layers.xavier_initializer()
            self.kernel = tf.Variable(initial_value=self.initializer(self.kernel_shape),
                                      name=self.scope + '/kernel',
                                      trainable=True,
                                      )
        self.tensors['weight'] = self.kernel

        # build bias
        _, _, _, num_input_channels, num_output_channels = self.kernel.get_shape()
        if self.bias:
            self.bias = tf.Variable(initial_value=tf.constant(0.1, shape=[num_input_channels]),
                                    name=self.scope + '/bias',
                                    trainable=True,
                                    )
            self.tensors['bias'] = self.bias

    def build(self, input_tensor, output_shape=None, training=True):
        """
        Build the convolution process
        :param input_tensor: A Tensor with shape
                    [batch_size, height, width, depth, num_input_channels * num_output_channels]
        :param output_shape: The tensor shape of output tensor
        :return: The output tensor with shape
                    [batch_size, height_out, width_out, depth_out, num_input_channels]
        """
        self.tensors['input'] = input_tensor

        # convolution
        num_input_channels, num_output_channels = self.kernel_shape[-2:]
        input_tensor_slices = tf.split(
            value=input_tensor, num_or_size_splits=num_input_channels, axis=-1)
        kernel_slices = tf.split(
            value=self.kernel, num_or_size_splits=num_input_channels, axis=-2)
        output_shape[-1] = 1
        output_tensor_slices = [self.conv_fun(value=input_tensor_slice,
                                              filter=kernel_slice,
                                              output_shape=output_shape,
                                              strides=self.strides,
                                              padding=self.parametersdding)
                                for input_tensor_slice, kernel_slice in zip(input_tensor_slices, kernel_slices)]
        output = tf.concat(values=output_tensor_slices, axis=-1)
        self.tensors['output_conv'] = output

        # bias
        if self.bias:
            output = output + self.bias
            self.tensors['output_bias'] = output

        # batch_normalization
        if self.batch_normalization:
            output = self.batch_normalization(tensor=output,
                                              scope=self.scope + '/bn',
                                              training=training)
            self.tensors.update(output)
            output = self.tensors['output_bn']

        # activation
        if self.activation:
            output = self.activation(output)
            self.tensors['output_activation'] = output

        self.tensors['output'] = output

        return output

    def call(self, input_tensor, output_shape=None, training=True):
        return self.build(input_tensor, output_shape=output_shape, training=training)
