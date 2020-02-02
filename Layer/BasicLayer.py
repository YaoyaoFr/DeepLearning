import math

import numpy as np
import tensorflow as tf

from Layer.LayerObject import LayerObject
from Model.utils_model import load_initial_value


class Placeholder(LayerObject):
    """

    """
    required_pa = ['input_shape']

    def __init__(self,
                 arguments,
                 parameters: dict = None,
                 ):
        LayerObject.__init__(self)
        self.optional_pa.update({'dtype': tf.float32,
                                 'scope': 'input',
                                 })
        self.set_parameters(arguments=arguments, parameters=parameters)

    def build(self, *args, **kwargs):
        return tf.placeholder(dtype=self.parameters['dtype'],
                              shape=self.parameters['input_shape'],
                              name=self.parameters['scope'])

    def __call__(self, *args, **kwargs):
        return self.build()


class Placeholders(LayerObject):
    """

    """
    required_pa = []

    def __init__(self,
                 arguments,
                 parameters=None,
                 ):
        LayerObject.__init__(self)
        self.optional_pa.update({'dtype': tf.float32,
                                 'scope': 'input',
                                 'input_num': 0,
                                 'input_shape': [],
                                 })
        self.parameters = self.set_parameters(arguments=arguments,
                                              parameters=parameters,
                                              )

        if 'input_shape' in self.parameters and 'input_num' in self.parameters:
            input_shape = self.parameters['input_shape']
            input_num = self.parameters['input_num']
            self.input_shapes = [input_shape for _ in range(input_num)]

    def build(self):
        return [tf.placeholder(dtype=self.parameters['dtype'],
                               shape=input_shape,
                               name='{:s}_{:d}'.format(self.parameters['scope'], index + 1))
                for index, input_shape in enumerate(self.input_shapes)]

    def call(self):
        return self.build()


class FullyConnected(LayerObject):
    """

    """
    required_pa = ['kernel_shape']

    def __init__(self,
                 arguments,
                 parameters: dict = None,
                 ):
        LayerObject.__init__(self)
        self.optional_pa.update({'bias': True,
                                 'batch_normalization': False,
                                 'activation': None,
                                 'load_weight': False,
                                 'scope': 'FC',
                                 })
        self.set_parameters(arguments=arguments,
                            parameters=parameters)

        initializer = self.get_initial_weight(kernel_shape=self.parameters['kernel_shape'],
                                              distribution='norm')
        # Load weight
        if self.parameters['load_weight']:
            initializer = load_initial_value(type='weight',
                                             name=self.parameters['scope'])

        # build weights
        self.weight = tf.Variable(initial_value=initializer,
                                  name=self.parameters['scope'] + '/kernel',
                                  trainable=True,
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
        input_tensor = kwargs['output' if 'output' in kwargs else 'input_tensor']
        training = kwargs['training']
        self.tensors['input'] = input_tensor

        # fully connected layer
        output = tf.matmul(input_tensor, self.weight)
        self.tensors['output_matmul'] = output

        # bias
        if self.parameters['bias']:
            output = output + self.bias
            self.tensors['output_bias'] = output

        # batch normalization
        if self.parameters['batch_normalization']:
            output_bn = self.batch_normalization(tensor=output,
                                                 scope=self.parameters['scope'] + '/bn',
                                                 training=training)['output_bn']
            self.tensors['output_bn'] = output_bn
            output = output_bn

        # activation
        if self.parameters['activation']:
            output = self.parameters['activation'](output)
            self.tensors['output_activation'] = output

        self.tensors['output'] = output
        return output


class Unfold(LayerObject):
    """

    """
    required_pa = []
    optional_pa = {'scope': 'unfolds',
                   }

    def __init__(self,
                 arguments,
                 parameters=None,
                 ):
        LayerObject.__init__(self)

        self.tensors = {}
        self.parameters = self.set_parameters(arguments=arguments,
                                              parameters=parameters)

    def build(self, *args, **kwargs):
        if 'input_tensor' in kwargs:
            self.tensors['input'] = kwargs['input_tensor']
        elif 'output' in kwargs:
            self.tensors['input'] = kwargs['output']

        input_tensor = tf.concat(self.tensors['input'], axis=-1)
        tensor_shape = input_tensor.get_shape().as_list()
        output = tf.reshape(input_tensor, [-1, np.prod(tensor_shape[1:])])

        self.tensors['output'] = output
        return output


class Fold(LayerObject):
    """

    """

    def __init__(self,
                 arguments,
                 ):
        LayerObject.__init__(self)

        self.tensors = dict()
        self.scope = arguments['scope']

    def build(self, input_tensor, output_shape=None):
        self.tensors['input'] = input_tensor

        output = tf.reshape(input_tensor, output_shape)
        self.tensors['output'] = output
        return output

    def call(self, input_tensor, output_shape, training=True):
        return self.build(input_tensor, output_shape=output_shape)


class Softmax(LayerObject):
    """

    """
    required_pa = []
    optional_pa = {'scope': 'softmax',
                   }

    def __init__(self,
                 arguments):
        LayerObject.__init__(self)

        self.tensors = {}
        self.parameters = self.set_parameters(arguments=arguments)

    def build(self, input_tensor, output_shape=None):
        self.tensors['input'] = input_tensor

        # softmax layer
        output = tf.nn.softmax(logits=input_tensor, name='softmax')
        self.tensors['output'] = output

        return output

    def call(self, input_tensor, output_shape=None, training=True):
        return self.build(input_tensor, output_shape=output_shape)


class MaxPooling(LayerObject):
    """

    """
    required_pa = ['kernel_shape']
    optional_pa = {'strides': [1, 2, 2, 1],
                   'padding': 'VALID',
                   'scope': 'max_pool',
                   'pool_fun': tf.nn.max_pool2d,
                   }

    def __init__(self,
                 arguments,
                 ):
        LayerObject.__init__(self)
        self.tensors = {}
        self.parameters = self.set_parameters(arguments=arguments)

    def build(self, input_tensor, output_shape=None):
        self.tensors['input'] = input_tensor

        # max pool
        output = self.parameters['pool_fun'](input_tensor,
                                             ksize=self.parameters['kernel_shape'],
                                             strides=self.parameters['strides'],
                                             padding=self.parameters['padding'])
        self.tensors['output'] = output

        return output

    def call(self, input_tensor, output_shape=None, training=True):
        return self.build(input_tensor, output_shape=output_shape)
        pass


class MaxPoolings(LayerObject):
    """

    """
    required_pa = ['kernel_shape']
    optional_pa = {'strides': [2, 2],
                   'padding': 'VALID',
                   'scope': 'pool',
                   'pool_fun': tf.layers.max_pooling2d,
                   }

    def __init__(self,
                 arguments,
                 ):
        LayerObject.__init__(self)

        self.tensors = {}
        self.parameters = self.set_parameters(arguments=arguments)

    def build(self, input_tensors, output_shape=None):
        self.tensors['input'] = input_tensors

        # max pool
        output = [self.parameters['pool_fun'](input_tensor,
                                              ksize=self.parameters['kernel_shape'],
                                              strides=self.parameters['strides'],
                                              padding=self.parameters['padding'])
                  for input_tensor in input_tensors]
        self.tensors['output'] = output

        return output

    def call(self, input_tensor, output_shape=None, training=True):
        return self.build(input_tensors=input_tensor, output_shape=output_shape)


class SpatialPyramidPool3D(LayerObject):

    def __init__(self, arguments):
        self.tensors = dict()
        self.kernel_shape = arguments['kernel_shape']
        self.scope = arguments['scope']

    def build(self, input_tensors):
        """
        previous_conv: Tensors vector of previous convolution layer
        num_sample: an int number of image in the batch
        previous_conv_size: an int vector [height, width] of the matrix features size of previous convolution layer
        out_pool_size: a int vector of expected output size of max pooling layer

        returns: a tensor vector with shape [1 x n] is the concentration of multi-level pooling
        """
        self.tensors['input'] = input_tensors
        output_tensors = []
        for index, input_tensor in enumerate(input_tensors):
            shape = input_tensor.get_shape().as_list()
            batch_size = shape[0]
            conv_size = shape[1:4]

            spps = []
            for i in range(len(self.kernel_shape)):
                h_strd = h_size = math.ceil(
                    float(conv_size[0]) / self.kernel_shape[i])
                w_strd = w_size = math.ceil(
                    float(conv_size[1]) / self.kernel_shape[i])
                d_strd = d_size = math.ceil(
                    float(conv_size[2]) / self.kernel_shape[i])
                pad_h = int(self.kernel_shape[i] * h_size - conv_size[0])
                pad_w = int(self.kernel_shape[i] * w_size - conv_size[1])
                pad_d = int(self.kernel_shape[i] * d_size - conv_size[2])
                new_previous_conv = tf.pad(tensor=input_tensor,
                                           paddings=tf.constant(
                                               [[0, 0],
                                                [0, pad_h],
                                                [0, pad_w],
                                                [0, pad_d],
                                                [0, 0]]
                                           ))
                max_pool = tf.nn.max_pool3d(input=new_previous_conv,
                                            ksize=[1, h_size,
                                                   h_size, d_size, 1],
                                            strides=[1, h_strd,
                                                     w_strd, d_strd, 1],
                                            padding='SAME',
                                            name='{:s}_{:d}'.format(self.scope, index + 1))
                new_shape = max_pool.get_shape().as_list()
                spp = tf.reshape(max_pool, [-1, np.prod(new_shape[1:])])
                spps.append(spp)
            spps = tf.concat(values=spps, axis=-1)
            output_tensors.append(spps)
        output_tensor = tf.concat(output_tensors, axis=-1)
        self.tensors['output'] = output_tensor
        return output_tensor

    def call(self, input_tensor, output_shape=None, training=True):
        return self.build(input_tensors=input_tensor)


class UnPooling(LayerObject):
    """
    Unpool a max-pooled layer.

    Currently this method does not use the argmax information from the previous pooling layer.
    Currently this method assumes that the size of the max-pooling filter is same as the strides.

    Each entry in the pooled map would be replaced with an NxN kernel with the original entry in the upper left.
    For example: a 1x2x2x1 map of

        [[[[1], [2]],
          [[3], [4]]]]

    could be unpooled to a 1x4x4x1 map of

        [[[[ 1.], [ 0.], [ 2.], [ 0.]],
          [[ 0.], [ 0.], [ 0.], [ 0.]],
          [[ 3.], [ 0.], [ 4.], [ 0.]],
          [[ 0.], [ 0.], [ 0.], [ 0.]]]]
    """

    def __init__(self,
                 arguments,
                 ):
        LayerObject.__init__(self)
        self.tensors = dict()
        self.kernel_shape = [2, 2]

        if 'kernel_shape' in arguments:
            self.kernel_shape = arguments['kernel_shape']
        self.scope = arguments['scope']

    def build(self, input_tensor, output_shape=None, training=True):
        self.tensors['input'] = input_tensor

        num_channels = input_tensor.get_shape()[-1]
        input_dtype_as_numpy = input_tensor.dtype.as_numpy_dtype()
        kernel_width, kernel_height = self.kernel_shape

        # build kernel
        kernel_value = np.zeros(
            (kernel_width, kernel_height, num_channels, num_channels), dtype=input_dtype_as_numpy)
        kernel_value[0, 0, :, :] = np.eye(num_channels, num_channels)
        kernel = tf.constant(kernel_value, name=self.scope + '/kernel')

        # do the un-pooling using conv2d_transpose
        output = tf.nn.conv2d_transpose(input_tensor,
                                        kernel,
                                        output_shape=output_shape,
                                        strides=(1, kernel_width,
                                                 kernel_height, 1),
                                        padding='VALID')
        self.tensors['output'] = output
        return output

    def call(self, input_tensor, output_shape=None, training=True):
        return self.build(input_tensor, output_shape=output_shape)


class UnPooling3D(LayerObject):

    def __init__(self,
                 arguments,
                 ):
        LayerObject.__init__(self)
        self.tensors = dict()
        self.kernel_shape = [2, 2, 2]

        if 'kernel_shape' in arguments:
            self.kernel_shape = arguments['kernel_shape']
        self.scope = arguments['scope']

    def build(self, input_tensor, output_shape=None):
        self.tensors['input'] = input_tensor

        num_channels = input_tensor.get_shape()[-1]
        input_dtype_as_numpy = input_tensor.dtype.as_numpy_dtype()
        kernel_width, kernel_height, kernel_depth = self.kernel_shape

        # build kernel
        kernel_value = np.zeros((kernel_width, kernel_height, kernel_depth, num_channels, num_channels),
                                dtype=input_dtype_as_numpy)
        kernel_value[0, 0, 0, :, :] = np.eye(num_channels, num_channels)
        kernel = tf.constant(kernel_value, name=self.scope + '/kernel')

        # do the un-pooling using conv3d_transpose
        output = tf.nn.conv3d_transpose(input_tensor,
                                        kernel,
                                        output_shape=output_shape,
                                        strides=(
                                            1, kernel_width, kernel_height, kernel_depth, 1),
                                        padding='VALID')
        self.tensors['output'] = output
        return output

    def call(self, input_tensor, output_shape=None, training=True):
        return self.build(input_tensor, output_shape=output_shape)
