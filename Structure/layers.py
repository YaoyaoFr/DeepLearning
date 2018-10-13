from abc import ABCMeta, abstractmethod

import tensorflow as tf
import numpy as np


def batch_normalization(tensor, scope, axis=None):
    shape = tensor.get_shape()
    # offset = tf.Variable(tf.zeros(shape[-1]), name=scope + '/offset')
    # scale = tf.Variable(tf.ones(shape[-1]), name=scope + '/scale')

    if not axis:
        axis = list(range(len(shape) - 1))  # [0,1,2]
    mean, var = tf.nn.moments(tensor, axis)

    if axis == -1:
        mean = tf.expand_dims(input=mean, axis=axis)
        var = tf.expand_dims(input=var, axis=axis)

    output = tf.nn.batch_normalization(tensor,
                                       mean=mean,
                                       variance=var,
                                       offset=None,
                                       scale=None,
                                       variance_epsilon=1e-20)
    return {
        'output_bn': output,
        'bn_mean': mean,
        'bn_var': var,
    }


class Layer(object, metaclass=ABCMeta):
    """

    """

    def __init__(self):
        pass

    @abstractmethod
    def call(self, *args, **kwargs):
        raise NotImplementedError

    def __call__(self, *args, **kwargs):
        return self.call(*args, **kwargs)


class Convolution2D(Layer):
    """

    """

    def __init__(self,
                 arguments,
                 ):
        Layer.__init__(self)
        self.tensors = dict()
        self.kernel = None
        self.bias = True
        self.strides = [1, 1, 1, 1]
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
        self.padding = arguments['padding']
        self.activation = arguments['activation']
        self.scope = arguments['scope']

        # build kernel
        if self.kernel:
            assert self.kernel.get_shape() == self.kernel_shape
        else:
            self.kernel = tf.get_variable(initializer=tf.contrib.layers.xavier_initializer(),
                                          name=self.scope + '/kernel',
                                          shape=self.kernel_shape,
                                          trainable=True,
                                          )
        self.tensors['weight'] = self.kernel

        # build bias
        kernel_height, kernel_width, num_input_channels, num_output_channels = self.kernel.get_shape()
        if self.bias:
            self.bias = tf.Variable(initial_value=tf.constant(0.1, shape=[num_output_channels]),
                                    name=self.scope + '/bias',
                                    trainable=True,
                                    )
            self.tensors['bias'] = self.bias

    def build(self, input_tensor, output_shape=None):
        self.tensors['input'] = input_tensor

        # convolution
        output = tf.nn.conv2d(input_tensor, self.kernel, strides=self.strides, padding=self.padding)
        self.tensors['output_conv'] = output

        # bias
        if self.bias:
            output = output + self.bias
            self.tensors['output_bias'] = output

        # batch_normalization
        if self.batch_normalization:
            output = batch_normalization(tensor=output, scope=self.scope)
            self.tensors.update(output)
            output = self.tensors['output_bn']

        # activation
        if self.activation:
            output = self.activation(output + self.bias)
            self.tensors['output_activation'] = output

        self.tensors['output'] = output

        return output

    def call(self, input_tensor, output_shape=None):
        return self.build(input_tensor, output_shape=output_shape)


class DeConvolution2D(Layer):
    """

    """

    def __init__(self, arguments):
        Layer.__init__(self)
        self.tensors = dict()
        self.kernel = None
        self.bias = True
        self.strides = [1, 1, 1, 1]
        self.padding = 'SAME'
        self.activation = tf.nn.relu

        self.kernel_shape = arguments['kernel_shape']
        if 'kernel' in arguments:
            self.kernel = arguments['kernel']
        if 'bias' in arguments:
            self.bias = arguments['bias']
        if 'strides' in arguments:
            self.strides = arguments['strides']
        if 'padding' in arguments:
            self.padding = arguments['padding']
        if 'activation' in arguments:
            self.activation = arguments['activation']
        self.scope = arguments['scope']

        # build kernel
        if self.kernel:
            assert self.kernel.get_shape() == self.kernel_shape
        else:
            self.kernel = tf.get_variable(initializer=tf.contrib.layers.xavier_initializer(),
                                          name=self.scope + '/kernel',
                                          shape=self.kernel_shape,
                                          trainable=True,
                                          )
        self.tensors['weight'] = self.kernel

        # build bias
        window_height, window_width, num_output_channels, num_input_channels = self.kernel.get_shape()
        if self.bias:
            self.bias = tf.Variable(initial_value=tf.constant(0.1, shape=[num_output_channels]),
                                    name=self.scope + '/bias')
            self.tensors['bias'] = self.bias

    def build(self, input_tensor, output_shape=None):
        self.tensors['input'] = input_tensor

        # convolution
        output = tf.nn.conv2d_transpose(input_tensor,
                                        self.kernel,
                                        output_shape=output_shape,
                                        strides=self.strides,
                                        padding=self.padding)
        self.tensors['output_conv'] = output

        # bias
        if self.bias:
            output = output + self.bias
            self.tensors['output_bias'] = output

        # activation
        if self.activation:
            output = self.activation(output + self.bias)
            self.tensors['output_activation'] = output

        self.tensors['output'] = output

        return output

    def call(self, input_tensor, output_shape=None):
        return self.build(input_tensor, output_shape=output_shape)


class MaxPooling(Layer):
    """

    """

    def __init__(self,
                 arguments,
                 ):
        Layer.__init__(self)
        self.strides = [2, 2]
        self.tensors = dict()
        self.kernel_shape = arguments['kernel_shape']
        if 'strides' in arguments:
            self.strides = arguments['strides']
        self.padding = arguments['padding']
        self.scope = arguments['scope']

    def build(self, input_tensor, output_shape=None):
        self.tensors['input'] = input_tensor

        # max pool
        output = tf.nn.max_pool(input_tensor, ksize=self.kernel_shape, strides=self.strides, padding=self.padding)
        self.tensors['output'] = output

        return output

    def call(self, input_tensor, output_shape=None):
        return self.build(input_tensor, output_shape=output_shape)
        pass


class UnPooling(Layer):
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
        Layer.__init__(self)
        self.tensors = dict()
        self.kernel_shape = [2, 2]

        if 'kernel_shape' in arguments:
            self.kernel_shape = arguments['kernel_shape']
        self.scope = arguments['scope']

    def build(self, input_tensor, output_shape=None):
        self.tensors['input'] = input_tensor

        num_channels = input_tensor.get_shape()[-1]
        input_dtype_as_numpy = input_tensor.dtype.as_numpy_dtype()
        kernel_rows, kernel_cols = self.kernel_shape

        # build kernel
        kernel_value = np.zeros((kernel_rows, kernel_cols, num_channels, num_channels), dtype=input_dtype_as_numpy)
        kernel_value[0, 0, :, :] = np.eye(num_channels, num_channels)
        kernel = tf.constant(kernel_value, name=self.scope + '/kernel')

        # do the un-pooling using conv2d_transpose
        output = tf.nn.conv2d_transpose(input_tensor,
                                        kernel,
                                        output_shape=output_shape,
                                        strides=(1, kernel_rows, kernel_cols, 1),
                                        padding='VALID')
        self.tensors['output'] = output
        return output

    def call(self, input_tensor, output_shape=None):
        return self.build(input_tensor, output_shape=output_shape)


class Unfold(Layer):
    """

    """

    def __init__(self,
                 arguments,
                 ):
        Layer.__init__(self)

        self.tensors = dict()
        self.scope = arguments['scope']

    def build(self, input_tensor, output_shape=None):
        self.tensors['input'] = input_tensor
        num_batch, height, width, num_channels = input_tensor.get_shape()

        output = tf.reshape(input_tensor, [-1, (height * width * num_channels).value])
        self.tensors['output'] = output

        return output

    def call(self, input_tensor, output_shape=None):
        return self.build(input_tensor, output_shape=output_shape)


class Fold(Layer):
    """

    """

    def __init__(self,
                 arguments,
                 ):
        Layer.__init__(self)

        self.tensors = dict()
        self.fold_shape = arguments['fold_shape']
        self.scope = arguments['scope']

    def build(self, input_tensor, output_shape=None):
        self.tensors['input'] = input_tensor

        output = tf.reshape(input_tensor, self.fold_shape)
        self.tensors['output'] = output
        return output

    def call(self, input_tensor, output_shape=None):
        return self.build(input_tensor, output_shape=output_shape)


class FullyConnected(Layer):
    """

    """

    def __init__(self,
                 arguments):
        Layer.__init__(self)
        self.tensors = dict()
        self.weights = None
        self.bias = True
        self.batch_normalization = False

        self.kernel_shape = arguments['kernel_shape']
        if 'weights' in arguments:
            self.weights = arguments['weights']
        if 'bias' in arguments:
            self.bias = arguments['bias']
        if 'batch_normalization' in arguments:
            self.batch_normalization = arguments['batch_normalization']
        self.activation = arguments['activation']
        self.scope = arguments['scope']
        # build weights
        if self.weights:
            assert self.weights.get_shape() == self.kernel_shape
        else:
            self.weights = tf.get_variable(initializer=tf.contrib.layers.xavier_initializer(),
                                          name=self.scope + '/kernel',
                                          shape=self.kernel_shape,
                                          trainable=True,
                                          )
        self.tensors['weight'] = self.weights

        # build bias
        if self.bias:
            self.bias = tf.Variable(tf.constant(0.1, shape=[self.kernel_shape[1]]),
                                    name=self.scope + '/bias')
            self.tensors['bias'] = self.bias

    def build(self, input_tensor, output_shape=None):
        self.tensors['input'] = input_tensor

        # fully connected layer
        output = tf.matmul(input_tensor, self.weights)
        self.tensors['output_matmul'] = output

        # bias
        if self.bias:
            output = output + self.bias
            self.tensors['output_bias'] = output

        # batch normalization
        if self.batch_normalization:
            output = batch_normalization(tensor=output, scope=self.scope)
            self.tensors.update(output)
            output = self.tensors['output_bn']

        # activation
        if self.activation:
            output = self.activation(output)
            self.tensors['output_activation'] = output

        self.tensors['output'] = output
        return output

    def call(self, input_tensor, output_shape=None):
        return self.build(input_tensor, output_shape=output_shape)


class Softmax(Layer):
    """

    """

    def __init__(self,
                 arguments):
        Layer.__init__(self)

        self.tensors = dict()
        self.scope = arguments['scope']

    def build(self, input_tensor, output_shape=None):
        self.tensors['input'] = input_tensor

        # softmax layer
        output = tf.nn.softmax(logits=input_tensor, name='softmax')
        self.tensors['output'] = output

        return output

    def call(self, input_tensor, output_shape=None):
        return self.build(input_tensor, output_shape=output_shape)


class Placeholder(Layer):
    """

    """

    def __init__(self,
                 arguments,
                 ):
        Layer.__init__(self)

        self.input_shape = arguments['input_shape']
        self.dtype = arguments['dtype']
        self.scope = arguments['scope']

    def build(self):
        return tf.placeholder(dtype=self.dtype, shape=self.input_shape, name=self.scope)

    def call(self):
        return self.build()


def get_layer_by_arguments(arguments):
    type = arguments['type']
    layer = None

    if type == 'Placeholder':
        layer = Placeholder(arguments=arguments)
    elif type == 'Convolution2D':
        layer = Convolution2D(arguments=arguments)
    elif type == 'MaxPooling':
        layer = MaxPooling(arguments=arguments)
    elif type == 'Fold':
        layer = Fold(arguments=arguments)
    elif type == 'Unfold':
        layer = Unfold(arguments=arguments)
    elif type == 'FullyConnected':
        layer = FullyConnected(arguments=arguments)
    elif type == 'UnPooling':
        layer = UnPooling(arguments=arguments)
    elif type == 'DeConvolution2D':
        layer = DeConvolution2D(arguments=arguments)
    elif type == 'Softmax':
        layer = Softmax(arguments=arguments)

    return layer
