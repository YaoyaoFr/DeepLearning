from abc import ABCMeta, abstractmethod

import math
import numpy as np
import tensorflow as tf
from Data.utils_prepare_data import select_top_significance_ROIs


def batch_normalization(tensor, scope, training=True, axis=None):
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


class Convolution(Layer):
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
        self.conv_fun = arguments['convolution']

        # build kernel
        if self.kernel:
            assert self.kernel.get_shape() == self.kernel_shape
        else:
            initializer = tf.contrib.layers.xavier_initializer()
            self.kernel = tf.Variable(initial_value=initializer(self.kernel_shape),
                                      name=self.scope + '/kernel',
                                      trainable=True,
                                      )
        self.tensors['weight'] = self.kernel

        # build bias
        num_input_channels, num_output_channels = self.kernel_shape[-2:]
        if self.bias:
            self.bias = tf.Variable(initial_value=tf.constant(0.01, shape=[num_output_channels]),
                                    name=self.scope + '/bias',
                                    trainable=True,
                                    )
            self.tensors['bias'] = self.bias

    def build(self, input_tensor, output_shape=None):
        self.tensors['input'] = input_tensor

        # convolution
        output = self.conv_fun(input_tensor, self.kernel, strides=self.strides, padding=self.padding)
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
            output = self.activation(output)
            self.tensors['output_activation'] = output

        self.tensors['output'] = output

        return output

    def call(self, input_tensor, output_shape=None):
        return self.build(input_tensor, output_shape=output_shape)


class Convolutions(Layer):
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
        self.view_num = arguments['view_num']
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
        self.conv_fun = arguments['convolution']

        # build kernel
        if self.kernel:
            assert len(self.kernel) == self.view_num
        else:
            initializer = tf.contrib.layers.xavier_initializer()
            self.kernel = [tf.Variable(initial_value=initializer(self.kernel_shape),
                                       name=self.scope + '/kernel_{:d}'.format(index + 1),
                                       dtype=tf.float32
                                       )
                           for index in range(self.view_num)]
        self.tensors['weight'] = self.kernel

        # build bias
        num_input_channels, num_output_channels = self.kernel_shape[-2:]
        if self.bias:
            self.bias = [tf.Variable(initial_value=tf.constant(0.01, shape=[num_output_channels]),
                                     name=self.scope + '/bias',
                                     trainable=True,
                                     )
                         for _ in range(self.view_num)]
            self.tensors['bias'] = self.bias

    def build(self, input_tensors, output_shape=None):
        self.tensors['input'] = input_tensors

        # convolution
        outputs = [self.conv_fun(input_tensors[index],
                                 self.kernel[index],
                                 strides=self.strides,
                                 padding=self.padding)
                   for index in range(len(input_tensors))]
        self.tensors['output_conv'] = outputs

        # bias
        if self.bias:
            outputs = [outputs[index] + self.bias[index] for index in range(len(outputs))]
            self.tensors['output_bias'] = outputs

        # batch_normalization
        if self.batch_normalization:
            outputs = [batch_normalization(tensor=output, scope=self.scope + '{:d}'.format(index + 1))['output_bn']
                       for index, output in enumerate(outputs)]
            self.tensors['output_bn'] = outputs

        # activation
        if self.activation:
            output = [self.activation(output) for output in outputs]
        self.tensors['output_activation'] = outputs
        self.tensors['output'] = outputs

        return outputs

    def call(self, input_tensor, output_shape=None):
        return self.build(input_tensor, output_shape=output_shape)


class DeConvolution(Layer):
    """

    """

    def __init__(self, arguments):
        Layer.__init__(self)
        self.tensors = dict()
        self.kernel = None
        self.bias = True
        self.padding = 'SAME'
        self.activation = tf.nn.relu

        self.kernel_shape = arguments['kernel_shape']
        self.scope = arguments['scope']
        self.conv_fun = arguments['convolution']
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

        # build kernel
        if self.kernel:
            assert self.kernel.get_shape() == self.kernel_shape
        else:
            initializer = tf.contrib.layers.xavier_initializer()
            self.kernel = tf.Variable(initial_value=initializer(self.kernel_shape),
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

    def build(self, input_tensor, output_shape=None):
        self.tensors['input'] = input_tensor

        # convolution
        output = self.conv_fun(value=input_tensor,
                               filter=self.kernel,
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
            output = self.activation(output)
            self.tensors['output_activation'] = output

        self.tensors['output'] = output

        return output

    def call(self, input_tensor, output_shape=None):
        return self.build(input_tensor, output_shape=output_shape)


class DepthwiseConvolution(Layer):
    """

        """

    def __init__(self,
                 arguments,
                 ):
        Layer.__init__(self)
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
        self.padding = arguments['padding']
        self.activation = arguments['activation']
        self.scope = arguments['scope']

        # build kernel
        if self.kernel:
            assert self.kernel.get_shape() == self.kernel_shape
        else:
            initializer = tf.contrib.layers.xavier_initializer()
            self.kernel = tf.Variable(initial_value=initializer(self.kernel_shape),
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

    def build(self, input_tensor, output_shape=None):
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
        input_tensor_slices = tf.split(value=input_tensor, num_or_size_splits=num_input_channels, axis=-1)
        kernel_slices = tf.split(value=self.kernel, num_or_size_splits=num_input_channels, axis=-2)
        output_tensor_slices = [tf.nn.conv3d(input=input_tensor_slice,
                                             filter=kernel_slice,
                                             strides=self.strides,
                                             padding=self.padding
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


class DepthwiseDeConvolution(Layer):
    """

    """

    def __init__(self,
                 arguments,
                 ):
        Layer.__init__(self)
        self.tensors = dict()
        self.kernel = None
        self.bias = True
        self.strides = [1, 1, 1, 1, 1]
        self.batch_normalization = False

        self.kernel_shape = arguments['kernel_shape']
        self.conv_fun = arguments['convolution']
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
            initializer = tf.contrib.layers.xavier_initializer()
            self.kernel = tf.Variable(initial_value=initializer(self.kernel_shape),
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

    def build(self, input_tensor, output_shape=None):
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
        input_tensor_slices = tf.split(value=input_tensor, num_or_size_splits=num_input_channels, axis=-1)
        kernel_slices = tf.split(value=self.kernel, num_or_size_splits=num_input_channels, axis=-2)
        output_shape[-1] = 1
        output_tensor_slices = [self.conv_fun(value=input_tensor_slice,
                                              filter=kernel_slice,
                                              output_shape=output_shape,
                                              strides=self.strides,
                                              padding=self.padding)
                                for input_tensor_slice, kernel_slice in zip(input_tensor_slices, kernel_slices)]
        output = tf.concat(values=output_tensor_slices, axis=-1)
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
            output = self.activation(output)
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
        self.pool_fun = arguments['pool']
        if 'strides' in arguments:
            self.strides = arguments['strides']
        self.padding = arguments['padding']
        self.scope = arguments['scope']

    def build(self, input_tensor, output_shape=None):
        self.tensors['input'] = input_tensor

        # max pool
        output = self.pool_fun(input_tensor, ksize=self.kernel_shape, strides=self.strides, padding=self.padding)
        self.tensors['output'] = output

        return output

    def call(self, input_tensor, output_shape=None):
        return self.build(input_tensor, output_shape=output_shape)
        pass


class SpatialPyramidPool3D(Layer):

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

        output_tensors = []
        for index, input_tensor in enumerate(input_tensors):
            shape = input_tensor.get_shape().as_list()
            batch_size = shape[0]
            conv_size = shape[1:4]

            spps = []
            for i in range(len(self.kernel_shape)):
                h_strd = h_size = math.ceil(float(conv_size[0]) / self.kernel_shape[i])
                w_strd = w_size = math.ceil(float(conv_size[1]) / self.kernel_shape[i])
                d_strd = d_size = math.ceil(float(conv_size[2]) / self.kernel_shape[i])
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
                                            ksize=[1, h_size, h_size, d_size, 1],
                                            strides=[1, h_strd, w_strd, d_strd, 1],
                                            padding='SAME',
                                            name='{:s}_{:d}'.format(self.scope, index + 1))
                new_shape = max_pool.get_shape().as_list()
                spp = tf.reshape(max_pool, [-1, np.prod(new_shape[1:])])
                spps.append(spp)
            spps = tf.concat(values=spps, axis=-1)
            output_tensors.append(spps)
        output_tensor = tf.concat(values=output_tensors, axis=-1)
        return output_tensor

    def call(self, input_tensor):
        return self.build(input_tensors=input_tensor)


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
        kernel_width, kernel_height = self.kernel_shape

        # build kernel
        kernel_value = np.zeros((kernel_width, kernel_height, num_channels, num_channels), dtype=input_dtype_as_numpy)
        kernel_value[0, 0, :, :] = np.eye(num_channels, num_channels)
        kernel = tf.constant(kernel_value, name=self.scope + '/kernel')

        # do the un-pooling using conv2d_transpose
        output = tf.nn.conv2d_transpose(input_tensor,
                                        kernel,
                                        output_shape=output_shape,
                                        strides=(1, kernel_width, kernel_height, 1),
                                        padding='VALID')
        self.tensors['output'] = output
        return output

    def call(self, input_tensor, output_shape=None):
        return self.build(input_tensor, output_shape=output_shape)


class UnPooling3D(Layer):

    def __init__(self,
                 arguments,
                 ):
        Layer.__init__(self)
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
                                        strides=(1, kernel_width, kernel_height, kernel_depth, 1),
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
        self.tensor_shape = input_tensor.get_shape().as_list()
        output = tf.reshape(input_tensor, [-1, np.prod(self.tensor_shape[1:])])
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
        self.scope = arguments['scope']

    def build(self, input_tensor, output_shape=None):
        self.tensors['input'] = input_tensor

        output = tf.reshape(input_tensor, output_shape)
        self.tensors['output'] = output
        return output

    def call(self, input_tensor, output_shape):
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
            initializer = tf.contrib.layers.xavier_initializer()
            self.weights = tf.Variable(initial_value=initializer(self.kernel_shape),
                                       name=self.scope + '/kernel',
                                       trainable=True,
                                       )
        self.tensors['weight'] = self.weights

        # build bias
        if self.bias:
            self.bias = tf.Variable(tf.constant(0.01, shape=[self.kernel_shape[1]]),
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


from Data.utils_prepare_data import hdf5_handler


class Placeholders(Layer):
    """

    """

    def __init__(self,
                 arguments,
                 ):
        Layer.__init__(self)

        self.shape_file_path = arguments['shape_file_path'].encode()
        self.view_num = arguments['view_num']
        self.dataset = arguments['dataset']
        self.space = arguments['space']
        self.dtype = arguments['dtype']
        self.scope = arguments['scope']

        hdf5 = hdf5_handler(filename=self.shape_file_path)
        if self.dataset != '':
            self.ROIs = select_top_significance_ROIs(datasets=[self.dataset])[self.dataset]
        else:
            self.ROIs = range(self.view_num)
        self.input_shapes = [np.array(hdf5['{:s}/{:d}/size'.format(self.space, index + 1)],
                                      dtype=int)
                             for index in self.ROIs]
        channel = arguments['channel']
        none_list = [[None] for _ in self.input_shapes]
        for none, input_shape in zip(none_list, self.input_shapes):
            none.extend(input_shape)
            none.append(channel)
        self.input_shapes = none_list

    def build(self):
        return [tf.placeholder(dtype=self.dtype,
                               shape=input_shape,
                               name='{:s}_{:d}'.format(self.scope, index + 1))
                for index, input_shape in enumerate(self.input_shapes)]

    def call(self):
        return self.build()


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
    elif type == 'Placeholders':
        layer = Placeholders(arguments=arguments)
    elif type == 'Convolution2D':
        arguments['convolution'] = tf.nn.conv2d
        layer = Convolution(arguments=arguments)
    elif type == 'Convolution3D':
        arguments['convolution'] = tf.nn.conv3d
        layer = Convolution(arguments=arguments)
    elif type == 'Convolutions3D':
        arguments['convolution'] = tf.nn.conv3d
        layer = Convolutions(arguments=arguments)
    elif type == 'DeConvolution2D':
        arguments['convolution'] = tf.nn.conv2d_transpose
        layer = DeConvolution(arguments=arguments)
    elif type == 'DeConvolution3D':
        arguments['convolution'] = tf.nn.conv3d_transpose
        layer = DeConvolution(arguments=arguments)
    elif type == 'DepthwiseConvolution2D':
        arguments['convolution'] = tf.nn.depthwise_conv2d
        layer = Convolution(arguments=arguments)
    elif type == 'DepthwiseDeConvolution2D':
        arguments['convolution'] = tf.nn.conv2d_transpose
        layer = DepthwiseDeConvolution(arguments=arguments)
    elif type == 'DepthwiseConvolution3D':
        layer = DepthwiseConvolution(arguments=arguments)
    elif type == 'DepthwiseDeConvolution3D':
        arguments['convolution'] = tf.nn.conv3d_transpose
        layer = DepthwiseDeConvolution(arguments=arguments)
    elif type == 'MaxPooling':
        arguments['pool'] = tf.nn.max_pool
        layer = MaxPooling(arguments=arguments)
    elif type == 'MaxPooling3D':
        arguments['pool'] = tf.nn.max_pool3d
        layer = MaxPooling(arguments=arguments)
    elif type == 'SpatialPyramidPool3D':
        layer = SpatialPyramidPool3D(arguments=arguments)
    elif type == 'UnPooling':
        layer = UnPooling(arguments=arguments)
    elif type == 'UnPooling3D':
        layer = UnPooling3D(arguments=arguments)
    elif type == 'Fold':
        layer = Fold(arguments=arguments)
    elif type == 'Unfold':
        layer = Unfold(arguments=arguments)
    elif type == 'FullyConnected':
        layer = FullyConnected(arguments=arguments)
    elif type == 'Softmax':
        layer = Softmax(arguments=arguments)

    return layer
