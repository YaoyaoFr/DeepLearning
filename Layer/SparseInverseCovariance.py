import numpy as np
import tensorflow as tf

from Layer.LayerObject import LayerObject
from Model.utils_model import load_initial_value


class EdgeToNodeWithGLasso(LayerObject):
    """

    """
    required_pa = ['kernel_shape', 'n_class']

    def __init__(self,
                 arguments: dict,
                 parameters: dict = None,
                 ):
        LayerObject.__init__(self)
        self.optional_pa.update({'scope': 'SICE',
                                 })
        self.tensors = {}

        self.pa = self.set_parameters(arguments=arguments,
                                      parameters=parameters)

        n_features, n_features, in_channels, out_channels = self.pa['kernel_shape']
        out_channels *= self.pa['n_class']

        initializer = self.get_initial_weight(kernel_shape=[n_features, n_features, in_channels, out_channels],
                                              loc=0)
        L = tf.Variable(dtype=tf.float32,
                        initial_value=initializer,
                        )

        # build weights
        L_tril = tf.contrib.distributions.fill_triangular(L) + \
                 tf.eye(num_rows=n_features,
                        batch_shape=[in_channels, out_channels])
        self.tensors['L'] = L_tril

        self.weight_SICE = tf.transpose(a=tf.matmul(a=L_tril,
                                                    b=tf.transpose(L_tril, perm=[0, 1, 3, 2]),
                                                    name=self.pa['scope'] + 'weight_SICE'),
                                        perm=[3, 2, 0, 1])
        self.tensors['weight_SICE'] = self.weight_SICE

    def build(self, *args, **kwargs):
        covariance_tensor = kwargs['sample_covariance']
        output_tensor = kwargs['output']
        training = kwargs['training']

        self.tensors['output_tensor'] = output_tensor
        self.tensors['sample_covariance'] = covariance_tensor
        shape = covariance_tensor.shape.as_list()
        assert len(shape) == 4, 'The rank of input tensor must be 4 but go {:d}.'.format(len(shape))

        # convolution
        weight = self.tensors['weight']

        # Since the weights are naturally symmetric, it does not need to transpose
        # weight = tf.transpose(weight, perm=[1, 0, 2, 3])

        covariance_slices = tf.split(covariance_tensor, axis=1, num_or_size_splits=self.pa['kernel_shape'][0])
        weight_SICE_slices = tf.split(self.weight_SICE, axis=0, num_or_size_splits=self.pa['kernel_shape'][0])

        # Build sparse inverse covariance matrix regularization

        output_SICE = []
        for covariance_slice, weight_SICE_slice in zip(covariance_slices,
                                                       weight_SICE_slices):
            feature_map_SICE = self.pa['conv_fun'](covariance_slice,
                                                   weight_SICE_slice,
                                                   strides=self.pa['strides'],
                                                   padding=self.pa['padding'],
                                                   )
            output_SICE.append(feature_map_SICE)
        output_SICE = tf.concat(output_SICE, axis=1)
        self.tensors['output_SICE'] = output_SICE

        regularizer_results = build_SICE_regularizer(weight=weight,
                                                     L=self.tensors['L'],
                                                     output=output_SICE)
        self.tensors.update(regularizer_results)

        # mean, var = tf.nn.moments(SICE_regularizer, axes=[1], keep_dims=True)
        # SICE_regularizer_norm = tf.subtract(SICE_regularizer, mean)
        # SICE_regularizer_norm = tf.div(SICE_regularizer_norm, tf.sqrt(var))

        SICE_regularizer = self.tensors['Log determinant'] + self.tensors['Trace']
        self.tensors['SICE regularizer'] = SICE_regularizer
        SICE_regularizer_norm = SICE_regularizer

        # Reshape the regularizer to shape of [batch_size, n_class, out_channels] and softmax
        regularizer_softmax = tf.transpose(tf.nn.softmax(tf.reshape(SICE_regularizer_norm,
                                                                    shape=[-1, self.pa['n_class'],
                                                                           self.pa['kernel_shape'][3]]),
                                                         axis=-1),
                                           perm=[2, 0, 1])

        # label_unsupervise = tf.reduce_sum(tf.reshape(SICE_regularizer_norm,
        #                                              shape=[-1, self.pa['n_class'], self.pa['kernel_shape'][3]]),
        #                                   axis=-1)
        #
        # label_unsupervise = tf.cast(tf.concat(
        #     (tf.expand_dims(tf.argmin(label_unsupervise, axis=-1), axis=-1),
        #      tf.expand_dims(tf.argmax(label_unsupervise, axis=-1), axis=-1)),
        #     axis=-1), dtype=tf.float32)
        # self.tensors['Label unsupervise'] = label_unsupervise

        # Mask with label
        regularizer_softmax = tf.cond(training,
                                      lambda: regularizer_softmax * output_tensor,
                                      lambda: regularizer_softmax
                                      )
        # Reshape to previous shape
        regularizer_softmax = tf.reshape(tf.transpose(regularizer_softmax, perm=[1, 2, 0]),
                                         shape=[-1, self.pa['n_class'] * self.pa['kernel_shape'][3]])
        self.tensors['Regularizer softmax'] = regularizer_softmax

        #
        SICE_regularizer = self.tensors['Log determinant'] + \
                           self.tensors['Trace'] + \
                           self.pa['lambda'] * self.tensors['Norm 1']
        tf.add_to_collection('SICE_loss', tf.reduce_mean(tf.multiply(SICE_regularizer, regularizer_softmax)))
        # tf.add_to_collection('L1_loss', self.pa['lambda'] * tf.reduce_mean(self.tensors['Norm 1']))

        # output = tf.transpose(tf.multiply(tf.transpose(output, perm=[1, 2, 0, 3]), regularizer_softmax),
        #                       perm=[2, 0, 1, 3]) * self.pa['kernel_shape'][3] * self.pa['n_class']
        # self.tensors['output_regularized_conv'] = output

        return self.weight_SICE


    def get_initial_weight(self,
                           kernel_shape: list,
                           mode: str = 'fan_in',
                           distribution: str = 'norm',
                           loc: float = 1,
                           ):
        rank = len(kernel_shape)
        assert rank in {2, 3, 4, 5}, 'The rank of kernel expected in {2, 3, 4, 5} but go {:d}'.format(rank)
        assert 'activation' in self.pa, 'The activation function must be given. '

        if rank == 2:
            fan_in = kernel_shape[0] + kernel_shape[1]
            fan_out = kernel_shape[1]
        else:
            receptive_field_size = np.prod(kernel_shape[:-2])
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
        else:
            raise TypeError('The distribution of EdgeToNodeWithGLasso only support norm')

        [size, _, in_channels, out_channels] = kernel_shape
        tril_vec = np.zeros(shape=[in_channels, out_channels, int(size * (size + 1) / 2), ])
        stddev = np.sqrt(stddev)

        for in_channel in range(in_channels):
            for out_channel in range(out_channels):
                for i in range(int(size / 2) + 1):
                    tril_vec[in_channel, out_channel, i * size + i - 1] = np.random.normal(scale=stddev)
                    tril_vec[in_channel, out_channel, i * size] = np.random.normal(loc=loc, scale=stddev)
                    tril_vec[in_channel, out_channel, i * size - 1] = np.random.normal(loc=loc, scale=stddev)
                    tril_vec[in_channel, out_channel, i * size - 1 - size + i] = np.random.normal(scale=stddev)

        return tril_vec


def build_SICE_regularizer(weight, L, output):
    logdet = -tf.reduce_sum(tf.log(tf.square(tf.matrix_diag_part(L))), axis=(0, 2))
    # trace = tf.trace(tf.transpose(output, perm=[0, 3, 1, 2]))
    trace = tf.reduce_sum(output, axis=(1, 2))
    norm_1 = tf.reduce_sum(input_tensor=tf.abs(weight), axis=(0, 1, 2))

    return {'Log determinant': logdet,
            'Norm 1': norm_1,
            'Trace': trace,
            }
