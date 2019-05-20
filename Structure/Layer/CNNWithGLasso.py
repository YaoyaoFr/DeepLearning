import tensorflow as tf

from Structure.Layer.LayerObject import LayerObject
from Structure.utils_structure import load_initial_value


class EdgeToEdgeWithGLasso(LayerObject):
    """

    """
    required_pa = ['kernel_shape',
                   'n_class']

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
                                 'scope': 'E2EGLasso',
                                 })
        self.tensors = {}

        self.pa = self.set_parameters(arguments=arguments,
                                      parameters=parameters)

        n_features, n_features, in_channels, out_channels = self.pa['kernel_shape']
        out_channels *= self.pa['n_class']

        L = tf.Variable(dtype=tf.float32,
                        initial_value=tf.truncated_normal(
                            shape=[in_channels,
                                   out_channels,
                                   int(n_features * (n_features + 1) / 2)],
                            mean=0,
                            stddev=0.01),
                        )

        # build weights
        L_tril = tf.contrib.distributions.fill_triangular(L) + \
                 tf.eye(num_rows=n_features,
                        batch_shape=[in_channels, out_channels])

        self.tensors['L'] = L_tril

        weight = tf.transpose(a=tf.matmul(L_tril, tf.transpose(L_tril, perm=[0, 1, 3, 2]), name='weight'),
                              perm=[3, 2, 0, 1])
        self.tensors['weight'] = weight

        # build bias
        if self.pa['bias']:
            if self.pa['load_bias']:
                initializer = load_initial_value(type='bias',
                                                 name=self.pa['scope'])
            else:
                initializer = tf.constant(0.0, shape=[out_channels])
            self.bias = tf.Variable(initial_value=initializer,
                                    name=self.pa['scope'] + '/bias',
                                    )
            self.tensors['bias'] = self.bias

    def build(self, input_tensor, output_tensor=None, training=True):
        self.tensors['input'] = input_tensor
        shape = input_tensor.shape.as_list()
        if len(shape) == 3:
            input_tensor = tf.expand_dims(input_tensor, axis=-1)

        # convolution
        weight = self.tensors['weight']

        # Since the weights are naturally symmetric, it does not need to transpose
        # weight = tf.transpose(weight, perm=[1, 0, 2, 3])

        weight_slices = tf.split(weight, axis=0, num_or_size_splits=self.pa['kernel_shape'][0])
        output = []
        for weight_slice in weight_slices:
            feature_map = self.pa['conv_fun'](input_tensor,
                                              weight_slice,
                                              strides=self.pa['strides'],
                                              padding=self.pa['padding'],
                                              )
            output.append(feature_map)
        output = tf.concat(output, axis=2)
        self.tensors['output_conv'] = output

        # Build sparse inverse covariance matrix regularization
        SICE_regularizer = build_SICE_regularizer(weight, self.tensors['L'], output)
        SICE_regularizer = tf.transpose(tf.reshape(SICE_regularizer,
                                                   shape=[-1, self.pa['n_class'], self.pa['kernel_shape'][3]]),
                                        perm=[2, 0, 1])
        self.tensors['SICE_regularizer'] = tf.reshape(tf.transpose(SICE_regularizer, perm=[1, 2, 0]),
                                                      shape=[-1, self.pa['kernel_shape'][3] * self.pa['n_class']])

        SICE_regularizer = tf.cond(training,
                                   lambda: SICE_regularizer * output_tensor,
                                   lambda: SICE_regularizer)

        SICE_regularizer = tf.reshape(tf.transpose(SICE_regularizer, perm=[1, 2, 0]),
                                      shape=[-1, self.pa['kernel_shape'][3] * self.pa['n_class']])
        self.tensors['SICE_regularizer_masked'] = SICE_regularizer

        regularizer_softmax = tf.nn.softmax(-SICE_regularizer)
        self.tensors['softmax_regularizer'] = regularizer_softmax

        tf.add_to_collection('L1_loss', tf.reduce_sum(tf.multiply(SICE_regularizer, regularizer_softmax)))

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

    def call(self, input_tensor, output_tensor=None, training=True):
        return self.build(input_tensor=input_tensor,
                          output_tensor=output_tensor,
                          training=training)


class EdgeToNodeWithGLasso(LayerObject):
    """

    """
    required_pa = ['kernel_shape', 'n_class']

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
                                 'scope': 'E2NGLasso',
                                 })
        self.tensors = {}

        self.pa = self.set_parameters(arguments=arguments,
                                      parameters=parameters)

        n_features, n_features, in_channels, out_channels = self.pa['kernel_shape']
        out_channels *= self.pa['n_class']

        L = tf.Variable(dtype=tf.float32,
                        initial_value=tf.truncated_normal(
                            shape=[in_channels,
                                   out_channels,
                                   int(n_features * (n_features + 1) / 2)],
                            mean=0,
                            stddev=0.001),
                        )

        # build weights
        L_tril = tf.contrib.distributions.fill_triangular(L) + \
                 0.2 * tf.eye(num_rows=n_features,
                              batch_shape=[in_channels, out_channels])

        self.tensors['L'] = L_tril

        weight = tf.transpose(a=tf.matmul(L_tril, tf.transpose(L_tril, perm=[0, 1, 3, 2]), name='weight'),
                              perm=[3, 2, 0, 1])
        self.tensors['weight'] = weight

        # build bias

        if self.pa['bias']:
            if self.pa['load_bias']:
                initializer = load_initial_value(type='bias',
                                                 name=self.pa['scope'])
            else:
                initializer = tf.constant(0.0, shape=[out_channels])
            self.bias = tf.Variable(initial_value=initializer,
                                    name=self.pa['scope'] + '/bias',
                                    )
            self.tensors['bias'] = self.bias

    def build(self, input_tensor, output_tensor, training=True):
        self.tensors['output_tensor'] = output_tensor
        self.tensors['input'] = input_tensor
        shape = input_tensor.shape.as_list()
        if len(shape) == 3:
            input_tensor = tf.expand_dims(input_tensor, axis=-1)

        # convolution
        weight = self.tensors['weight']

        # Since the weights are naturally symmetric, it does not need to transpose
        # weight = tf.transpose(weight, perm=[1, 0, 2, 3])

        input_slices = tf.split(input_tensor, axis=1, num_or_size_splits=self.pa['kernel_shape'][0])
        weight_slices = tf.split(weight, axis=0, num_or_size_splits=self.pa['kernel_shape'][0])

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

        # Build sparse inverse covariance matrix regularization
        regularizer_results = build_SICE_regularizer(weight=weight,
                                                     L=self.tensors['L'],
                                                     output=output)
        self.tensors.update(regularizer_results)

        SICE_regularizer = self.tensors['Trace']

        # Reshape the regularizer to shape of [batch_size, n_class, out_channels]
        SICE_regularizer = tf.transpose(tf.reshape(SICE_regularizer,
                                                   shape=[-1, self.pa['n_class'], self.pa['kernel_shape'][3]]),
                                        perm=[2, 0, 1])

        regularizer_softmax = tf.nn.softmax(-SICE_regularizer, axis=0)
        regularizer_softmax = tf.cond(training,
                                      lambda: regularizer_softmax * output_tensor,
                                      lambda: regularizer_softmax)
        regularizer_softmax = tf.reshape(tf.transpose(regularizer_softmax, perm=[1, 2, 0]),
                                         shape=[-1, self.pa['n_class'] * self.pa['kernel_shape'][3]])
        self.tensors['Regularizer softmax'] = regularizer_softmax

        SICE_regularizer = self.tensors['Log determinant'] + \
                           self.tensors['Trace'] + \
                           self.pa['lambda'] * self.tensors['Norm 1']
        self.tensors['SICE regularizer'] = SICE_regularizer

        tf.add_to_collection('L1_loss', tf.reduce_sum(tf.multiply(SICE_regularizer, regularizer_softmax)))
        # tf.add_to_collection('L1_loss', tf.reduce_sum(SICE_regularizer))

        # output = tf.transpose(tf.multiply(tf.transpose(output, perm=[1, 2, 0, 3]), regularizer_softmax),
        #                       perm=[2, 0, 1, 3])
        # self.tensors['output_regularized_conv'] = output

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

    def call(self, input_tensor, output_tensor, training=True):
        return self.build(input_tensor, output_tensor, training=training)


def build_SICE_regularizer(weight, L, output):
    logdet = -tf.reduce_sum(tf.log(tf.square(tf.matrix_diag_part(L))))
    # trace = tf.trace(tf.transpose(output, perm=[0, 3, 1, 2]))
    trace = tf.reduce_sum(output, axis=(1, 2))
    norm_1 = tf.reduce_sum(input_tensor=tf.abs(weight), axis=(0, 1, 2))

    return {'Log determinant': logdet,
            'Norm 1': norm_1,
            'Trace': trace,
            }
