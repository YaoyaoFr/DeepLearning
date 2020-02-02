import numpy as np
import tensorflow as tf

from tensorflow.python.training import moving_averages
from Dataset.utils import AAL
from Layer.LayerObject import LayerObject
from Model.utils_model import load_initial_value


class GraphConnected(LayerObject):
    required_pa = ['kernel_shape']

    def __init__(self,
                 arguments,
                 parameters: dict = None,
                 ):
        LayerObject.__init__(self)

        self.optional_pa.update({'bias': False,
                                 'padding': 'VALID',
                                 'activation': None,
                                 'scope': 'GraphNN',
                                 'batch_normalization': True,
                                 })
        self.tensors = {}

        self.parameters = self.set_parameters(arguments=arguments,
                                      parameters=parameters)

        # Mask the weight by adj_matrix with depth
        adj_matrix = AAL().get_adj_matrix(metric='Adjacent')
        self.parameters['adj_matrix'] = adj_matrix

        if 'depth' in self.parameters:
            graph_matrix = self.parameters['adj_matrix']
            for _ in range(self.parameters['depth'] - 1):
                graph_matrix = np.matmul(graph_matrix, self.parameters['adj_matrix'])
            graph_matrix[graph_matrix != 0] = 1
            self.parameters['graph_matrix'] = graph_matrix

        # build kernel
        assert len(self.parameters['kernel_shape']) == 2
        assert 'channel' in self.parameters, 'The number of channels must be fed.'

        weights = self.get_initial_weight()
        self.weight = [weights[..., index] * adj_matrix for index in range(self.parameters['channel'])]
        self.tensors['weight'] = self.weight

        for w in self.weight:
            L2 = tf.contrib.layers.l2_regularizer(self.parameters['L2_lambda'])(w)
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

    def build(self,
              input_tensor: tf.Tensor or list,
              input_place: tf.Tensor,
              training: tf.Tensor,
              ):
        assert len(input_tensor.get_shape().as_list()) == 4
        assert len(input_place.get_shape().as_list()) == 4 and input_place.get_shape().as_list()[-1] == 1
        channel = input_tensor.get_shape().as_list()[-1]
        assert channel == 1 or channel == self.parameters['channel'], \
            'The channel of input_placeholders must be 1 or {:d}, which is {:d}'.format(self.parameters['channel'], channel)

        # G_hats = tf.pow(x=tf.subtract(1 + 1e-5, tf.square(input_tensor)),
        #                 y=-1 / 2)
        # R_hats = tf.pow(x=tf.subtract(1 + 1e-5, tf.square(input_place)),
        #                 y=-1 / 2)

        self.tensors['G'] = input_tensor
        self.tensors['R'] = input_place
        # self.tensors['G_hat'] = G_hats
        # self.tensors['R_hat'] = R_hats

        output = []
        middle_variables = []
        for channel_index in range(self.parameters['channel']):
            G = input_tensor[..., 0] if channel == 1 else input_tensor[..., channel_index]
            R = input_place[..., 0]
            W = self.weight[channel_index]
            # G_hat = G_hats[..., 0] if channel == 1 else G_hats[..., channel_index]
            # R_hat = R_hats[..., 0]

            # item1 = W * R_hat
            # item2 = tf.matmul(G, item1)
            # item3 = G * G_hat
            # item4 = item1 * R
            # item5 = tf.matmul(item3, item4)
            item1 = W * R
            item2 = tf.matmul(G, item1)
            middle_variables.append(
                {
                    'item1': item1,
                    'item2': item2,
                    # 'item3': item3,
                    # 'item4': item4,
                    # 'item5': item5,
                })
            o = tf.expand_dims(R * self.parameters['graph_matrix'] - item2, axis=-1)
            output.append(o)
        output = tf.concat(output, axis=-1)
        self.tensors['middle_var'] = middle_variables
        self.tensors['output_matmul'] = output

        # bias
        if self.parameters['bias']:
            output = output + self.bias
            self.tensors['output_bias'] = output

        # batch_normalization
        if self.parameters['batch_normalization']:
            output = tf.cond(training,
                             lambda: self.batch_normalization(x=output,
                                                              scope=self.parameters['scope'] + '/bn',
                                                              is_training=True),
                             lambda: self.batch_normalization(x=output,
                                                              scope=self.parameters['scope'] + '/bn',
                                                              is_training=False))
            self.tensors.update(output)
            output = self.tensors['output_bn']

        # activation
        if self.parameters['activation']:
            output = self.parameters['activation'](output)
            self.tensors['output_activation'] = output

        self.tensors['output'] = output
        return output

    def call(self, **kwargs):
        return self.build(input_tensor=kwargs['input_tensor'],
                          input_place=kwargs['input_place'],
                          training=kwargs['training'])

    def generate_weight_by_adj(self,
                               adj_matrix: np.ndarray = None,
                               ):
        if self.parameters['load_weight']:
            initializer = load_initial_value(type='weight',
                                             name=self.parameters['scope'])
        else:
            # The graph convolution cannot be initialized by the conventional method, must be redesigned.
            # kernel_shape = [count_nonzero]
            # kernel_shape.extend(self.parameters['kernel_shape'][2:])
            initializer = self.get_initial_weight(kernel_shape=self.parameters['kernel_shape'])

        # sparse_all = matrix_to_sparse(adj_sum)
        #
        # slice_shape = self.parameters['kernel_shape'][2:]
        # indices_repeat = np.concatenate([sparse_all['indices'] for _ in range(np.prod(slice_shape))],
        #                                 axis=0)
        # for index in slice_shape:
        #     dim_indice = np.reshape(np.arange(index),
        #                             newshape=[index, 1],
        #                             )
        #     dim_indice = np.repeat(dim_indice,
        #                            axis=0,
        #                            repeats=int(len(sparse_all['indices']) * np.prod(slice_shape) / index),
        #                            )
        #     indices_repeat = np.concatenate((indices_repeat, dim_indice), axis=1)
        # self.weight = tf.sparse_to_dense(sparse_values=tf.reshape(tf.Variable(initial_value=initializer),
        #                                                           shape=[-1]),
        #                                  sparse_indices=indices_repeat,
        #                                  output_shape=self.parameters['kernel_shape'],
        #                                  )
        if adj_matrix is None:
            adj_matrix = AAL().get_adj_matrix(nearest_k=self.parameters['nearest_k'],
                                              # depth=self.parameters['depth'],
                                              depth=0,
                                              ).astype(dtype=np.float32)
        if len(np.shape(adj_matrix)) == 3:
            adj_sum = np.sum(adj_matrix, axis=1)
            # count_nonzero = np.count_nonzero(adj_sum)
            self.weight = tf.Variable(initial_value=initializer)
            self.weight = self.weight * tf.expand_dims(tf.expand_dims(adj_sum, -1), -1)
            weights = []
            for z in range(np.size(adj_matrix, -1)):
                adj_slice = tf.expand_dims(tf.expand_dims(adj_matrix[..., z], axis=-1), axis=-1)
                # sparse_slice = matrix_to_sparse(adj_slice)
                #
                # mask_indices, indices = sparse_mask(sparse_all, sparse_slice)
                #
                # indexed_slices = tf.IndexedSlices(values=self.weight,
                #                                   indices=tf.range(count_nonzero,
                #                                                    dtype=tf.int64),
                #                                   dense_shape=kernel_shape)
                # values = tf.reshape(tf.sparse_mask(a=indexed_slices,
                #                                    mask_indices=mask_indices).values,
                #                     shape=[-1, ])
                #
                # w = tf.sparse_to_dense(sparse_indices=indices_repeat,
                #                        sparse_values=values,
                #                        output_shape=self.parameters['kernel_shape'],
                #                        )
                w = self.weight * adj_slice
                weights.append(w)
        elif len(np.shape(adj_matrix)) == 2 and np.shape():
            weights = self.weight * np.expand_dims(np.expand_dims(adj_matrix, -1), -1)
        return weights

    def get_initial_weight(self,
                           mode: str = 'fan_in',
                           distribution: str = None,
                           ):
        kernel_shape = self.parameters['kernel_shape']
        channel = self.parameters['channel']

        assert len(kernel_shape) == 2, 'The rank of kernel must be 3.'
        fan_in = kernel_shape[0]
        fan_out = kernel_shape[1]

        if 'adj_matrix' in self.parameters and 'depth' in self.parameters:
            adj_matrix = self.parameters['adj_matrix']
            depth = self.parameters['depth']
            current_graph = adj_matrix
            for _ in range(depth - 2):
                current_graph = np.matmul(current_graph, adj_matrix)
            current_graph[current_graph > 0] = 1
            graph_matrix = np.matmul(current_graph, adj_matrix)
            fan_in = np.array([np.mean(graph_matrix[:, col][np.nonzero(graph_matrix[:, col])])
                               for col in range(kernel_shape[1])])
            scale = 1 / fan_in

        if distribution is None:
            distribution = 'uniform'
            if self.parameters['activation']:
                if self.parameters['activation']._tf_api_names[0] == 'nn.relu':
                    distribution = 'norm'

        weights = []
        for col in range(kernel_shape[1]):
            shape = [kernel_shape[1], channel]
            s = scale[col]
            if distribution == 'uniform':
                limit = np.sqrt(3. * s)
                initial_value = tf.random_uniform(shape=shape,
                                                  minval=-limit,
                                                  maxval=limit)
            elif distribution == 'norm':
                stddev = np.sqrt(2. * s) / .87962566103423978
                initial_value = tf.truncated_normal(shape=shape,
                                                    stddev=stddev)

            weights.append(tf.Variable(initial_value=initial_value))
        weights = tf.concat(weights,
                            axis=1,
                            name=self.parameters['scope'] + 'weight')

        return weights

    def batch_normalization(self,
                            x: tf.Tensor,
                            axis: int = -1,
                            scope: str = 'bn',
                            decay: float = 0.99,
                            epsilon: float = 1e-30,
                            is_training: np.bool = True,
                            reuse: tf.compat.v1.variable_scope = tf.compat.v1.AUTO_REUSE,
                            ):
        """
        Performs a batch normalization layer

        Args:
            x: input tensor
            scope: scope name
            is_training: python boolean value
            epsilon: the variance epsilon - a small float number to avoid dividing by 0
            decay: the moving average decay
            reuse:
            axis:

        Returns:
            The ops of a batch normalization layer
        """
        with tf.variable_scope(scope, reuse=reuse):
            shape = x.get_shape().as_list()
            # gamma: a trainable scale factor
            gamma = tf.get_variable("gamma", shape[axis],
                                    initializer=tf.constant_initializer(1.0),
                                    trainable=True)
            # beta: a trainable shift value
            beta = tf.get_variable("beta", shape[axis], initializer=tf.constant_initializer(0.0), trainable=True)
            moving_avg = tf.get_variable("moving_avg", shape[axis], initializer=tf.constant_initializer(0.0),
                                         trainable=False)
            moving_var = tf.get_variable("moving_var", shape[axis], initializer=tf.constant_initializer(1.0),
                                         trainable=False)

            results = {}
            if is_training:
                # tf.nn.moments == Calculate the mean and the variance of the tensor x
                mean, var = tf.nn.moments(x, list(np.arange(len(shape) - 1)))
                results['mean0'] = mean
                results['var0'] = var
                if 'graph_matrix' in self.parameters:
                    matrix = self.parameters['graph_matrix']
                    N = np.prod(np.shape(matrix))
                    N_1 = np.count_nonzero(matrix)
                    a = N / N_1
                    var = a * (var + tf.pow(mean, 2) * (1 - a))
                    mean *= a

                update_moving_mean = moving_averages.assign_moving_average(moving_avg, mean, decay)
                update_moving_var = moving_averages.assign_moving_average(moving_var, var, decay)
                results['mean'] = mean
                results['var'] = var
                control_inputs = [update_moving_mean, update_moving_var]
            else:
                mean = moving_avg
                var = moving_var
                results['mean0'] = mean
                results['var0'] = var
                results['mean'] = mean
                results['var'] = var
                control_inputs = []
            with tf.control_dependencies(control_inputs):
                output = tf.nn.batch_normalization(x=x,
                                                   mean=mean,
                                                   variance=var,
                                                   offset=beta,
                                                   scale=gamma,
                                                   variance_epsilon=epsilon)
                results['output_bn_without_mask'] = output
                try:
                    graph_matrix = self.parameters['graph_matrix']
                    graph_matrix = np.expand_dims(np.expand_dims(graph_matrix, axis=0), axis=-1)
                    output = output * graph_matrix
                except KeyError:
                    pass
                results['output_bn'] = output
        return results
