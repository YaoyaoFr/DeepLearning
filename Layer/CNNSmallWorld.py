import numpy as np
import tensorflow as tf

from tensorflow.python.ops import nn_ops
from Layer.LayerObject import LayerObject
from Model.utils_model import load_initial_value


class EdgeToCluster(LayerObject):
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
                                 'padding': 'VALID',
                                 'scope': 'E2C',
                                 'conv_fun': tf.nn.conv2d,
                                 'use_bias': True,
                                 })
        self.tensors = {}

        self.pa = self.set_parameters(arguments=arguments,
                                      parameters=parameters)
        # Cluster kernel
        [width, height, in_channels, out_channels] = self.pa['kernel_shape']
        kernel_shape_row = [1, height, in_channels, 1]
        self.weight_row = tf.Variable(initial_value=self.get_initial_weight(kernel_shape=kernel_shape_row),
                                      name=self.pa['scope'] + '/kernel_row',
                                      )
        self.tensors['weight_row'] = self.weight_row
        L2_weight_row = tf.contrib.layers.l2_regularizer(self.pa['L2_lambda'])(self.weight_row)
        tf.add_to_collection('L2_loss', L2_weight_row)

        kernel_shape_col = [width, 1, 1, out_channels]
        self.weight_col = tf.Variable(initial_value=self.get_initial_weight(kernel_shape=kernel_shape_col),
                                      name=self.pa['scope'] + '/kernel_col')
        self.tensors['weight_col'] = self.weight_col
        L2_weight_col = tf.contrib.layers.l2_regularizer(self.pa['L2_lambda'])(self.weight_col)
        tf.add_to_collection('L2_loss', L2_weight_col)

        # build bias
        out_channels = self.pa['kernel_shape'][-1]

        if self.pa['bias']:
            initializer = tf.constant(0.0, shape=[out_channels, ])
            if self.pa['load_bias']:
                initializer = load_initial_value(type='bias',
                                                 name=self.pa['scope'])

            self.bias = tf.Variable(initial_value=initializer,
                                    name=self.pa['scope'] + '/bias',
                                    )
            self.tensors['bias'] = self.bias

    def build(self, *args, **kwargs):
        # Get adjacency matrix from inputs, which should be a Tensor or a dictionary
        adjacency_matrix = kwargs['input_tensor'] if 'input_tensor' in kwargs else kwargs['adjacency_matrix']
        self.tensors['input'] = adjacency_matrix
        self.tensors['adjacency_matrix'] = adjacency_matrix

        [_, _, num_nodes, _] = adjacency_matrix.shape.as_list()

        # Cluster adjacency matrix calculation
        neighbour_vectors = tf.split(tf.abs(adjacency_matrix), num_or_size_splits=num_nodes, axis=2)

        node_features = []
        cluster_adjacency_list = []
        for neighbour_vector in neighbour_vectors:
            cross_multiply = tf.transpose(tf.matmul(tf.transpose(neighbour_vector, perm=[0, 3, 1, 2]),
                                                    tf.transpose(neighbour_vector, perm=[0, 3, 2, 1])),
                                          perm=[0, 2, 3, 1])
            cluster_adjacency = tf.multiply(adjacency_matrix, cross_multiply)
            cluster_adjacency_list.append(cluster_adjacency)
            node_feature_row = self.pa['conv_fun'](input=cluster_adjacency,
                                                   filter=self.weight_row,
                                                   strides=self.pa['strides'],
                                                   padding=self.pa['padding'])
            node_feature = self.pa['conv_fun'](input=node_feature_row,
                                               filter=self.weight_col,
                                               strides=self.pa['strides'],
                                               padding=self.pa['padding'])
            if self.pa['use_bias']:
                node_feature = tf.nn.bias_add(node_feature, self.bias)
            node_features.append(node_feature)
        node_features = tf.concat(node_features, axis=-2)
        node_features = tf.squeeze(node_features, axis=-3)
        self.tensors['cluster_adjacency'] = cluster_adjacency_list

        if self.pa['activation']:
            node_features = self.pa['activation'](node_features)

        self.tensors['node_features'] = node_features


class SelfAttentionGraphPooling(LayerObject):
    """

    """
    required_pa = ['kernel_shape']

    def __init__(self,
                 arguments: dict,
                 parameters: dict = None,
                 ):
        LayerObject.__init__(self)
        self.optional_pa.update({'L2_lambda': 5e-3,
                                 'activation': None,
                                 'scope': 'SAGPool',
                                 })
        self.tensors = {}
        self.pa = self.set_parameters(arguments=arguments,
                                      parameters=parameters)

        # Attention kernel
        initializer_att = self.get_initial_weight(kernel_shape=self.pa['kernel_shape'])
        self.weight_att = tf.Variable(initial_value=initializer_att,
                                      name=self.pa['scope'] + '/kernel',
                                      )
        self.tensors['weight_att'] = self.weight_att
        L2_att = tf.contrib.layers.l2_regularizer(self.pa['L2_lambda'])(self.weight_att)
        tf.add_to_collection('L2_loss', L2_att)

    def build(self, *args, **kwargs):
        node_features = kwargs['node_features']
        adjacency_matrix = kwargs['adjacency_matrix']
        self.tensors['input'] = adjacency_matrix

        [_, _, num_nodes, _] = adjacency_matrix.shape.as_list()

        # Self-attention graph pooling
        adjacency_matrix = tf.squeeze(adjacency_matrix, axis=-1)
        y = tf.matmul(adjacency_matrix, node_features)
        cluster_mapping = tf.matmul(tf.reshape(y,
                                               shape=[-1, self.pa['kernel_shape'][0]]),
                                    self.weight_att)

        if self.pa['activation']:
            cluster_mapping = self.pa['activation'](cluster_mapping)

        cluster_mapping = tf.reshape(cluster_mapping, shape=[-1, num_nodes, self.pa['kernel_shape'][-1]])
        cluster_mapping = tf.nn.softmax(cluster_mapping, axis=-1)
        self.tensors['cluster_mapping'] = cluster_mapping

        self.add_loss_function()

        adjacency_matrix = tf.matmul(tf.matmul(a=tf.transpose(cluster_mapping, perm=[0, 2, 1]),
                                               b=adjacency_matrix),
                                     b=cluster_mapping)
        adjacency_matrix = tf.expand_dims(adjacency_matrix, axis=-1)
        node_features = tf.matmul(tf.transpose(cluster_mapping, perm=[0, 2, 1]),
                                               node_features)
        self.tensors['node_features'] = node_features
        self.tensors['adjacency_matrix'] = adjacency_matrix

        self.tensors['output'] = node_features
        return adjacency_matrix

    def add_loss_function(self):
        adjacency_matrix = self.tensors['input']
        cluster_mapping = self.tensors['cluster_mapping']

        link_loss = tf.squeeze(adjacency_matrix, axis=-1) - \
                    tf.matmul(cluster_mapping, tf.transpose(cluster_mapping, perm=[0, 2, 1]))
        link_loss = tf.norm(link_loss, ord='fro', axis=[1, 2])

        EPS = 1e-30

        entropy_loss = tf.reduce_mean(tf.reduce_sum(-tf.multiply(cluster_mapping,
                                                                 tf.log(cluster_mapping + EPS)), axis=-1))

        loss = link_loss + entropy_loss
        tf.add_to_collection('Mapping_loss', loss)
