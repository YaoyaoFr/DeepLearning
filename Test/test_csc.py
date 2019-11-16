import numpy as np
import tensorflow as tf

from Layer.CNNSmallWorld import EdgeToCluster, SelfAttentionGraphPooling

B = 64
N = 90
I = 1
O = 32

e2ncs1_pa = {'kernel_shape': [N, N, I, O],
             'placeholders': ['input_tensor', 'adjacency_matrix'],
             'activation': tf.nn.sigmoid,
             }
e2ncs1_pa = {'kernel_shape': [N, N, I, O],
             'placeholders': ['input_tensor', 'adjacency_matrix'],
             'activation': tf.nn.sigmoid,
             }

sag1_pa = {'kernel_shape': [90, 45],
           'activation': tf.nn.relu,
           'tensors': ['output', 'adjacency_matrix']
           }

e2ncs1 = EdgeToCluster(arguments=e2ncs1_pa)
sag1 = SelfAttentionGraphPooling(arguments=sag1_pa)

input_tensor = tf.Variable(initial_value=tf.random_normal(shape=[B, N, N, I], dtype=tf.float32),
                           dtype=tf.float32)
placeholders = {'input_tensor': input_tensor,
                'adjacency_matrix': input_tensor,
                }

e2ncs1(input_tensor=None, placeholders=placeholders)
sag1(tensors=e2ncs1.tensors, placeholders=placeholders)
