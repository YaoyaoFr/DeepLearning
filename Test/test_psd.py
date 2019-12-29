import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.datasets import make_sparse_spd_matrix

out_channels = 16
size = 30

tril_vec = np.zeros(shape=[int(size * (size + 1) / 2), ])
for i in range(int(size / 2) + 1):
    tril_vec[i * size + i - 1] = 1
    tril_vec[i * size] = 1
    tril_vec[i * size - 1] = 1
    tril_vec[i * size - 1 - size + i] = 1

L_initial = tf.Variable(dtype=tf.float32, initial_value=tril_vec)
L = tf.contrib.distributions.fill_triangular(L_initial)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    L_val = sess.run(L)

L_list = []
weight_list = []
for i in range(out_channels):
    L = np.zeros(shape=[size, size])
    L_column = np.random.normal(loc=0, scale=0.01, size=[size, 1])
    L_diagnal = np.random.normal(loc=0, scale=0.01, size=[size, 1])
    for i in range(size):
        L[i, 0] = L_column[i]
        if i != 0:
            L[i, i] = L_diagnal[i]

    L_list.append(np.expand_dims(L, axis=-1))
    weight = np.matmul(L, L.T)
    weight_list.append(np.expand_dims(weight, axis=-1))

L_list = np.concatenate(L_list, axis=-1)
weight_list = np.concatenate(weight_list, axis=-1)

weight_mean = np.mean(weight_list, axis=-1)
weight_std = np.std(weight_list, axis=-1)

pass
