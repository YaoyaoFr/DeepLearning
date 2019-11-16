import numpy as np
import tensorflow as tf

a = tf.placeholder(shape=[64, 1, 90, 1], dtype=tf.float32)
w = tf.placeholder(shape=[90, 1, 1, 45], dtype=tf.float32)
out = tf.nn.depthwise_conv2d(input=a,
                             filter=w,
                             data_format='NCHW',
                             strides=[1, 1, 1, 1],
                             padding='VALID')
out_squeeze = tf.squeeze(out)

a_value = np.random.random(size=[64, 90, 1, 45])
w_value = np.random.random(size=[90, 1, 1, 45])
target_value = np.zeros(shape=[64, 45])
for i in range(64):
    for j in range(45):
        a_row = np.squeeze(a_value[i, ...])
        w_row = np.squeeze(w_value[..., j])
        target_value[i, j] = np.matmul(a_row, w_row)
with tf.Session() as sess:
    out_value = sess.run(fetches=out_squeeze,
                         feed_dict={a: a_value, w: w_value})
    res = out_value - target_value
