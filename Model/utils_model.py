import os
import shutil

import numpy
import numpy as np
import scipy.io as sio
import tensorflow as tf
from scipy import io as sio

from Log.log import Log


def check_dataset(data: dict,
                  tag: str,
                  data_placeholder: dict):
    sample_size = []
    for key in data_placeholder:
        name = '{:s} {:s}'.format(tag, key)
        assert (name in data), '{:s} is not in given dataset.'.format(name)

        size = len(data[name])
        if sample_size and size not in sample_size:
            raise TypeError(
                'The sample size of {:s} is different from the others.'.format(name))
        else:
            sample_size.append(size)

    return size


def get_weights(op_layers, sess):
    weight_all = {}
    for layer in op_layers:
        weight_layer = {}
        tensors = layer.tensors
        for tensor_name in tensors:
            if 'weight' in tensor_name:
                weight_layer[tensor_name] = tensors[tensor_name]
        weight_all[layer.pa['scope']] = weight_layer

    weight_all = sess.run(fetches=weight_all)
    return weight_all


def save_weights(sess,
                 op_layers,
                 save_dir: str = None,
                 run_time: int = None,
                 fold_index: int = None,
                 pre_fix: str = 'trained',
                 ):
    """
    Save weight in each layer to .mat file
    :param sess: Session
    :param op_layers: operation layers
    :param save_dir: The parent directory path
    :param run_time: Run time
    :param fold_index: Fold index
    :param pre_fix: Pre fix
    :return:
    """

    if save_dir is None:
        save_dir = 'F:/OneDriveOffL/Data/Result/Net'

    if run_time is None:
        save_path = save_dir
    else:
        save_path = os.path.join(save_dir, 'time {:d}'.format(run_time))
        if fold_index is not None:
            save_path = os.path.join(save_path, 'fold {:d}'.format(fold_index))

    if not os.path.exists(save_path):
        os.makedirs(save_path, exist_ok=True)

    weights = get_weights(op_layers, sess)

    sio.savemat(os.path.join(
        save_path, '{:s}_weights.mat'.format(pre_fix)), weights)
    print('Save weights in {:s}.'.format(save_path))


def get_metrics(output_tensor: tf.Tensor,
                ground_truth: tf.Tensor,
                task: str = 'prediction',
                ) -> dict:
    """
    calculate several metrics include mean square error, accuracy, precision, recall, specificity and F1 score.
    :param output_tensor: Output of the neural network
    :param ground_truth:
    :param task: 'prediction' or 'regression'
    :return: dictionary contains all the metrics
    """

    assert task in [
        'regression', 'prediction'], 'Only support regression or prediction task but get {:s}'.format(task)

    metrics = {}

    if task == 'regression':
        metrics['MSE'] = tf.reduce_mean(tf.reduce_sum(tf.squared_difference(output_tensor,
                                                                            ground_truth), axis=1))
    elif task == 'prediction':
        metrics['Cross Entropy'] = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=ground_truth,
                                                                                             logits=output_tensor))
        predict = tf.argmax(input=tf.nn.softmax(output_tensor), axis=-1)
        labels = tf.argmax(input=ground_truth, axis=1)
        metrics['Accuracy'] = tf.reduce_mean(tf.cast(x=tf.equal(x=predict, y=labels),
                                                     dtype=tf.float32))

        TP = tf.count_nonzero(predict * labels)
        TN = tf.count_nonzero((predict - 1) * (labels - 1))
        FP = tf.count_nonzero(predict * (labels - 1))
        FN = tf.count_nonzero((predict - 1) * labels)

        precision = 0
        recall = 0
        specificity = 0
        f1 = 0

        try:
            precision = TP / (TP + FP)
            recall = TP / (TP + FN)
            specificity = TN / (TN + FP)
            f1 = 2 * precision * recall / (precision + recall)
        except ZeroDivisionError:
            pass

        metrics.update({'Precision': precision,
                        'Recall': recall,
                        'Specificity': specificity,
                        'F1 Score': f1, })

    return metrics


def feed_data_list(placeholders: list, datas: list):
    if len(placeholders) != len(datas):
        raise TypeError('Length of placeholders and datas dosen\'t match!')

    feed_dict = {}
    for placeholder, data in zip(placeholders, datas):
        feed_dict[placeholder] = data

    return feed_dict


def get_initial_weight(kernel_shape: list,
                       activation: str = 'nn.relu',
                       ):
    std = 0.001
    if activation == 'nn.relu' or activation == '':
        if len(kernel_shape) >= 4:
            n = np.prod(kernel_shape) * kernel_shape[-2]
        elif len(kernel_shape) == 2:
            n = np.sum(kernel_shape)
        std = np.sqrt(2 / n)
    initial_value = tf.truncated_normal(shape=kernel_shape, stddev=std)
    return initial_value


def load_initial_value(type: str, name: str):
    initial_value = sio.loadmat('{:}.mat'.format(type))[name]
    return initial_value


def upper_triangle(data: dict):
    transoformed_data = {}
    for key in data:
        if 'data' in key:
            d = np.squeeze(data[key], axis=-1)
            assert d.ndim == 3

            [_, width, height] = np.shape(d)
            assert height == width, '{:s} must be square matrix but get shape {:}.'.format(
                key, np.shape(d))

            triu_indices = np.triu_indices(n=height, k=1)
            d = np.transpose(np.transpose(d, axes=[1, 2, 0])[
                triu_indices], axes=[1, 0])

            transoformed_data[key] = d
        elif 'label' in key:
            transoformed_data[key] = data[key]

    return transoformed_data
