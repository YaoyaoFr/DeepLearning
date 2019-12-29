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


class EarlyStop:
    epoch = 1
    tolerance = 0
    optimize_type = 'Cross Entropy'
    optimize_dataset = 'valid'

    max_epoch = 1
    max_epochs = []

    strategy_dict = {}

    def __init__(self,
                 log: Log,
                 data: dict,
                 results: dict,
                 pas: dict):

        self.log = log
        self.training_cycle = pas['training_cycle']
        for pa in pas:
            setattr(self, pa, pas[pa])

        self.results = {result: {tag: [] for tag in ['train', 'valid', 'test'] if '{:s} data'.format(tag) in data}
                        for result in results}

        self.strategy_dict = {
            'basic': self.basic,
            'early_stop': self.early_stop,
            'restore': self.restore,
        }

        if self.strategy == 'basic':
            need_attrs = [
                'learning_rate',
            ]
        else:
            need_attrs = [
                'save_cycle',
                'tolerance_all',
                'learning_rate',
                'decay_rate',
            ]
        for attr in need_attrs:
            assert hasattr(
                self, attr), 'The attribution {:s} is necessary.'.format(attr)

        if self.strategy == 'restore':
            need_attrs = ['back_epoch', 'min_learning_rate']
        elif self.strategy == 'early_stop':
            need_attrs = ['decay_step']
        for attr in need_attrs:
            assert hasattr(self, attr), 'if strategy is {:s}, the attribution {:s} is necessary.'.format(self.strategy,
                                                                                                         attr)

    def next(self,
             results: dict,
             ):
        self.show_results_epoch(results=results)

        if self.epoch % self.save_cycle == 0:
            self.log.save_model(epoch=self.epoch, show_info=False)

        # Update results
        for tag in results:
            for type in results[tag]:
                result = results[tag][type]
                self.results[type][tag].append(result)

        if self.epoch == 1:
            self.max_epoch = self.epoch
            self.max_epochs.append(self.max_epoch)
            self.log.save_model(epoch=self.epoch, show_info=True)
            self.epoch += 1
            return

        self.strategy_dict[self.strategy](results=results)

    def basic(self,
              results: dict):
        self.max_epoch = self.epoch
        self.epoch += 1
        return

    def early_stop(self,
                   results: dict):
        # Skip early stop while the training accuracy is too low
        if hasattr(self, 'stop_accuracy'):
            if results['train']['Accuracy'] < self.stop_accuracy:
                self.epoch += 1
                return

        if self.strategy == 'early_stop' and self.epoch + 1 % self.decay_step == 0:
            self.learning_rate *= self.decay_rate
            print('Change learning rate to {:.5f}.'.format(self.learning_rate))

        if results[self.optimize_dataset][self.optimize_type] < np.min(
                np.array(self.results[self.optimize_type][self.optimize_dataset])[:-1]):
            self.max_epoch = self.epoch
            self.tolerance = 0
        else:
            self.tolerance += 1

        # When tolerance greater less than tolerance all
        if self.tolerance >= self.tolerance_all:
            self.epoch = self.training_cycle

        self.epoch += 1

    def restore(self,
                results: dict):
        # Skip early stop while the training accuracy is too low
        if hasattr(self, 'stop_accuracy'):
            if results['train']['Accuracy'] < self.stop_accuracy:
                self.epoch += 1
                return

        if results[self.optimize_dataset][self.optimize_type] < np.min(
                np.array(self.results[self.optimize_type][self.optimize_dataset])[:-1]):
            self.max_epoch = self.epoch
            self.tolerance = 0
            self.log.save_model(epoch=self.epoch, show_info=True)
            self.max_epochs.append(self.epoch)
        else:
            self.tolerance += 1

        # When tolerance greater less than tolerance all
        if self.tolerance >= self.tolerance_all:
            self.learning_rate *= self.decay_rate
            print('Change learning rate to {:.8f}.'.format(self.learning_rate))

            # Stop restore
            if self.learning_rate < self.min_learning_rate:
                self.epoch = self.training_cycle
            else:
                back_epoch = - \
                    self.back_epoch if len(
                        self.max_epochs) >= self.back_epoch else 0
                self.max_epoch = self.max_epochs[back_epoch]
                if back_epoch + 1 < 0:
                    self.max_epochs = self.max_epochs[:back_epoch + 1]
                self.epoch = self.max_epoch

                self.log.restore(restored_epoch=self.max_epoch)
                self.tolerance = 0
                self.results = {result: {tag: self.results[result][tag][:self.max_epoch]
                                         for tag in self.results[result]} for result in self.results}

        self.epoch += 1

    def show_results_epoch(self,
                           results: dict,
                           ):
        info = 'Epoch: {:d}\t'.format(self.epoch)

        result_tags = [tag for tag in ['train', 'valid', 'test'] if tag in results]
        for tag in result_tags:
            try:
                info += '{:s} CE: {:.5f}\taccuracy: {:.4f}\t'.format(tag,
                                                                     results[tag]['Cross Entropy'],
                                                                     results[tag]['Accuracy'])
            except KeyError:
                info += '{:s} MSE: {:.5f}\t'.format(tag,
                                                    results[tag]['MSE'])

        print(info)

    def show_results(self,
                     run_time: int,
                     fold_index: int = None):
        info = 'Time: {:d}\t'.format(run_time)
        if fold_index:
            info += 'Fold: {:d}\t'.format(fold_index)
        for tag in ['Accuracy', 'Cross Entropy']:
            try:
                info += '{:s}: {:.5f}\t'.format(tag,
                                                self.results[tag]['test'][self.max_epoch - 1])
            except KeyError:
                pass
        print(info)

    def clear_models(self,
                     save_optimal_model: bool = True):
        if save_optimal_model:
            optimal_dir = os.path.join(self.log.dir_path, 'optimal_model')
            if not os.path.exists(optimal_dir):
                os.mkdir(optimal_dir)
            self.log.restore(restored_epoch=self.max_epoch)
            optimal_path = os.path.join(
                optimal_dir, 'train_model_{:d}'.format(self.max_epoch))
            self.log.save_model(save_path=optimal_path)

        rm_dir = os.path.join(self.log.dir_path, 'model')
        shutil.rmtree(rm_dir)


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
        FN = tf.count_nonzero(predict * (labels - 1))
        FP = tf.count_nonzero((predict - 1) * labels)

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
    for tag in ['train', 'valid', 'test']:
        key = '{:s} data'.format(tag)
        if key not in data:
            continue

        d = np.squeeze(data[key], axis=-1)
        assert np.linalg.matrix_rank(d) == 3

        [_, width, height] = np.shape(d)
        assert height == width, '{:s} must be square matrix but get shape {:}.'.format(key, np.shape(d))

        triu_indices = np.triu_indices(n=height, k=1)
        d = np.transpose(np.transpose(d, axes=[1, 2, 0])[
                         triu_indices], axes=[1, 0])

        data[key] = d

    return data
