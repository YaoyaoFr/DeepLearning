import os
import sys
import time
import logging
import scipy.io as sio
import tensorflow as tf
from Structure.Schemes.xml_parse import parse_log_parameters


class Log:
    pa = parse_log_parameters('Structure/Schemes/Log.xml')
    basic_path = pa['basic_path']
    restored_date = pa['restored_date']
    restored_time = pa['restored_time']
    sub_folder_name = pa['sub_folder_name']

    graph = None
    sess = None
    file_path = None
    train_writer = None

    def __init__(self,
                 graph=None,
                 sess=None,
                 restored_date=None,
                 restore_time=None,
                 sub_folder_name=None):
        self.set_file_path(restored_date=restored_date,
                           restore_time=restore_time,
                           sub_folder_name=sub_folder_name,
                           )
        self.set_graph(graph=graph, sess=sess)

    def set_graph(self, graph=None, sess=None):
        if graph:
            self.graph = graph
        else:
            self.graph = tf.Graph()

        if not sess:
            with tf.Session(graph=self.graph) as sess:
                self.sess = sess

    def set_file_path(self, restored_date=None, restore_time=None, sub_folder_name=None):
        if restored_date:
            self.restored_date = '{:s}'.format(restored_date)
        if restore_time:
            self.restored_time = '{:s}'.format(restore_time)
        if sub_folder_name:
            self.sub_folder_name = '{:s}'.format(sub_folder_name)

        if self.restored_date is None:
            self.restored_date = time.strftime('%Y-%m-%d', time.localtime(time.time()))

        if self.restored_time is None:
            self.restored_time = time.strftime('%H-%M', time.localtime(time.time()))

        self.file_path = ''
        if self.restored_date is None or self.restored_time is None:
            raise TypeError('File path cannot be set!')

        self.set_filepath_by_subfolder()

    def set_filepath_by_subfolder(self, subfolder_name=None):
        if subfolder_name:
            self.sub_folder_name = subfolder_name
        self.file_path = '/'.join([self.basic_path, self.restored_date, self.restored_time])
        self.file_path += '/{:s}'.format(self.sub_folder_name) if self.sub_folder_name else ''

    def get_save_dir(self):
        save_dir = '/'.join([self.basic_path, self.restored_date, self.restored_time])
        return save_dir

    def get_restored_pa(self):
        pas = {}
        if self.sub_folder_name is None:
            return None
        pas_list = self.sub_folder_name.split('/')
        pas['fold'] = int(pas_list[0].split(' ')[1])
        process_list = ['pre_train_SCAE', 'fine_tune_SCAE', 'pre_train_Classifier', 'fine_tune_Classifier']
        pas['process'] = process_list.index(pas_list[1])
        if pas['process'] == 'pre_train_SCAE' and len(pas_list) == 3:
            pas['indexes'] = [int(i) for i in pas_list[2].split('-')]
        else:
            pas['indexes'] = None
        return pas

    def write_graph(self):
        if not os.path.exists(self.file_path):
            os.makedirs(self.file_path)
        print('Write Graph to File {:s}'.format(self.file_path + '\log'))
        self.train_writer = tf.summary.FileWriter(self.file_path + '\log', graph=self.graph)

    def write_log(self, res: dict,
                  epoch: int,
                  log_type: str = 'Train',
                  if_save: bool = True,
                  show_info: bool = True,
                  pre_fix: str = None,
                  new_line: bool = False,
                  ):
        error_str = '{:5s}:  {:2d}  '.format(log_type, epoch) + (pre_fix if pre_fix else '')

        value = list()
        for res_key in sorted(res):
            tag = '{:5s} {:s}'.format(log_type, res_key)
            try:
                value.append(tf.Summary.Value(tag=tag, simple_value=res[res_key]))
            except:
                continue
            train_summaries = tf.Summary(value=value)

            error_str += '{:5s}: {:.5e}  '.format(res_key, res[res_key])

        if show_info:
            if new_line:
                error_str = '\r\n' + error_str + '\r\n'
            else:
                error_str = '\r' + error_str
            sys.stdout.write(error_str)

        if epoch > 0 and if_save:
            self.train_writer.add_summary(train_summaries, epoch)

    def close(self):
        self.train_writer.close()

    def save_model(self, epoch, show_info: bool = True):
        save_path = os.path.join(self.file_path,
                                 'model/train.model_{:d}'.format(epoch))
        self.saver.save(self.sess, save_path)
        if show_info:
            print('Model saved in file: {:s}'.format(save_path))
        return save_path

    def restore(self,
                restored_epoch: int = None,
                restored_path: str = None,
                initialize: bool = True) -> int:
        """
        Restored neural network model with restore parameters.
        :param restored_epoch: Restored model by epoch
        :param restored_path: Restored model directly by save path.
        :param initialize: Initialize the restore epoch after restored.
        :return: The training epoch of restored model.
        """
        if restored_path is None:
            restored_epoch = self.pa['restored_epoch']
            restored_path = os.path.join(self.file_path,
                                         'model/train.model_{:d}'.format(restored_epoch)) if restored_epoch else None

        try:
            self.saver.restore(self.sess, restored_path)
            print('Model restored from file: {:s}'.format(restored_path))
        except Exception as e:
            print(e)
            return 0

        if initialize:
            self.pa['restored_epoch'] = None

        return restored_epoch if restored_epoch else 0

    def save_features(self, debug_train, debug_test, train_label, test_label, epoch=None, save_path=None):
        if not epoch:
            epoch = 0
        if not save_path:
            save_path = self.file_path

        data = dict()
        data['train data'] = debug_train[5]['output']
        data['test data'] = debug_test[5]['output']
        data['train label'] = train_label
        data['test label'] = test_label

        sio.savemat('{:s}/test data {:d}.mat'.format(save_path, epoch), data)
