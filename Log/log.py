import os
import sys
import time
import scipy.io as sio
import tensorflow as tf


class Log:
    date = None
    clock = None

    scheme_folder = None
    subfolder = None

    pa = {'restored_epoch': 0}

    graph = None
    sess = None
    dir_path = None
    saver = None
    train_writer = None

    def __init__(self,
                 dir_path: str,
                 scheme_folder: str = None):
        # Set dir path: [Scheme name]/[date]/[clock]/[run time]
        self.basic_path = os.path.join(dir_path, 'Result/DeepLearning')
        self.set_folders(scheme_folder=scheme_folder)
        self.set_graph()

    def set_graph(self):
        """Get a new graph and session
        """
        self.graph = tf.Graph()

        tf_config = tf.compat.v1.ConfigProto()
        tf_config.gpu_options.allow_growth = True
        self.sess = tf.Session(graph=self.graph, config=tf_config)

    def set_folders(self, scheme_folder: str):
        self.scheme_folder = scheme_folder
        self.date = time.strftime('%Y-%m-%d', time.localtime(time.time()))
        self.clock = time.strftime('%H-%M-%S', time.localtime(time.time()))

    def set_path(self,
                 scheme_folder: str = None,
                 date: str = None,
                 clock: str = None,
                 subfolder: str = None,
                 ):
        if scheme_folder is not None:
            self.scheme_folder = scheme_folder

        if date is not None:
            self.date = date

        if clock is not None:
            self.clock = clock

        path = os.path.join(
            self.basic_path, self.scheme_folder, self.date, self.clock)
        if subfolder is not None:
            self.subfolder = subfolder
            path = os.path.join(path, self.subfolder)

        self.dir_path = path

        for subfolder in ['log', 'model', 'optimal_model']:
            subfolder_path = os.path.join(self.dir_path, subfolder)
            if not os.path.exists(subfolder_path):
                os.makedirs(subfolder_path, exist_ok=True)

    def reset_graph(self):
        tf.reset_default_graph()
        self.set_graph()

    def get_restored_pa(self):
        pas = {}
        if self.subfolder is None:
            return None
        pas_list = self.subfolder.split('/')
        pas['fold'] = int(pas_list[0].split(' ')[1])
        process_list = ['pre_train_SCAE', 'fine_tune_SCAE',
                        'pre_train_Classifier', 'fine_tune_Classifier']
        pas['process'] = process_list.index(pas_list[1])
        if pas['process'] == 'pre_train_SCAE' and len(pas_list) == 3:
            pas['indexes'] = [int(i) for i in pas_list[2].split('-')]
        else:
            pas['indexes'] = None
        return pas

    def write_graph(self):
        self.train_writer = tf.summary.FileWriter(
            self.dir_path + '/log', graph=self.graph)

    def write_log(self, res: dict,
                  epoch: int,
                  log_type: str = 'Train',
                  if_save: bool = True,
                  show_info: bool = True,
                  pre_fix: str = None,
                  new_line: bool = False,
                  ):
        error_str = '{:5s}:  {:2d}  '.format(
            log_type, epoch) + (pre_fix if pre_fix else '')

        value = list()
        for res_key in sorted(res):
            tag = '{:5s} {:s}'.format(log_type, res_key)
            try:
                value.append(tf.Summary.Value(
                    tag=tag, simple_value=res[res_key]))
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

    def save_model(self,
                   epoch: int = None,
                   show_info: bool = True,
                   save_path: str = None):
        assert epoch or save_path, 'At least one of save epoch and path is not None.'

        if save_path is None:
            save_path = os.path.join(
                self.dir_path, 'model', 'train.model_{:d}'.format(epoch))
        self.saver.save(self.sess, save_path)
        if show_info:
            print('Model saved in file: {:s}'.format(save_path))
        return save_path

    def restore(self,
                restored_epoch: int = None,
                restored_path: str = None,
                restored_dir: str = 'model',
                initialize: bool = True) -> int:
        """
        Restored neural network model with restore parameters.
        :param restored_epoch: Restored model by epoch
        :param restored_path: Restored model directly by save path.
        :param initialize: Initialize the restore epoch after restored.
        :return: The training epoch of restored model.
        """
        if restored_path is None:
            if restored_epoch is None:
                restored_epoch = self.pa['restored_epoch']
            restored_path = os.path.join(self.dir_path,
                                         '{:s}/train_model_{:}'.format(restored_dir, restored_epoch)) if restored_epoch else None

        try:
            self.saver.restore(self.sess, restored_path)
            restored_epoch = int(restored_path.split('_')[-1])
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
            save_path = self.dir_path

        data = dict()
        data['train data'] = debug_train[5]['output']
        data['test data'] = debug_test[5]['output']
        data['train label'] = train_label
        data['test label'] = test_label

        sio.savemat('{:s}/test data {:d}.mat'.format(save_path, epoch), data)
