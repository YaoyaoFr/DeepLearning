import os
import h5py
import numpy as np
import tensorflow as tf

from abc import ABCMeta
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score

from Log.log import Log
from Schemes.xml_parse import parse_xml_file
from Dataset.utils import onehot2vector
from Model.utils_model import upper_triangle


class SupportVectorMachine(object, metaclass=ABCMeta):
    classifier = None
    NN_type = 'Support Vector Machine'

    def __init__(self,
                 scheme: str,
                 dir_path: str,
                 log: Log = None,
                 graph: tf.Graph = None,
                 spe_pas: dict = None,
                 ):
        self.pas = {}
        self.scheme = scheme
        self.project_path = os.path.join(
            dir_path, 'Program/Python/DeepLearning')
        self.load_parameters(scheme=scheme, spe_pas=spe_pas)

    def load_parameters(self,
                        scheme: str,
                        spe_pas: dict = None):
        """
        Load parameters from configuration file (.xml)
        :param scheme: file path of configuration file
        :return:
        """
        if not spe_pas:
            pas = parse_xml_file(os.path.join(
                self.project_path, 'Schemes/{:s}.xml'.format(scheme)))
        self.pas['basic'] = pas['parameters']['basic']

    def build_structure(self):
        self.classifier = LinearSVC(penalty='l2')
        print('Build classifier: Linear Support Vector Machine. ')

    def training(self,
                 data: h5py.Group or dict,
                 run_time: int = 1,
                 fold_name: str = None,
                 if_show: bool = True,
                 ):
        self.build_structure()

        data = self.load_data(data=data)
        results = self.fit(data=data)
        if show_info:
            self.show_results(results=results,
                              run_time=run_time,
                              fold_name=fold_name)

        return results

    def fit(self,
            data: dict,
            ):
        train_data = data['train data']
        train_label = data['train label']
        self.classifier.fit(X=train_data, y=train_label)
        results = self.predict(data=data)

        return results

    def predict(self,
                data: np.ndarray,
                ):
        results = {}
        for tag in ['train', 'valid', 'test']:
            try:
                prediction = self.classifier.predict(
                    data['{:s} data'.format(tag)])
                label = data['{:s} label'.format(tag)]

                results[tag] = self.get_metrics(
                    predict=prediction, labels=label)
            except KeyError:
                continue
        return results

    @staticmethod
    def get_metrics(predict: np.ndarray,
                    labels: np.ndarray):
        metrics = dict()
        metrics['Accuracy'] = accuracy_score(predict, labels)

        TP = np.count_nonzero(predict * labels)
        TN = np.count_nonzero((predict - 1) * (labels - 1))
        FN = np.count_nonzero(predict * (labels - 1))
        FP = np.count_nonzero((predict - 1) * labels)

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

    @staticmethod
    def show_results(results: dict,
                     run_time: int,
                     alpha: float = None,
                     fold_name: str = None):
        info = 'Time: {:d}\t'.format(run_time)
        if alpha:
            info += 'alpha: {:f}\t'.format(alpha)

        if fold_name is not None:
            info += '{:s}\t'.format(fold_name)
        for tag in ['Accuracy', 'Cross Entropy']:
            try:
                info += '{:s}: {:.5f}\t'.format(tag,
                                                results[tag]['test'])
            except KeyError:
                pass
        print(info)

    @staticmethod
    def load_data(data: h5py.Group or dict):
        if isinstance(data, h5py.Group):
            data = {tag: np.array(data[tag]) for tag in data}

        data.update({tag: onehot2vector(data[tag])
                     for tag in data if 'label' in tag})
        data = upper_triangle(data=data)

        return data
