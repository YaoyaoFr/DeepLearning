from Structure.DeepNerualNetwork.DeepNeuralNetwork import DeepNeuralNetwork
from utils import *
from sklearn import svm
from Structure.Layer import *
from data.utils_prepare_data import *
from Structure.Schemes.xml_parse import *
from abc import ABCMeta


class Classifier(object, metaclass=ABCMeta):
    """
    The basic object of classifiers
    """

    def __init__(self):
        pass

    @staticmethod
    def run(self):
        raise NotImplementedError


class SupportVectorMachine(Classifier):

    def __init__(self):
        Classifier.__init__(self)
        self.clf = svm.SVC(kernel='linear')

    def run(self, data):
        self.check_data_format(data)

        self.clf.fit(data['train data'], data['train label'])
        predict_train = self.clf.predict(data['train data'])
        train = np.array([predict_train, data['train label']])
        mse_train = self.clf.score(data['train data'], data['train label'])
        predict_valid = self.clf.predict(data['valid data'])
        valid = np.array([predict_valid, data['valid label']])
        mse_valid = self.clf.score(data['valid data'], data['valid label'])
        predict_test = self.clf.predict(data['test data'])
        test = np.array([predict_test, data['test label']])
        mse_test = self.clf.score(data['test data'], data['test label'])
        print('Train: {:5e}    Valid: {:5e}    Test: {:5e}'.format(mse_train, mse_valid, mse_test))

    def check_data_format(self, data):
        for tvt in ['train', 'valid', 'test']:
            # Data
            tvt_flag = '{:s} data'.format(tvt)
            shape = np.shape(data[tvt_flag])
            data[tvt_flag] = np.reshape(data[tvt_flag], newshape=[shape[0], -1])

            # Label
            tvt_flag = '{:s} label'.format(tvt)
            shape = np.shape(data[tvt_flag])
            if len(shape) == 2:
                data[tvt_flag] = onehot_to_vector(data[tvt_flag])


def ann_classify(folds: h5py.Group = None):
    if folds is None:
        hdf5_path = b'F:/OneDriveOffL/Data/Data/DCAE.hdf5'
        hdf5 = hdf5_handler(hdf5_path, 'a')
        folds = hdf5['scheme 1/falff']

    for fold_idx in folds:
        sub_folder_name = 'fold_{:d}'.format(fold_idx + 1)
        # Load Data
        fold = hdf5.require_group('experiments/falff_reho_whole/{:d}'.format(fold_idx + 1))
        data = dict()
        for tvt in ['train', 'valid', 'test']:
            for flag in ['data', 'label']:
                tvt_flag = '{:s} {:s} encoder'.format(tvt, flag)
                data_tmp = np.array(fold[tvt_flag])
                shape = np.shape(data_tmp)

                data_tmp = np.array(fold[tvt_flag])
                data[tvt_flag] = data_tmp

        # Set Graph
        graph = tf.Graph()
        ann = DeepNeuralNetwork(graph=graph, subfolder_name=sub_folder_name)
        ann.write_graph()
        ann.initialization()
        ann.backpropagation(data=data)


def svm_classify(datas=None, folds=None, data_flag='data encoder'):
    svm_classifier = SupportVectorMachine()
    if datas is None:
        datas = prepare_classify_data(folds=folds, data_flag=data_flag)
    for data in datas:
        svm_classifier.run(data)


def cnn_classify(datas=None, folds=None):
    cnn_classifier = DeepNeuralNetwork(scheme=1)
    if datas is None:
        datas = prepare_classify_data(folds=folds,
                                      data_flag='data',
                                      new_shape=[-1, 61, 61, 1],
                                      one_hot=True)
    for data in datas:
        cnn_classifier.backpropagation(data=data)
