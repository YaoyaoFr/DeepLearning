import h5py
import scipy.io as sio
import numpy as np
from Structure.classfier import Classifier, SupportVectorMachine
from Data.utils_prepare_data import hdf5_handler


def onehot_to_vector(data, class_num=2):
    data_tmp = np.zeros(shape=[np.size(data, 0)], dtype=int)
    for class_index in range(class_num):
        data_tmp[np.where(data[:, class_index] == 1)] = class_index
    return data_tmp


def run_classifier(folds=None, classifier: Classifier = None):
    """

    :param model_dir_path: The directory path of the saved model.
    :param dataset: The dataset to be prepared.
    :param folds:
    :param classifier:
    :return:
    """
    if folds is None:
        hdf5_path = b'F:/OneDriveOffL/Data/Data/ABIDE/abide.hdf5'
        hdf5 = hdf5_handler(hdf5_path, 'a')
        folds = hdf5['experiments/falff_whole']

    if classifier is None:
        classifier = SupportVectorMachine()

    datas = list()
    for fold_idx in folds:
        data = dict()
        fold = folds[fold_idx]
        for tvt in ['train', 'valid', 'test']:
            for flag in ['data', 'label']:
                tvt_flag = '{:s} {:s}'.format(tvt, flag)
                data_tmp = np.array(fold['{:s} encoder'.format(tvt_flag)])
                data[tvt_flag] = data_tmp
        classifier.run(data)


def calculate_MSE(folds: h5py.Group = None, model=None):
    if folds is None:
        hdf5_path = b'F:/OneDriveOffL/Data/Data/DCAE.hdf5'
        hdf5 = hdf5_handler(hdf5_path, 'a')
        folds = hdf5['scheme 3/falff']

    MSEs = dict()
    for fold_idx in folds:
        MSE = list()
        fold = folds[fold_idx]
        for tvt in ['pre training', 'train', 'train', 'test']:
            tvt_data = '{:s} data'.format(tvt)
            tvt_reconstruction = '{:s} data output'.format(tvt)

            data = np.array(fold[tvt_data])
            if model is None:
                try:
                    reconstruction = np.array(fold[tvt_reconstruction])
                    square_error = np.square(np.subtract(data, reconstruction))
                    for i in range(len(np.shape(square_error)) - 1):
                        square_error = np.mean(square_error, -1)
                    mses = square_error
                except:
                    return
            else:
                data, _, reconstruction, mses = model.feedforward(data, if_print=True)
            mse = np.mean(mses)
            MSEs['{:s}_{:s}'.format(fold_idx.replace(' ', '_'), tvt.replace(' ', '_'))] = mses
            print('{:5s}    {:5s}    MSE:  {:5e}'.format(fold_idx, tvt, mse))
    sio.savemat('MSE.mat', MSEs)


def get_slice(dataset, feature, fold_idx, subject_idx, slice_idx):
    hdf5_path = 'F:/OneDriveOffL/Data/Data/{:s}/{:s}.hdf5'.format(dataset.upper(), dataset.lower()).encode()
    hdf5 = hdf5_handler(hdf5_path, 'a')
    fold = hdf5['experiments/{:s}_whole/{:s}'.format(feature, fold_idx)]
    train_data = np.array(fold['train data'])
    train_data_reconstruction = np.array(fold['train data output'])

    data_slice = train_data[subject_idx * 61 + slice_idx]
    recons_slice = train_data_reconstruction[subject_idx * 61 + slice_idx]

    shape = np.shape(data_slice)
    data_slice = np.reshape(data_slice, [shape[0], shape[1]])
    recons_slice = np.reshape(recons_slice, [shape[0], shape[1]])

    return data_slice, recons_slice

