import contextlib
import multiprocessing
import string
import sys
import time

import h5py
import numpy as np
import numpy.ma as ma
from nipy import load_image
from sklearn.preprocessing import scale


def load_fold(subjs, experiment, fold):
    '''
    Load data in each fold given the corresponding subjects ids
    :param subjs: All subjects
    :param experiment: The experiments settings
    :param fold: The cross validation fold
    :return: The data
    '''
    feature = experiment.attrs['feature']
    features = [d.decode('utf-8') for d in feature]

    data = dict()
    for tvt in ['train', 'valid', 'test']:
        if tvt in fold:
            data_list = list()
            for pid in fold[tvt]:
                data_list.append(load_patient_data(patient=subjs[pid], features=features))
            data_temp = np.array(data_list)
            # data_temp = data_normalization(data=data_temp, axis=0)
            # data_temp = split_slices(data_temp)
            data['{:s} data'.format(tvt)] = data_temp

            label_list = list()
            for pid in fold[tvt]:
                label_list.append(subjs[pid].attrs["y"])
            label_tmp = np.array(label_list)
            # label_tmp = vecter2onehot(label_tmp, 2)
            # label_tmp = repeatmap(label_tmp, num_repeat=61)
            data['{:s} label'.format(tvt)] = label_tmp
    return data


def load_patient_data(patient, features):
    data = []
    for feature in features:
        data.append(np.expand_dims(np.array(patient[feature]), axis=-1))
    data = np.concatenate(data, axis=-1)
    return data


def load_nifti_data(data_path: str, mask: bool = False, normalization: bool = False):
    img = load_image(data_path)
    data = np.array(img.get_data())

    atlas = None
    if mask:
        shape = np.shape(data)
        atlas_path = 'aal_{:s}.nii'.format('_'.join([str(i) for i in shape]))
        atlas = np.array(load_image(atlas_path).get_data()) == 0
        data[atlas] = 0

    if normalization:
        mean = np.mean(data)
        var = np.var(data)
        data = (data - mean) / var
        if atlas is not None:
            data[atlas] = 0
    return np.array(data)


def hdf5_handler(filename, mode="r"):
    h5py.File(filename, "a").close()
    propfaid = h5py.h5p.create(h5py.h5p.FILE_ACCESS)
    settings = list(propfaid.get_cache())
    settings[1] = 0
    settings[2] = 0
    propfaid.set_cache(*settings)
    with contextlib.closing(h5py.h5f.open(filename, fapl=propfaid)) as fid:
        return h5py.File(fid, mode)


def create_dataset_hdf5(group: h5py.Group,
                        data: np.ndarray,
                        name: str,
                        is_save: bool = True,
                        cover: bool = True,
                        dtype: np.dtype = float) -> None:
    if name in group:
        if cover:
            group.pop(name)
        else:
            print('\'{:s}\' already exist in \'{:s}\'.'.format(name, group.name))
            return

    if is_save:
        group.create_dataset(name=name,
                             data=data,
                             dtype=dtype)
        print('Create \'{:s}\' in \'{:s}\''.format(name, group.name))


def prepare_classify_data(folds: h5py.Group = None,
                          new_shape: list = None,
                          one_hot: bool = False,
                          normalization: bool = False,
                          data_flag: str = 'data encoder',
                          slice_index: int or list = None
                          ):
    if folds is None:
        hdf5_path = b'F:/OneDriveOffL/Data/Data/DCAE.hdf5'
        hdf5 = hdf5_handler(hdf5_path, 'a')
        folds = hdf5.require_group('scheme 1/falff')

    data_list = list()
    for fold_idx in folds:
        # Load Data
        fold = folds[fold_idx]
        try:
            data = dict()
            for tvt in ['train', 'valid', 'test']:
                tvt_data = '{:s} {:s}'.format(tvt, data_flag)
                data_tmp = np.array(fold[tvt_data])

                if slice_index is not None:
                    data_tmp = data_tmp[:, slice_index, :]
                if new_shape is not None:
                    data_tmp = np.reshape(a=data_tmp, newshape=new_shape)
                if normalization:
                    data_tmp = data_normalization(data=data_tmp, axis=1, normalization=True, sigmoid=False)

                tvt_label = '{:s} label'.format(tvt)
                label_tmp = np.array(fold[tvt_label])

                if one_hot:
                    label_tmp = vecter2onehot(label_tmp, 2)

                data['{:s} data'.format(tvt)] = data_tmp
                data[tvt_label] = label_tmp
            data_list.append(data)
            print('{:s} prepared.'.format(fold.name))
        except:
            print('{:s} prepared failed.'.format(fold.name))

    return data_list


def compute_connectivity(functional):
    with np.errstate(invalid="ignore"):
        corr = np.nan_to_num(np.corrcoef(functional))
        return corr
        # mask = np.invert(np.tri(corr.shape[0], k=-1, dtype=bool))
        # m = ma.masked_where(mask == 1, mask)
        # return ma.masked_where(m, corr).compressed()


class SafeFormat(dict):

    def __missing__(self, key):
        return "{" + key + "}"

    def __getitem__(self, key):
        if key not in self:
            return self.__missing__(key)
        return dict.__getitem__(self, key)


def merge_dicts(*dict_args):
    result = {}
    for dictionary in dict_args:
        result.update(dictionary)
    return result


def format_config(s, *d):
    dd = merge_dicts(*d)
    return string.Formatter().vformat(s, [], SafeFormat(dd))


def run_progress(callable_func, items, message=None, jobs=10):
    results = []

    print('Starting pool of %d jobs' % jobs)

    current = 0
    total = len(items)

    if jobs == 1:
        results = []
        for item in items:
            results.append(callable_func(item))
            current = len(results)
            if message is not None:
                args = {'current': current, 'total': total}
                sys.stdout.write("\r" + message.format(**args))
                sys.stdout.flush()

    # Or allocate a pool for multi-threading
    else:
        pool = multiprocessing.Pool(processes=jobs)
        for item in items:
            pool.apply_async(callable_func, args=(item,), callback=results.append)

        while current < total:
            current = len(results)
            if message is not None:
                args = {'current': current, 'total': total}
                sys.stdout.write("\r" + message.format(**args))
                sys.stdout.flush()
            time.sleep(0.5)

        pool.close()
        pool.join()

    print
    return results


def split_slices(data, axis=[0, 3, 1, 2, 4]):
    """
    Split data to slices
    :param data: with the shape of [batch_num, width, height, depth, channels]
    :param axis: one of the (0, 1, 2) corresponding (width, height, depth)
    :return:
    """

    data = data.transpose(axis)
    shape = np.shape(data)
    data = np.reshape(a=data,
                      newshape=[shape[0] * shape[1], shape[2], shape[3], shape[4]],
                      )
    return data


def repeatmap(data, num_repeat):
    shape = np.shape(data)
    new_data = list()
    for data_slice in data:
        for i in range(num_repeat):
            new_data.append(data_slice)
    new_data = np.array(new_data)
    new_data = np.reshape(new_data, [num_repeat * shape[0], -1])
    return new_data


def data_normalization(data, axis=-1, normalization=False, sigmoid=True):
    # Format list to ndarray
    if isinstance(data, list):
        data = np.array(data)

    if normalization:
        # Format data to two-dimension
        shape = np.shape(data)
        new_data = np.reshape(a=data,
                              newshape=[np.prod(shape[0:-1]), shape[-1]],
                              )
        new_data = scale(X=new_data, axis=axis)
        new_data = np.reshape(a=new_data,
                              newshape=shape,
                              )
    if sigmoid:
        new_data = 1.0 / (1 + np.exp(-data)) - 0.5
    return new_data


def vecter2onehot(data, class_num=2):
    data_tmp = np.zeros([np.size(data, 0), class_num])
    for class_index in range(class_num):
        data_tmp[np.where(data == class_index)[0], class_index] = 1
    return data_tmp


def get_folds(dataset: str, feature: str, fold_indexes: list = None) -> h5py.Group:
    hdf5_path = 'F:/OneDriveOffL/Data/Data/{:s}/{:s}.hdf5'.format(dataset.upper(), dataset.lower()).encode()
    hdf5 = hdf5_handler(hdf5_path, 'a')
    folds_basic = hdf5['experiments/{:s}_whole'.format(feature)]

    if fold_indexes is None:
        fold_indexes = folds_basic.keys()

    folds = dict()
    for fold_idx in fold_indexes:
        folds[fold_idx] = folds_basic[fold_idx]

    return folds


def get_datas(dataset, feature, fold_indexes, indexes, tvt='train'):
    folds = get_folds(dataset=dataset,
                      feature=feature,
                      fold_indexes=fold_indexes)
    datas = dict()
    for fold_idx in folds:
        fold = folds[fold_idx]
        data_basic = np.array(fold['{:s} data'.format(tvt)])
        recons_basic = np.array(fold['{:s} data output'.format(tvt)])

        data = dict()
        data['data'] = data_basic[indexes, :, :, 0]
        data['output'] = recons_basic[indexes, :, :, 0]
        datas[fold_idx] = data

    return datas
