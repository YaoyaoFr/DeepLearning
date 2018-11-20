import contextlib
import multiprocessing
import string
import sys
import time

import os
import h5py
import numpy as np
import numpy.ma as ma
import pandas as pd
from scipy import sparse
from scipy import stats
from nipy import load_image
from sklearn.preprocessing import scale
from Data.load_data import load_phenotypes


def load_fold(dataset_group: h5py.Group,
              fold_group: h5py.Group,
              experiment: h5py.Group = None,
              features: list = None,
              dataset: str = None, ) -> dict:
    """
    load data in each fold given the corresponding data_group ids
    :param dataset_group: list of all data_group
    :param fold_group: The fold of cross validation
    :param experiment: The experiments settings
    :param features: the list of features to be loaded
    :param dataset: the dataset to be loaded
    :return: dictionary {'train_data' ,'valid_data', 'test_data' if exist}
    """
    if features is None:
        features = [feature.decode() for feature in experiment.attrs['features']]
    if dataset is None:
        dataset = experiment.attrs['dataset']

    datas = {}
    for flag in ['train', 'valid', 'test']:
        if flag not in fold_group:
            continue

        print('Loading  {:5s} data of {:5} in dataset: {:7s} ...'.format(flag, '_'.join(features), dataset))
        data = np.array([load_subject_data(subject_group=dataset_group[subject],
                                           features=features) for subject in fold_group[flag]])
        label = np.array([dataset_group[subject_id].attrs['y'] for subject_id in fold_group[flag]])
        datas['{:s} data'.format(flag)] = data
        datas['{:s} label'.format(flag)] = label
    return datas


def load_subject_data(subject_group: h5py.Group, features: list) -> np.ndarray:
    """
    Load data from h5py file in terms of subject list
    :param subject_group: group of subject to be loaded
    :param features: list of features to be loaded
    :return: An np.ndarray with shape of [data_shape, feature_num]
    """
    datas = []
    for feature in features:
        data = np.array(subject_group[feature])
        data = np.expand_dims(data, axis=-1)
        datas.append(data)
    datas = np.squeeze(np.concatenate(datas, axis=-1))
    return datas


def load_nifti_data(data_path: str, mask: bool = False, normalization: bool = False):
    img = load_image(data_path)
    data = np.array(img.get_data())

    atlas = None
    if mask:
        shape = np.shape(data)
        atlas_path = 'aal_{:s}.nii'.format('_'.join([str(i) for i in shape]))
        if not os.path.exists(atlas_path):
            atlas_path = 'Data/aal_{:s}.nii'.format('_'.join([str(i) for i in shape]))
        atlas = np.array(load_image(atlas_path).get_data()) == 0
        data[atlas] = 0

    if normalization:
        mean = np.mean(data)
        var = np.var(data)
        data = (data - mean) / var
        if atlas is not None:
            data[atlas] = 0

    data = np.array(data)
    return data


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
                    data_tmp, mean, std = data_normalization(data=data_tmp, axis=1, standardization=True, sigmoid=False)

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


def data_normalization(data: np.ndarray or list,
                       axis: int = 0,
                       mean: np.ndarray = None,
                       std: np.ndarray = None,
                       standardization: bool = True,
                       normalization: bool = False,
                       sigmoid: bool = False
                       ):
    # Format list to ndarray
    if isinstance(data, list):
        data = np.array(data)

    if standardization:
        if mean is None:
            mean = np.mean(data, axis=axis)
        if std is None:
            std = np.std(data, axis=axis)
            std[std == 0] = 1
        data = (data - mean) / std

    if normalization:
        max = np.max(data, axis=axis)
        min = np.min(data, axis=axis)
        scale = max - min
        scale[scale == 0] = 1
        data = (data - min) / scale

    if sigmoid:
        data = 1.0 / (1 + np.exp(-data)) - 0.5
    return data, mean, std


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


def get_subjects(pheno: pd.DataFrame, group: str) -> pd.DataFrame:
    dx_group = {'health': 1, 'patient': 0, 'all': 2}[group]
    pheno = pheno[pheno['DX_GROUP'] != dx_group]
    ids = pheno['FILE_ID']
    ids_encode = list()
    for id in ids:
        ids_encode.append(id.encode())
    ids_encode = pd.Series(ids_encode)
    return ids_encode


def extract_regions(data: np.ndarray,
                    regions: int or list = None,
                    atlas: h5py.Group = None,
                    mask: bool = False) -> list:
    if atlas is None:
        hdf5_path = b'F:/OneDriveOffL/Data/Data/DCAE_aal.hdf5'
        atlas = hdf5_handler(hdf5_path)['MNI']

    if regions is None:
        regions = range(90)
    elif isinstance(regions, int):
        regions = range(regions)

    shape = np.shape(data)
    batch_size = shape[0]
    if len(shape) > 4:
        channel_num = shape[-1]
    else:
        data = np.expand_dims(data, axis=-1)
        channel_num = 1

    data_list = []
    for region_index in regions:
        data_region = []
        for channel in range(channel_num):
            data_channel = data[:, :, :, :, channel]
            region_index_str = str(region_index + 1)
            region_group = atlas[region_index_str]
            bounds = np.array(region_group['bounds'], dtype=int)

            data_channel = data_channel[
                           :,
                           bounds[0][0]:bounds[0][1],
                           bounds[1][0]:bounds[1][1],
                           bounds[2][0]:bounds[2][1],
                           ]

            if mask:
                mask = np.concatenate(
                    [np.expand_dims(np.array(region_group['mask'], dtype=int), 0)
                     for _ in range(batch_size)], axis=0)
                data_channel[mask] = 0

            data_region.append(np.expand_dims(data_channel, axis=-1))
        data_region = np.concatenate(data_region, axis=-1)
        data_list.append(data_region)
    return data_list


def function_connectivity_ttest(normal_controls: np.ndarray,
                                patients: np.ndarray,
                                significance: float = 0.05,
                                triangle='lower') -> dict:
    """
    Independent two-sample t-test of normal control and disease for each element of functional connectivity.
    :param normal_controls: The functional connectivity of normal controls
    :param patients: The functional connectivity of patients
    :param significance: The threshold of statistic significance, default=0.05
    :param triangle: The triangle of the output matrices, default='lower' indicate return lower triangle matrices
    :return: A dictionary contains:
                hypothesis: np.ndarray(bool), indicate whether accept the null hypothesis \mu_1=\mu_2
                t_value: np.ndarray(float), the statistic value
                p_value: np.ndarray(float), the calculated p-value
    """
    shape1 = np.shape(normal_controls)
    shape2 = np.shape(patients)
    if len(shape1) > 3:
        raise TypeError('The dimension of functional connectivity must be 3 but is {:d}.'.format(len(shape1)))
    if len(shape2) > 3:
        raise TypeError('The dimension of functional connectivity must be 3 but is {:d}.'.format(len(shape2)))

    if shape1[1] != shape2[1] or shape1[2] != shape2[2]:
        raise TypeError(
            'The dimension of two functional connectivity '
            'doesn\'t match which are {:} and {:}'.format(shape1[1:3], shape2[1:3]))

    shape = shape1[1:3]

    t_value = np.zeros(shape=shape, dtype=float)
    p_value = np.ones(shape=shape, dtype=float)

    for row in range(shape[0]):
        if triangle == 'lower':
            start = 0
            stop = row
        elif triangle == 'upper':
            start = row + 1
            stop = shape[1]
        for column in np.arange(start=start, stop=stop):
            nc = normal_controls[:, row, column]
            patient = patients[:, row, column]
            result = stats.levene(nc, patient)
            equal_var = result.pvalue > significance
            result = stats.ttest_ind(nc, patient, equal_var=equal_var)
            t_value[row, column] = result.statistic
            p_value[row, column] = result.pvalue

    hypothesis = p_value > significance

    return {
        'hypothesis': hypothesis,
        't_value': t_value,
        'p_value': p_value,
    }


def select_top_significance_ROIs(datasets: list = ['ABIDE'],
                                 hdf5: h5py.Group = None,
                                 top_k_ROI: int = 5,
                                 top_k_relevant: int = 1,
                                 ):
    """
    Select the top-k significance difference brain regions by independent two-sample t-test for functional connectivity.
    :param datasets: A list of dataset to process
    :param hdf5: The hdf5 file include all datasets
    :param top_k_ROI: The number of ROIs to be selected by t-value.
    :param top_k_relevant: The number to be selected which relevant to the ROIs selected by t-value
    :return: A dict contains 
    """
    if hdf5 is None:
        hdf5_path = b'F:/OneDriveOffL/Data/Data/DCAE_data.hdf5'
        hdf5 = hdf5_handler(hdf5_path)

    result = {}
    for dataset in datasets:
        subjects_group = hdf5['{:s}/subjects'.format(dataset)]
        pheno = load_phenotypes(dataset=dataset)
        groups = {'normal_controls': 0,
                  'patients': 1}

        FCs = {}
        for group in groups:
            subjects = list(pheno[pheno['DX_GROUP'] == groups[group]]['FILE_ID'])
            FC = np.array([load_subject_data(subject_group=subjects_group[subject], features=['FC'])
                           for subject in subjects])
            FCs[group] = FC

        result = function_connectivity_ttest(FCs['normal_controls'], FCs['patients'])
        ROI_num = np.shape(result['t_value'])[0]
        t_value = np.abs(result['t_value'])
        t_value[result['hypothesis']] = 0
        t_value_sparse = dict(sparse.dok_matrix(t_value))
        sparse_items = [item for item in t_value_sparse.items()]

        ROIs_result = [[ROI_index, 0, []] for ROI_index in range(ROI_num)]

        for t_value_tuple in sparse_items:
            coordinate, t_value = t_value_tuple
            ROIs_result[coordinate[0]][1] += t_value
            ROIs_result[coordinate[1]][1] += t_value

            ROIs_result[coordinate[0]][2].append(t_value_tuple)
            ROIs_result[coordinate[1]][2].append(((coordinate[1], coordinate[0]), t_value))

        for ROI in ROIs_result:
            ROI[2] = sorted(ROI[2], key=lambda x: x[1], reverse=True)

        # Sort and select
        ROIs_result = sorted(ROIs_result, key=lambda x: x[1], reverse=True)
        ROIs = []
        for index in range(top_k_ROI):
            ROI = ROIs_result[index]
            ROIs.append(ROI[0])
            for index_relevant in range(top_k_relevant):
                relevant_ROI = ROI[2][0][index_relevant][1]
                if relevant_ROI not in ROIs:
                    ROIs.append(relevant_ROI)

        result[dataset] = ROIs

    return result
