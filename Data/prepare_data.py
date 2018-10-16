import h5py
import random
import numpy as np
import pandas as pd
from docopt import docopt
from Data.utils_prepare_data import *
from Data.load_data import load_subjects_to_file, load_phenotypes
from sklearn.model_selection import StratifiedKFold, train_test_split


def load_datas(cover: bool = False) -> h5py.File:
    """
    prepare the subjects' data
    :param cover: whether rewrite the data once it has been existed in the h5py file
    :return: the handler of h5py file
    """
    hdf5_path = b'F:/OneDriveOffL/Data/Data/DCAE_data.hdf5'
    hdf5 = hdf5_handler(hdf5_path, 'a')

    features = ['falff', 'reho', 'vmhc']
    datasets = ['ABIDE II', 'ABIDE', 'ADHD', 'FCP']

    if cover:
        load_subjects_to_file(hdf5, features, datasets)
    return hdf5


def prepare_folds(parameters: dict, folds_hdf5: h5py.File = None) -> None:
    """
    generate a list of subjects for train, valid, test in each fold
    :param parameters: parameters for generating
    :param folds_hdf5: the handler of h5py file for storing the lists
    :return: None
    """
    if folds_hdf5 is None:
        folds_hdf5_path = b'F:/OneDriveOffL/Data/Data/DCAE_folds.hdf5'
        folds_hdf5 = hdf5_handler(folds_hdf5_path, 'a')

    print('Preparing folds...')
    datasets = parameters['datasets']
    fold_nums = parameters['fold_nums']
    groups = parameters['groups']
    for group, dataset, fold_num in zip(groups, datasets, fold_nums):
        dataset_group = folds_hdf5.require_group(dataset)
        pheno = load_phenotypes(dataset=dataset)

        for gro, num in zip(group, fold_num):
            subject_list = get_subjects(pheno=pheno, group=gro)

            if num == 0:
                fold = dataset_group.require_group(str(num))
                if 'train' in fold:
                    fold.pop('train')
                fold['train'] = subject_list.tolist()
                sizes = {'train': len(subject_list)}
            else:
                skf = StratifiedKFold(n_splits=num, shuffle=True)
                for i, (train_index, test_index) in enumerate(skf.split(subject_list, pheno['STRAT'])):
                    train_index, valid_index = train_test_split(train_index, test_size=0.2)
                    sizes = {'train': len(train_index),
                             'valid': len(valid_index),
                             'test': len(test_index)}

                    fold = dataset_group.require_group(str(i + 1))
                    for flag, index in zip(['train', 'test', 'valid'],
                                           [train_index, test_index, valid_index]):
                        if flag in fold:
                            fold.pop(flag)
                        fold[flag] = subject_list[index].tolist()
                    continue
            print('Dataset: {:8s}    group: {:6s}    {:s}'.format(dataset, gro, str(sizes)))


def prepare_scheme(parameters: dict,
                   data_hdf5: h5py.File = None,
                   folds_hdf5: h5py.Group = None,
                   scheme_hdf5: h5py.Group = None):
    if data_hdf5 is None:
        data_hdf5_path = b'F:/OneDriveOffL/Data/Data/DCAE_data.hdf5'
        data_hdf5 = hdf5_handler(data_hdf5_path, 'a')
    if folds_hdf5 is None:
        folds_hdf5_path = b'F:/OneDriveOffL/Data/Data/DCAE_folds.hdf5'
        folds_hdf5 = hdf5_handler(folds_hdf5_path, 'a')
    if scheme_hdf5 is None:
        scheme_hdf5_path = b'F:/OneDriveOffL/Data/Data/DCAE_scheme.hdf5'
        scheme_hdf5 = hdf5_handler(scheme_hdf5_path, 'a')

    scheme = parameters['scheme']
    features = parameters['features']
    features_str = '_'.join(features)
    datasets = parameters['datasets']
    scheme_group = scheme_hdf5.require_group('scheme {:d}'.format(scheme))

    if scheme == 1:
        folds = folds_hdf5['ABIDE/{:s}'.format(features_str)]
        basic_fold = folds_hdf5['ADHD/{:s}/0'.format(features_str)]
        basic_data = np.array(basic_fold['train data'])
        basic_label = np.array(basic_fold['train label'])

        for fold_index in folds:
            fold = folds[fold_index]
            fold_new = scheme_group.require_group('fold {:s}'.format(fold_index))

            # Pre-training
            data = np.array(fold['train data'])
            label = np.array(fold['train label'])
            data = data[label == 0]
            data = np.concatenate((basic_data, data), axis=0)
            data, mean, std = data_normalization(data=data, axis=0)
            data = split_slices(data)
            create_dataset_hdf5(group=fold_new,
                                name='pre training data',
                                data=data,
                                )

            for tvt in ['train', 'valid', 'test']:
                data = np.array(fold['{:s} data'.format(tvt)])
                label = np.array(fold['{:s} label'.format(tvt)])
                data, mean, std = data_normalization(data=data, axis=0)
                data = split_slices(data)

                create_dataset_hdf5(group=fold_new,
                                    name='{:s} data'.format(tvt),
                                    data=data,
                                    )
                create_dataset_hdf5(group=fold_new,
                                    name='{:s} label'.format(tvt),
                                    data=label,
                                    )
    elif scheme == 3:
        folds = folds_hdf5['ABIDE/{:s}'.format(features_str)]
        pass
        for fold_index in folds:
            fold = folds[fold_index]
            fold_new = scheme_group.require_group('fold {:s}'.format(fold_index))
            for tvt in ['train', 'valid', 'test']:
                data = np.array(fold['{:s} data'.format(tvt)])
                label = np.array(fold['{:s} label'.format(tvt)])
                create_dataset_hdf5(group=fold_new,
                                    name='{:s} data'.format(tvt),
                                    data=data,
                                    )
                create_dataset_hdf5(group=fold_new,
                                    name='{:s} label'.format(tvt),
                                    data=label,
                                    )
    elif scheme == 4:
        for dataset in datasets:
            folds = scheme_group.require_group('{:s}/{:s}'.format(dataset, features_str))
            scheme_exp = folds.require_group('experiment')
            scheme_exp.attrs['dataset'] = dataset
            scheme_exp.attrs['features'] = [feature.encode() for feature in features]

            datasets_all = ['ABIDE', 'ABIDE II', 'ADHD', 'FCP']
            datasets_all.remove(dataset)

            # pre train data
            health_data = np.concatenate([load_fold(dataset_group=data_hdf5['{:s}/subjects'.format(d)],
                                                    features=features,
                                                    dataset=d,
                                                    fold_group=folds_hdf5['{:s}/0'.format(d)])['train data']
                                          for d in datasets_all], axis=0)
            health_data_size = np.size(health_data, axis=0)

            # processing data for each fold
            for fold_index in np.arange(start=1, stop=6):
                fold = folds.require_group('fold {:d}'.format(fold_index))

                # load and scaling
                fold_data = load_fold(dataset_group=data_hdf5['{:s}/subjects'.format(dataset)],
                                      experiment=scheme_exp,
                                      fold_group=folds_hdf5['{:s}/{:d}'.format(dataset, fold_index)])
                scaled_data = np.concatenate((health_data, fold_data['train data']), axis=0)
                scaled_data, mean, std = data_normalization(data=scaled_data)
                create_dataset_hdf5(group=fold,
                                    name='mean',
                                    data=mean,
                                    )
                create_dataset_hdf5(group=fold,
                                    name='std',
                                    data=std,
                                    )

                pre_train_data = scaled_data[0:health_data_size]
                fold_data['pre train data'] = pre_train_data
                fold_data['train data'] = scaled_data[health_data_size:]
                fold_data['valid data'], _, _ = data_normalization(data=fold_data['valid data'],
                                                                   mean=mean,
                                                                   std=std)
                fold_data['test data'], _, _ = data_normalization(data=fold_data['test data'],
                                                                  mean=mean,
                                                                  std=std)

                for tvt in ['pre train', 'train', 'valid', 'test']:
                    for flag in ['data', 'label']:
                        name = '{:s} {:s}'.format(tvt, flag)
                        if name not in fold_data:
                            continue

                        create_dataset_hdf5(group=fold,
                                            name=name,
                                            data=fold_data[name],
                                            )


def main(parameters: dict):
    hdf5 = load_datas(cover=True)
    prepare_folds(parameters=parameters)
    parameters['datasets'] = ['ABIDE']
    prepare_scheme(parameters=parameters)


if __name__ == '__main__':
    parameters = {
        'features': ['falff'],
        # 'datasets': ['ABIDE'],
        'datasets': ['ABIDE', 'ABIDE II', 'ADHD', 'FCP'],
        'fold_nums': [[0, 5], [0, 5], [0, 5], [0]],
        'groups': [['health', 'all'], ['health', 'all'], ['health', 'all'], ['all']],
        'scheme': 4,
    }

    main(parameters=parameters)
