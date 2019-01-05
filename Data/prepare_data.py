import h5py
import random
import numpy as np
import pandas as pd
from docopt import docopt
from Data.utils_prepare_data import hdf5_handler, load_fold, data_normalization, \
    data_normalization_fold, get_subjects, create_dataset_hdf5, split_slices
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

    # features = ['falff', 'reho', 'vmhc', 'FC']
    features = ['FC']
    datasets = ['ABIDE']

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

    seed = np.random.randint(low=1, high=100)
    print('Seed: {:}'.format(seed))
    random.seed(seed)

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
                    for flag, index in zip(['train', 'valid', 'test'],
                                           [train_index, valid_index, test_index]):
                        if flag in fold:
                            fold.pop(flag)
                        fold[flag] = subject_list[index].tolist()
                    continue

            print('Dataset: {:8s}    group: {:6s}    {:s}'.format(dataset, gro, str(sizes)))


def prepare_scheme(parameters: dict,
                   data_hdf5: h5py.File = None,
                   folds_hdf5: h5py.Group = None,
                   scheme_hdf5: h5py.Group = None,
                   normalization: bool = False,
                   standardization: bool = False,
                   ):
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
    print('Preparing scheme {:} ...'.format(scheme))

    features = parameters['features']
    features_str = '_'.join(features)
    datasets = parameters['datasets']
    scheme_group = scheme_hdf5.require_group('scheme {:}'.format(scheme))

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
            data, mean, std = data_normalization(data=data,
                                                 axis=0,
                                                 normalization=True
                                                 )
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
                fold_data = np.concatenate((health_data, fold_data['train data']), axis=0)
                fold_data, mean, std = data_normalization(data=fold_data,
                                                          standardization=standardization,
                                                          normalization=normalization,
                                                          )
                create_dataset_hdf5(group=fold,
                                    name='mean',
                                    data=mean,
                                    )
                create_dataset_hdf5(group=fold,
                                    name='std',
                                    data=std,
                                    )

                pre_train_data = fold_data[0:health_data_size]
                fold_data['pre train data'] = pre_train_data
                fold_data['train data'] = fold_data[health_data_size:]
                fold_data['valid data'], _, _ = data_normalization(data=fold_data['valid data'],
                                                                   mean=mean,
                                                                   std=std,
                                                                   standardization=standardization,
                                                                   normalization=normalization,
                                                                   )
                fold_data['test data'], _, _ = data_normalization(data=fold_data['test data'],
                                                                  mean=mean,
                                                                  std=std,
                                                                  standardization=standardization,
                                                                  normalization=normalization,
                                                                  )

                for tvt in ['pre train', 'train', 'valid', 'test']:
                    for flag in ['data', 'label']:
                        name = '{:s} {:s}'.format(tvt, flag)
                        if name not in fold_data:
                            continue

                        create_dataset_hdf5(group=fold,
                                            name=name,
                                            data=fold_data[name],
                                            )
    elif scheme == 5:
        for dataset in datasets:
            folds = scheme_group.require_group('{:s}/{:s}'.format(dataset, features_str))
            scheme_exp = folds.require_group('experiment')
            scheme_exp.attrs['dataset'] = dataset
            scheme_exp.attrs['features'] = [feature.encode() for feature in features]

            # processing data for each fold
            for fold_index in np.arange(start=1, stop=6):
                fold = folds.require_group('fold {:d}'.format(fold_index))

                # load and scaling
                fold_data = load_fold(dataset_group=data_hdf5['{:s}/subjects'.format(dataset)],
                                      experiment=scheme_exp,
                                      fold_group=folds_hdf5['{:s}/{:d}'.format(dataset, fold_index)])

                if normalization:
                    fold_data = data_normalization_fold(data_fold=fold_data)

                for tvt in ['train', 'valid', 'test']:
                    for flag in ['data', 'label']:
                        name = '{:s} {:s}'.format(tvt, flag)
                        if name not in fold_data:
                            continue

                        create_dataset_hdf5(group=fold,
                                            name=name,
                                            data=fold_data[name],
                                            )
    elif scheme == 6:
        dataset_str = '_'.join(datasets)
        folds = scheme_group.require_group('{:s}/{:s}'.format(dataset_str, features_str))
        scheme_exp = folds.require_group('experiment')
        scheme_exp.attrs['dataset'] = dataset_str
        scheme_exp.attrs['features'] = [feature.encode() for feature in features]
        scheme_exp.attrs['description'] = 'Autism diagnosis use FC in ABIDE and ABIDE II.'
        for fold_index in np.arange(start=1, stop=6):
            fold = folds.require_group('fold {:d}'.format(fold_index))

            fold_datas = []
            for dataset in datasets:
                fold_datas.append(load_fold(dataset_group=data_hdf5['{:s}/subjects'.format(dataset)],
                                            experiment=scheme_exp,
                                            fold_group=folds_hdf5['{:s}/{:d}'.format(dataset, fold_index)],
                                            dataset=dataset,
                                            )
                                  )

            fold_data = {'train data': np.expand_dims(np.concatenate([fd['train data'] for fd in fold_datas],
                                                                     axis=0), -1),
                         'valid data': np.expand_dims(np.concatenate([fd['valid data'] for fd in fold_datas],
                                                                     axis=0), -1),
                         'test data': np.expand_dims(np.concatenate([fd['test data'] for fd in fold_datas],
                                                                    axis=0), -1),
                         'train label': np.concatenate([fd['train label'] for fd in fold_datas],
                                                       axis=0),
                         'valid label': np.concatenate([fd['valid label'] for fd in fold_datas],
                                                       axis=0),
                         'test label': np.concatenate([fd['test label'] for fd in fold_datas],
                                                      axis=0),
                         }

            if normalization:
                fold_data = data_normalization_fold(data_fold=fold_data)

            for tvt in ['train', 'valid', 'test']:
                for flag in ['data', 'label']:
                    name = '{:s} {:s}'.format(tvt, flag)
                    if name not in fold_data:
                        continue

                    data = fold_data[name]

                    # if name == 'train data':
                    #     mask = np.expand_dims(np.expand_dims(np.load('../statistics.npy')[fold_index - 1]['S_1']
                    #                                          .astype(int), 0), -1)
                    #     data_size = np.size(data, 0)
                    #     mask = np.repeat(mask, axis=0, repeats=data_size)
                    #     data = data * mask

                    create_dataset_hdf5(group=fold,
                                        name=name,
                                        data=data,
                                        )
    elif scheme == 'GraphNN':
        dataset_str = '_'.join(datasets)
        folds = scheme_group.require_group('{:s}/{:s}'.format(dataset_str, features_str))
        scheme_exp = folds.require_group('experiment')
        scheme_exp.attrs['dataset'] = dataset_str
        scheme_exp.attrs['features'] = [feature.encode() for feature in features]
        scheme_exp.attrs['description'] = 'Autism diagnosis use FC in ABIDE and ABIDE II.'
        for fold_index in np.arange(start=1, stop=6):
            fold = folds.require_group('fold {:d}'.format(fold_index))

            fold_datas = []
            for dataset in datasets:
                fold_datas.append(load_fold(dataset_group=data_hdf5['{:s}/subjects'.format(dataset)],
                                            experiment=scheme_exp,
                                            fold_group=folds_hdf5['{:s}/{:d}'.format(dataset, fold_index)],
                                            dataset=dataset,
                                            )
                                  )

            fold_data = {'train data': np.expand_dims(np.concatenate([fd['train data'] for fd in fold_datas],
                                                                     axis=0), -1),
                         'valid data': np.expand_dims(np.concatenate([fd['valid data'] for fd in fold_datas],
                                                                     axis=0), -1),
                         'test data': np.expand_dims(np.concatenate([fd['test data'] for fd in fold_datas],
                                                                    axis=0), -1),
                         'train label': np.concatenate([fd['train label'] for fd in fold_datas],
                                                       axis=0),
                         'valid label': np.concatenate([fd['valid label'] for fd in fold_datas],
                                                       axis=0),
                         'test label': np.concatenate([fd['test label'] for fd in fold_datas],
                                                      axis=0),
                         }

            if normalization:
                fold_data = data_normalization_fold(data_fold=fold_data)

            for tvt in ['train', 'valid', 'test']:
                for flag in ['data', 'label']:
                    name = '{:s} {:s}'.format(tvt, flag)
                    if name not in fold_data:
                        continue

                    data = fold_data[name]

                    # if name == 'train data':
                    #     mask = np.expand_dims(np.expand_dims(np.load('../statistics.npy')[fold_index - 1]['S_1']
                    #                                          .astype(int), 0), -1)
                    #     data_size = np.size(data, 0)
                    #     mask = np.repeat(mask, axis=0, repeats=data_size)
                    #     data = data * mask

                    create_dataset_hdf5(group=fold,
                                        name=name,
                                        data=data,
                                        )


def main(parameters: dict):
    # hdf5 = load_datas(cover=True)
    prepare_folds(parameters=parameters)
    prepare_scheme(parameters=parameters)


if __name__ == '__main__':
    parameters = {
        'features': ['FC'],
        'datasets': ['ABIDE'],
        'fold_nums': [[0, 5]],
        'groups': [['health', 'all']],
        'scheme': 'GraphNN',
    }

    main(parameters=parameters)
