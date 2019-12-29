import numpy as np


def load_data():
    data_path = 'F:/OneDriveOffL/Data/Data/ABIDE/Results/fALFF_FunImgARCW/fALFFMap_0051585.nii'
    data = load_nifti_data(data_path=data_path, mask=True, normalization=True)
    fig = plt.figure()
    plt.imshow(data[:, 37, :])
    plt.show()


from Dataset.prepare_data import load_datas


def load_datas():
    hdf5 = load_datas()


from nilearn.datasets import fetch_atlas_harvard_oxford
from nilearn.plotting import plot_glass_brain
from nibabel import load


def download_harvard_oxford_atlas():
    atlas_dir = 'F:/OneDriveOffL/Program/Python/'
    atlas = fetch_atlas_harvard_oxford(atlas_name='cort-maxprob-thr25-2mm',
                                       data_dir=atlas_dir)
    volume = load(atlas['maps'])
    plot_glass_brain(volume)
    plt.show()


def check_zeros():
    """
    check if all the values in .nii file are zeros for each datasets and features
    :return:
    """
    hdf5_path = b'F:/OneDriveOffL/Data/Data/DCAE_data.hdf5'
    hdf5 = hdf5_handler(hdf5_path, 'a')
    features = ['falff', 'reho', 'vmhc']
    datasets = ['ABIDE II', 'ABIDE', 'ADHD', 'FCP']
    for dataset in datasets:
        print('Dataset: {:s}'.format(dataset))
        subjects = hdf5['{:s}/subjects'.format(dataset)]
        for subject in subjects:
            for feature in features:
                try:
                    data = np.array(subjects[subject][feature])
                except Exception as e:
                    print('Dataset: {:s}  subject: {:s}  feature: {:s} {}'.format(dataset, subject, feature, e))
                if np.isnan(data).sum() > 0:
                    print('Dataset: {:s}  subject: {:s}  feature: {:s}'.format(dataset, subject, feature))


from Dataset.utils import data_normalization, onehot2vector


def normalization_post_processing():
    scheme_hdf5_path = b'F:/OneDriveOffL/Data/Data/DCAE_scheme.hdf5'
    scheme_hdf5 = hdf5_handler(scheme_hdf5_path, 'a')
    for fold_index in np.arange(start=1, stop=6):
        fold = scheme_hdf5['scheme 4/ABIDE/falff/fold {:d}'.format(fold_index)]
        for flag in ['pre train', 'train', 'valid', 'test']:
            name = '{:s} data'.format(flag)
            data = np.array(fold[name])
            norm_data, _, _ = data_normalization(data=fold[name], standardization=False, normalization=True)

            if np.isnan(data).sum() == 0:
                print('Normalized {:s} in fold {:d}, max: {:f}    min: {:f}'.format(name, fold_index, np.max(data),
                                                                                    np.min(data)))
            else:
                print('error.')

            if 'norm {:s}'.format(name) in fold:
                fold.pop('norm {:s}'.format(name))
            create_dataset_hdf5(group=fold,
                                data=data,
                                name='unnorm {:s}'.format(name),
                                )
            create_dataset_hdf5(group=fold,
                                data=norm_data,
                                name=name,
                                )


def load_folds():
    scheme_hdf5_path = b'F:/OneDriveOffL/Data/Data/DCAE_scheme.hdf5'
    scheme_hdf5 = hdf5_handler(scheme_hdf5_path, 'a')
    scheme_group = scheme_hdf5.require_group('scheme {:d}'.format(4))
    datasets = ['ABIDE']
    features = ['falff']
    features_str = '_'.join(features)

    for dataset in datasets:
        folds = scheme_group.require_group('{:s}/{:s}'.format(dataset, features_str))
        for fold_index in np.arange(start=1, stop=6):
            fold = folds['fold ' + str(fold_index)]
            for flag in ['train', 'valid', 'test']:
                label = np.array(fold['{:s} label'.format(flag)])
                label = onehot2vector(label)
                create_dataset_hdf5(group=fold,
                                    data=label,
                                    name='{:s} label'.format(flag))


from Model.utils_model import get_metrics


def tensorflow_metrics():
    logits = tf.placeholder(tf.float32, [4, 2])
    labels = tf.Variable([[0, 1],
                          [0, 1],
                          [1, 0],
                          [1, 0],
                          ], dtype=tf.float32)

    metrics = get_metrics(prediction=labels, ground_truth=logits)

    with tf.Session() as sess:
        sess.run(tf.initialize_all_variables())
        for i in range(10):
            l = np.random.random(size=[4, 1])
            l = np.concatenate((np.ones(shape=[4, 1]) - l, l), axis=1)

            met = sess.run(fetches=[metrics],
                           feed_dict={
                               logits: l
                           })
            print(met)


def extract_non_zero_cube(data: np.ndarray) -> dict:
    """
    Extract a non-zero cube from the input data
    :param data: 3 dimensional data array
    :return: dictionary of {'cube': the non-zero cube extracted from the input data,
                             'bounds': the bound of the non-zero cube in the input data
                             }
    """
    shape = np.shape(data)
    if len(shape) != 3:
        raise TypeError('The dimension of the input data must be 3!')

    bounds = np.zeros(shape=[3, 2], dtype=int)
    for dim in range(3):
        dim_all = [0, 1, 2]
        dim_all.remove(dim)
        new_dim = [dim]
        new_dim.extend(dim_all)
        data_transpose = data.transpose(new_dim)
        for index in range(shape[dim]):
            if np.max(data_transpose[index, :, :]) > 0:
                if bounds[dim][0] == 0:
                    bounds[dim][0] = index
                else:
                    bounds[dim][1] = index
    return {'bounds': bounds,
            'mask': data == 0,
            'dimension': [dim[1] - dim[0] for dim in bounds],
            }


from Schemes import parse_aal_regions


def write_aal_parameters():
    aal_xml_path = 'Data/AAL/AAL.xml'
    aal_region_names = parse_aal_regions(aal_xml_path)

    aal_hdf5_path = b'F:/OneDriveOffL/Data/Data/DCAE_aal.hdf5'
    aal_hdf5 = hdf5_handler(aal_hdf5_path)

    MNI = aal_hdf5.require_group('MNI')
    aal = load_nifti_data('Data/AAL/aal_61_73_61.nii')
    for region_index in np.arange(start=1, stop=117):
        region_name = aal_region_names[region_index - 1]
        region_group = MNI.require_group(str(region_index))

        region_mask = aal != region_index
        region = np.array(aal)
        region[region_mask] = 0

        voxel_num = np.count_nonzero(region)
        region = extract_non_zero_cube(region)
        region['voxel_num'] = voxel_num

        bounds = region['bounds']
        region_mask = region_mask[
                      bounds[0][0]:bounds[0][1],
                      bounds[1][0]:bounds[1][1],
                      bounds[2][0]:bounds[2][1]
                      ]

        region['mask'] = region_mask

        region_group.attrs['name'] = region_name
        for attr in region:
            create_dataset_hdf5(group=region_group,
                                name=attr,
                                data=region[attr])


import tensorflow as tf


def convolution():
    value = np.reshape(np.arange(start=1, stop=7, dtype=float), newshape=[1, 6, 1])
    x1 = tf.constant(value=value, dtype=tf.float32)

    padding = 'SAME'
    kernel_size = 3
    for num in range(1, 6):
        strides_num = num
        kernel = tf.constant(1.0, shape=[kernel_size, 1, 1])
        y1 = tf.nn.convolution(x1, kernel, strides=[strides_num], padding=padding)
        with tf.Session() as sess:
            y1_v = sess.run([y1])
            y1_v = np.squeeze(np.array(y1_v))
            print('Strides: {:}'.format(num))
            print(y1_v)


def aal():
    aal_hdf5_path = b'E:/OneDriveOffL/Data/Data/DCAE_aal.hdf5'
    aal_hdf5 = hdf5_handler(aal_hdf5_path)

    MNI_group = aal_hdf5['MNI']
    for region_index in np.arange(1, 91):
        region_group = MNI_group[str(region_index)]
        bounds = np.array(region_group['bounds'])
        size = bounds[:, 1] - bounds[:, 0]
        print('Region: {:d}  Size: {:}'.format(region_index, size))


from Layer.LayerConstruct import build_layer


def spp_net_test_code():
    arguments = {
        'type': 'Convolutions3D',
        'kernel_shape': [5, 5, 5, 1, 4],
        'activation': tf.nn.relu,
        'strides': [1, 1, 1, 1, 1],
        'padding': 'VALID',
        'view_num': 90,
        'scope': 'convs_1'
    }

    layer = build_layer(arguments)

    input_tensors = []
    for i in range(90):
        width = np.random.randint(low=5, high=21)
        height = np.random.randint(low=5, high=21)
        deep = np.random.randint(low=5, high=21)
        input = tf.placeholder(shape=[100, width, height, deep, 1], dtype=tf.float32)
        input_tensors.append(input)

    output = layer(input_tensors)

    arguments = {
        'type': 'SpatialPyramidPool3D',
        'kernel_shape': [4, 2, 1],
        'activation': tf.nn.max_pool3d,
        'scope': 'spp_1'
    }

    spp_layer = build_layer(arguments=arguments)
    output = spp_layer(output)


import sklearn.preprocessing as prep


def mapStd(X, X_train, X_valid, X_test):
    [subjNum, n_roi0, n_roi1] = np.shape(X)
    X = np.reshape(X, [subjNum, n_roi0 * n_roi1])
    preprocessor = prep.StandardScaler().fit(X)
    X_train = np.reshape(X_train, [-1, n_roi0 * n_roi1])
    X_valid = np.reshape(X_valid, [-1, n_roi0 * n_roi1])
    X_test = np.reshape(X_test, [-1, n_roi0 * n_roi1])
    X_train = preprocessor.transform(X_train)
    X_valid = preprocessor.transform(X_valid)
    X_test = preprocessor.transform(X_test)
    X_train = np.reshape(X_train, [-1, n_roi0, n_roi1])
    X_valid = np.reshape(X_valid, [-1, n_roi0, n_roi1])
    X_test = np.reshape(X_test, [-1, n_roi0, n_roi1])
    return X_train, X_valid, X_test


def XXY_data_preprocessing():
    for fold_index in range(5):
        # frame = Architecture(scheme='BrainNetCNN')
        # frame.model.build_structure()
        hdf5_path = b'F:/OneDriveOffL/Data/Data/DCAE_scheme.hdf5'
        hdf5 = hdf5_handler(hdf5_path, 'a')

        group = hdf5['scheme {:d}/{:s}/{:s}/fold {:d}'.format(6, 'ABIDE', 'FC', fold_index + 1)]

        # # 测试在初始化过程中读取的参数与XXY中存储的是否相同
        # weights = sio.loadmat('weight.mat')
        # biass = sio.loadmat('bias.mat')
        # acc = sio.loadmat('acc.mat')
        # tensors = {}
        # for layer in ['E2N1', 'N2G1', 'hidden1', 'hidden2']:
        #     tensor = sio.loadmat('tensors_{:s}'.format(layer))
        #     tensors[layer] = tensor
        #     weight = weights[layer]
        #     bias = biass[layer]
        #     net_weight, net_bias = frame.model.sess.run([frame.model.structure['tensors'][layer]['weight'],
        #                                                      frame.model.structure['tensors'][layer]['bias']])
        #     diff = [weight - net_weight, bias - net_bias]
        #     print(np.max(diff[0]), np.max(diff[1]))
        #     pass
        load_data = sio.loadmat(
            'F:/OneDriveOffL/Data/Data/BrainNetCNN/noNorm_311/ALLASD{:d}_NETFC_SG_Pear.mat'.format(fold_index + 1))
        X = load_data['net']
        # # X_train = load_data['net']
        # # Y_train = load_data['all_labels']
        X_train = np.array(load_data['net_train'])
        X_valid = np.array(load_data['net_valid'])
        X_test = np.array(load_data['net_test'])
        Y_train = np.array(load_data['Y_train'])
        Y_valid = np.array(load_data['Y_valid'])
        Y_test = np.array(load_data['Y_test'])
        # Y_train = vecter2onehot(Y_train)
        # Y_valid = vecter2onehot(Y_valid)
        # Y_test = vecter2onehot(Y_test)
        #
        X_train, X_valid, X_test = mapStd(X, X_train, X_valid, X_test)
        X_train = np.expand_dims(np.array(load_data['net_train']), -1)
        X_valid = np.expand_dims(np.array(load_data['net_valid']), -1)
        X_test = np.expand_dims(np.array(load_data['net_test']), -1)

        data = {'train data': X_train,
                'valid data': X_valid,
                'test data': X_test,
                'train label': Y_train,
                'valid label': Y_valid,
                'test label': Y_test}
        for tvt in ['train', 'valid', 'test']:
            for flag in ['data', 'label']:
                name = '{:s} {:s}'.format(tvt, flag)
                create_dataset_hdf5(group=group,
                                    name=name,
                                    data=data[name])

    # print(acc[tvt+'_acc'])
    # frame.model.feedforward(data=X_train,
    #                             label=Y_train,
    #                             epoch=1,
    #                             tag='test',
    #                             if_save=False,
    #                             test_tensors=tensors,
    #                             )


from ops.sparse import sparse_to_matrix


def Laplacian():
    sparse = {(0, 1): 1,
              (1, 2): 1,
              (2, 3): 1,
              (3, 4): 1,
              }
    A = sparse_to_matrix(sparse, shape=[5, 5])
    A += A.T
    print(A)


from Dataset.utils import basic_path


def analyse_results(schemes: list = None,
                    times: int = 0):
    if schemes is None:
        schemes = ['BrainNetCNN', 'GraphNN']

    result_path = '{:s}/DCAE_results.hd5'.format(basic_path).encode()
    result_hdf5 = hdf5_handler(result_path)
    for scheme in schemes:
        scheme_group = result_hdf5.require_group('scheme {:}'.format(scheme))
        fold_num = 5
        result_valid = np.zeros(shape=[times, fold_num])
        result_test = np.zeros(shape=[times, fold_num])
        for time_index, time in enumerate(scheme_group):
            if time_index >= times:
                break
            time_group = scheme_group[time]
            for fold_index, fold in enumerate(time_group):
                fold_group = time_group[fold]
                result_valid[time_index, fold_index] = np.array(fold_group['max_acc_valid'])
                result_test[time_index, fold_index] = np.array(fold_group['max_acc_test'])

        print('Scheme {:s}:'.format(scheme))
        print('Mean: valid {}\ttest {}'.format(np.mean(result_valid[np.nonzero(result_valid)]),
                                               np.mean(result_test[np.nonzero(result_test)])))



class Test:

    def __init__(self):
        A = 89
        B = 90

        tf.SparseTensor()
        input = np.random.normal(size=[A, B])
        self.adj_matrix = np.random.normal(size=[B, A, A]) > 0
        self.weight = np.ones(shape=[B, A])

        # convolution
        weight = np.repeat(np.expand_dims(self.weight, axis=2), repeats=A, axis=2)
        weight = self.adj_matrix * weight

        input_tensor = np.transpose(input, axes=[1, 0])
        input_tensor = np.expand_dims(input_tensor, axis=1)
        input_tensor = np.repeat(input_tensor, repeats=A, axis=1)

        output = input_tensor * weight

        # [B, A]
        output = np.transpose(np.sum(output, axis=0), axes=[1, 0])
        # Test
        for row_index in range(A):
            for col_index in range(A):
                error = output[row_index, col_index] - np.matmul(input[row_index, :], weight[:, col_index, row_index])
                print(error)
        pass


from Layer.GraphNN import GraphConnected


def test_graph_NN():
    input_place = tf.placeholder(shape=[None, 90, 90, 1], dtype=tf.float32)

    arguments = {'type': 'GraphConv',
                 'kernel_shape': [90, 90, 1, 16],
                 'activation': tf.nn.tanh,
                 'depth': 1,
                 'strides': [1, 1, 1, 1],
                 'padding': 'VALID',
                 'bias': True,
                 'batch_normalization': False,
                 'scope': 'graph_conv_1',
                 'nearest_k': 10,
                 }

    layer = GraphConnected(arguments=arguments)

    output = layer(input_place)

    input = np.random.random(size=[2, 90, 90, 1])
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        out, weight = sess.run(fetches=[layer.tensors['output_matmul'],
                                        layer.adj_matrix,
                                        ],
                               feed_dict={
                                   input_place: input,
                               })
        batch_index = 0
        I_index = 0
        O_index = 0
        count = 0
        for row_index in range(90):
            for col_index in range(90):
                error = out[batch_index, row_index, col_index, O_index] \
                        - np.matmul(input[batch_index, row_index, :, I_index], weight[:, col_index, row_index])
                if error > 1e-5:
                    count += 1
                    print('error num: {:d}'.format(count))

        print('error num: {:d}'.format(count))


from Dataset.utils import get_structure, AAL
from ops.graph import Graph
from ops.sparse import matrix_to_sparse


def get_graph():
    distance = np.array(AAL().get_aal_hdf5()['MNI/61_73_61/Euclidean/distance'])
    adj_matrix = get_structure(distance=distance[0:90, 0:90], nearest_k=10)
    adj_matrix_sparse = matrix_to_sparse(adj_matrix)
    g = Graph(adj_matrix=adj_matrix)
    g.diffuse(max_depth=3)
    graph = g.graph_by_depth(depth=1)
    adj_matrix = g.adj_by_depth(depth=1)
    graph_sparse = matrix_to_sparse(graph[0, :])
    adj_sparse_0 = matrix_to_sparse(adj_matrix[..., 0])
    adj_sparse_1 = matrix_to_sparse(adj_matrix[..., 1])
    for sparse in [graph_sparse, adj_sparse_0, adj_sparse_1, adj_matrix_sparse]:
        sparse = sorted(sparse.items(), key=lambda x: x[0][0] if not isinstance(x[0], int) else x[0])
        for item in sparse:
            print(item)

        print()
    pass


from Dataset.utils import hdf5_handler, create_dataset_hdf5


def save_XXY_dataset(scheme: str = None):
    if scheme is None:
        scheme = 'BrainNetCNN'

    hdf5_path = b'F:/OneDriveOffL/Data/Data/DCAE_scheme.hdf5'
    for this_fold in range(5):
        group = hdf5_handler(hdf5_path, 'a')
        group = group['/scheme {:s}/ABIDE/FC/fold {:d}'.format(scheme, this_fold + 1)]
        data = sio.loadmat('F:/OneDriveOffL/Data/Data/BrainNetCNN/noNorm_311/ALLASD{}_NETFC_SG_Pear.mat'.
                           format(this_fold + 1))
        X = data['net']
        X_train = np.array(data['net_train'])
        X_valid = np.array(data['net_valid'])
        X_test = np.array(data['net_test'])
        Y_train = np.array(data['Y_train'])
        Y_valid = np.array(data['Y_valid'])
        Y_test = np.array(data['Y_test'])

        X_train, X_valid, X_test = mapStd(X, X_train, X_valid, X_test)

        # Yao
        datas = {'train data': X_train,
                 'valid data': X_valid,
                 'test data': X_test,
                 'train label': Y_train,
                 'valid label': Y_valid,
                 'test label': Y_test}

        mask = np.expand_dims(np.expand_dims(np.eye(90).astype(bool),
                                             axis=0),
                              axis=-1)
        for tvt in ['train', 'valid', 'test']:
            for flag in ['data', 'label']:
                name = '{:s} {:s}'.format(tvt, flag)
                data = datas[name]

                data_size = np.size(data, 0)
                if flag == 'data':
                    data = np.expand_dims(data, axis=-1)
                create_dataset_hdf5(group=group,
                                    name=name,
                                    data=data)
        continue


from Dataset.utils import get_folds_hdf5, get_data_hdf5, t_test
from Dataset.load_files import load_fold, load_nifti_data


def significance_analyse(fold_nums: int = 5,
                         dataset: str = 'ABIDE',
                         feature: str = 'FC',
                         load_statistics: bool = True,
                         ):
    """
    Analyse the spatial property of edges that significance abnormal between normal control and patience
    :return:
    """
    folds_hdf5 = get_folds_hdf5()
    data_hdf5 = get_data_hdf5()
    folds_group = folds_hdf5['{:s}'.format(dataset)]
    dataset_group = data_hdf5['{:s}/subjects'.format(dataset)]
    statistics = []
    for fold_index in np.arange(fold_nums):

        if load_statistics:
            statistic = np.load('statistics.npy')[fold_index]
        else:
            fold_group = folds_group['{:d}'.format(fold_index + 1)]
            data = load_fold(dataset_group=dataset_group,
                             fold_group=fold_group,
                             features=[feature],
                             dataset=dataset)
            train_data = data['train data']
            train_label = data['train label']

            valid_data = data['valid data']
            valid_label = data['valid label']

            test_data = data['test data']
            test_label = data['test label']
            train_statistic = t_test(normal_controls=train_data[train_label == 1],
                                     patients=train_data[train_label == 0],
                                     )
            valid_statistic = t_test(normal_controls=valid_data[valid_label == 1],
                                     patients=valid_data[valid_label == 0],
                                     )
            test_statistic = t_test(normal_controls=test_data[test_label == 1],
                                    patients=test_data[test_label == 0],
                                    )

            statistic = {'train': train_statistic,
                         'valid': valid_statistic,
                         'test': test_statistic,
                         }

        # $ S \in \mathbb{R}^{N \times N}$,
        # $S_{ij} = 1$ if edge (i, j) is significance abnormal, otherwise $S_{ij} = 0$.
        # $S^{(1)}, S^{(2)}, S^{(3)}$ indicate the significance matrix of train, valid and test dataset respectively.

        S_1 = statistic['train']['hypothesis']
        S_2 = statistic['valid']['hypothesis']
        S_3 = statistic['test']['hypothesis']

        S_12 = S_1 * S_2
        S_13 = S_1 * S_3
        S_23 = S_2 * S_3
        S_123 = S_1 * S_2 * S_3

        print('\r\nfold {:d}'.format(fold_index + 1))
        nonzero_dict = {'S_1': S_1,
                        'S_2': S_2,
                        'S_3': S_3,
                        'S_12': S_12,
                        'S_13': S_13,
                        'S_23': S_23,
                        'S_123': S_123
                        }
        statistic.update(nonzero_dict)
        nonzero_key = ['S_1', 'S_2', 'S_3', 'S_12', 'S_13', 'S_23', 'S_123']
        for key in nonzero_key:
            print('{:4s}: \t{:d}'.format(key, np.count_nonzero(nonzero_dict[key])))

        statistics.append(statistic)
    np.save('statistics.npy', statistics)


# author: Gael Varoquaux <gael.varoquaux@inria.fr>
# License: BSD 3 clause
# Copyright: INRIA

import numpy as np
from scipy import linalg, io as sio
from sklearn.datasets import make_sparse_spd_matrix
from sklearn.covariance import GraphicalLassoCV, ledoit_wolf
import matplotlib.pyplot as plt


def sparse_inverse_covariance_estimation():
    # #############################################################################
    # Generate the data
    n_samples = 60
    n_features = 20

    prng = np.random.RandomState(1)
    prec = make_sparse_spd_matrix(n_features, alpha=.98,
                                  smallest_coef=.4,
                                  largest_coef=.7,
                                  random_state=prng)
    cov = linalg.inv(prec)
    d = np.sqrt(np.diag(cov))
    cov /= d
    cov /= d[:, np.newaxis]
    prec *= d
    prec *= d[:, np.newaxis]
    X = prng.multivariate_normal(np.zeros(n_features), cov, size=n_samples)
    X -= X.mean(axis=0)
    X /= X.std(axis=0)

    # #############################################################################
    # Estimate the covariance
    emp_cov = np.dot(X.T, X) / n_samples

    model = GraphicalLassoCV(cv=5)
    model.fit(X)
    cov_ = model.covariance_
    prec_ = model.precision_

    lw_cov_, _ = ledoit_wolf(X)
    lw_prec_ = linalg.inv(lw_cov_)

    # #############################################################################
    # Plot the results
    plt.figure(figsize=(10, 6))
    plt.subplots_adjust(left=0.02, right=0.98)

    # plot the covariances
    covs = [('Empirical', emp_cov), ('Ledoit-Wolf', lw_cov_),
            ('GraphicalLassoCV', cov_), ('True', cov)]
    vmax = cov_.max()
    for i, (name, this_cov) in enumerate(covs):
        plt.subplot(2, 4, i + 1)
        plt.imshow(this_cov, interpolation='nearest', vmin=-vmax, vmax=vmax,
                   cmap=plt.cm.RdBu_r)
        plt.xticks(())
        plt.yticks(())
        plt.title('%s covariance' % name)

    # plot the precisions
    precs = [('Empirical', linalg.inv(emp_cov)), ('Ledoit-Wolf', lw_prec_),
             ('GraphicalLasso', prec_), ('True', prec)]
    vmax = .9 * prec_.max()
    for i, (name, this_prec) in enumerate(precs):
        ax = plt.subplot(2, 4, i + 5)
        plt.imshow(np.ma.masked_equal(this_prec, 0),
                   interpolation='nearest', vmin=-vmax, vmax=vmax,
                   cmap=plt.cm.RdBu_r)
        plt.xticks(())
        plt.yticks(())
        plt.title('%s precision' % name)
        if hasattr(ax, 'set_facecolor'):
            ax.set_facecolor('.7')
        else:
            ax.set_axis_bgcolor('.7')

    # plot the model selection metric
    plt.figure(figsize=(4, 3))
    plt.axes([.2, .15, .75, .7])
    plt.plot(model.cv_alphas_, np.mean(model.grid_scores_, axis=1), 'o-')
    plt.axvline(model.alpha_, color='.5')
    plt.title('Model selection')
    plt.ylabel('Cross-validation score')
    plt.xlabel('alpha')

    plt.show()


def save_sparse_inverse_covariance_matrices(name: str = 'SIC'):
    path = b'F:\OneDriveOffL\Data\Data\DCAE_data.hdf5'
    hdf5 = hdf5_handler(path)
    subjects = hdf5['ABIDE/subjects']

    path = 'F:\OneDriveOffL\Program\Matlab\GMGLASS\data.mat'
    SICs = sio.loadmat(path)['Ps']

    for sub_ID, SIC in zip(subjects, SICs):
        # raw_data = np.array(subjects[sub_ID]['raw_data'])
        # sio.savemat('F:\OneDrive - emails.bjut.edu.cn\Paper\姚垚\\2019.4.25例会PPT\\raw_data.mat', {'raw_data': raw_data})

        sub_group = subjects.require_group(sub_ID)

        if name == 'SIC':
            create_dataset_hdf5(data=SIC,
                                name='SIC',
                                group=sub_group)
        elif name == 'Laplacian':
            feature_num = np.size(SIC, 0)
            SIC[np.eye(feature_num) == 1] = 1
            degree = np.diag(np.power(np.sum(SIC, axis=0), -0.5))
            laplacian = np.matmul(degree, np.matmul(SIC, degree))
            create_dataset_hdf5(data=laplacian,
                                name='Laplacian',
                                group=sub_group)



save_sparse_inverse_covariance_matrices(name='Laplacian')


def analyse_weight():
    # Analyse the trained filters with physical connectivity
    aal_hdf5 = hdf5_handler(b'F:/OneDriveOffL/Data/Data/DCAE_aal.hdf5')
    adj_matrix = np.array(aal_hdf5['MNI/61_73_61/Adjacent/adj_matrix'])[:90, :90]

    order = 1
    adj_matrix_ho = np.copy(adj_matrix)
    for i in range(order - 1):
        adj_matrix_ho += np.matmul(adj_matrix_ho, adj_matrix)
    adj_matrix_ho[adj_matrix_ho != 0] = 1

    schemes = ['weight_GLasso', 'weight_CNNEW', 'weight_test']
    weights = [sio.loadmat('F:/Weight/{:s}.mat'.format(scheme))['weight'] for scheme in schemes]
    out_channels = 64
    n_feature = 90

    FP_rates = {scheme: () for scheme in schemes}
    for weight, scheme in zip(weights, schemes):
        out_channels = np.size(weight, -1)
        FP_rate = []
        for i in range(out_channels):
            w = np.abs(weight[..., 0, i])
            w[np.eye(n_feature) == 1] = 0
            mean = np.mean(w)
            w[w >= mean] = 1
            w[w < mean] = 0
            sparsity = np.sum(w) / n_feature ** 2

            # TP = np.count_nonzero(w * adj_matrix)
            # TN = np.count_nonzero((w - 1) * (adj_matrix - 1))
            # FP = np.count_nonzero(w * (adj_matrix - 1))
            # FN = np.count_nonzero((w - 1) * adj_matrix)
            #
            # precision = TP / (TP + FP)
            # recall = TP / (TP + FN)
            # specificity = TN / (TN + FP)
            # f1 = 2 * precision * recall / (precision + recall)

            TP = np.count_nonzero(w * adj_matrix_ho)
            TN = np.count_nonzero((w - 1) * (adj_matrix_ho - 1))
            FP = np.count_nonzero(w * (adj_matrix_ho - 1))
            FN = np.count_nonzero((w - 1) * adj_matrix_ho)

            precision = TP / (TP + FP)
            recall = TP / (TP + FN)
            specificity = TN / (TN + FP)
            f1 = 2 * precision * recall / (precision + recall)
            FP_rate.append(precision)
        print('Mean scheme {:s}:{:f}'.format(scheme, np.mean(np.array(FP_rate))))