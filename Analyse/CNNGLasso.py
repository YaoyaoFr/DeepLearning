import collections
import os

import numpy as np
import pandas as pd
import scipy.io as sio
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from scipy import sparse

from AAL.ROI import load_sorted_rois
from Dataset.DataBase import DataBase
from Dataset.utils import hdf5_handler, t_test
from Log.log import Log
from Model.Framework import Framework
from ops.matrix import matrix_sort

FRAME = Framework(scheme='CNNGLasso')
DIR_PATH = '/home/ai/data/yaoyao/'


def get_trained_parameters(date: str,
                           clock: str,
                           if_save: bool = False):
    FRAME.log.set_path(date=date, clock=clock)
    dir_path = FRAME.log.dir_path

    for time_dir in os.listdir(dir_path):
        for fold_dir in os.listdir(os.path.join(dir_path, time_dir)):
            # Preparing log director
            FRAME.log.set_path(
                subfolder='{:s}/{:s}'.format(time_dir, fold_dir))
            # Preparing tensorflow graph
            FRAME.log.reset_graph()

            # Rebuild the structure and log
            FRAME.model = FRAME.models[FRAME.scheme](scheme=FRAME.scheme,
                                                     log=FRAME.log,
                                                     dir_path=FRAME.dir_path,
                                                     spe_pas=FRAME.spe_pas,
                                                     )
            FRAME.model.build_structure()

            print(
                'Loading parameters of {:s}/{:s}...'.format(time_dir, fold_dir))
            trained_parameters = FRAME.model.get_parameters(
                restored_model='optimal')
            results = discriminant_power_analyse(
                trained_parameters=trained_parameters)

            if if_save:
                save_path = os.path.join(
                    FRAME.log.dir_path, 'results.mat')
                sio.savemat(save_path, results)
    return trained_parameters


def discriminant_power_analyse(date: str = None,
                               clock: str = None,
                               trained_parameters: dict = None):
    """Analyse important edges for discrimination.
    """

    if trained_parameters is None:
        log = Log(dir_path='/home/ai/data/yaoyao', scheme_folder='CNNGLasso')
        log.set_path(date=date, clock=clock)
        trained_parameters = sio.loadmat(os.path.join(
            log.dir_path, 'trained_parameters.mat'))

    z = np.array([1, -1])
    for layer_scope in ['hidden2', 'hidden1']:
        weight = trained_parameters['{:s}/weight'.format(layer_scope)]
        z = np.matmul(weight, z)

    # N2G
    weight = trained_parameters['N2G1/weight']
    z = np.multiply(weight, z)
    z = np.squeeze(np.sum(np.absolute(z), axis=-1))

    # E2N
    weight = trained_parameters['E2N1/weight_multiply']
    z = np.sum(np.abs(np.multiply(np.squeeze(weight), z)), axis=-1)

    results = {'F': z}

    # save_dir_path = 'F:/OneDriveOffL/Data/Result/Net'
    # save_path = os.path.join(log.dir_path, 'analysed_parameters.mat')

    # sio.savemat(save_path, results)
    return results


def statistical_analyse(date: str,
                        clock: str,
                        top_edges: int = None,
                        p_threshold: float = 0.05,
                        ):
    FRAME.log.set_path(date=date, clock=clock)
    dir_path = FRAME.log.dir_path

    Fs = []
    for time_dir in os.listdir(dir_path):
        for fold_dir in os.listdir(os.path.join(dir_path, time_dir)):
            # Preparing log director
            FRAME.log.set_path(
                subfolder='{:s}/{:s}'.format(time_dir, fold_dir))
            results = sio.loadmat(os.path.join(
                FRAME.log.dir_path, 'results.mat'))
            Fs.append(np.expand_dims(results['F'], axis=0))
    F = np.concatenate(Fs, axis=0)
    zeros = np.zeros(shape=np.shape(F))
    p_value = t_test(F, zeros)['p_value']

    p_value_sparse = collections.OrderedDict(sparse.dok_matrix(
        np.tril(p_value, k=-1), -1))
    p_value_sorted = sorted(
        p_value_sparse.items(), key=lambda x: x[1])

    edge_count = 0
    if top_edges is None:
        top_edges = len(p_value_sorted)

    # Load statistical results of raw data.
    db = DataBase(dataset_list=['ABIDE'])
    hdf5 = hdf5_handler(db.hdf5_path)
    p_values = np.array(hdf5['ABIDE/feature/pearson correlation XXY/p_value'])

    roi_names = load_sorted_rois()
    symmetric_count = 0
    for edge, p_value in p_value_sorted:
        if edge_count >= top_edges:
            break

        if p_threshold is not None and p_value > p_threshold:
            continue

        edge_count += 1
        if roi_names[edge[1] + 1].abbreviation == roi_names[edge[0]+1].abbreviation:
            print('Symmetric edge of {:s}'.format(
                roi_names[edge[1] + 1].abbreviation))
            symmetric_count += 1
        else:
            print('Edge {:d}: {:s} - {:s}\tp-value:{:e}'.format(
                edge_count,
                roi_names[edge[0]+1].roi_name,
                roi_names[edge[1] +
                          1].roi_name,
                p_values[edge[0], edge[1]]))

    print('Symmetric count: {:d}, total count: {:d}'.format(
        symmetric_count, edge_count))


def generate_table3():
    """Generate the content of Table 3.
    1. Top 10 common edges between SIC and discriminative edges of CNNGLasso.
    2. Top 10 edges of CNNGLasso.
    """
    dir_path = '/home/ai/data/yaoyao'

    # Get the edges sorted by their discriminate power
    f_path = os.path.join(dir_path,
                          'Result/DeepLearning/edges_CNNGLasso.mat')

    f_values = np.triu(sio.loadmat(f_path)['edges_CNNGLasso'])
    f_sort_edges = matrix_sort(f_values)[1]

    # Get the edges sorted by its absolute value in sparse inverse covariance matrix
    sic_path = os.path.join(dir_path,
                            'Result/DeepLearning/edges_SIC_NC.mat')
    sic_values = sio.loadmat(sic_path)['edges_SIC_NC']
    sic_sort_edges = matrix_sort(sic_values)[1]

    # Get the common edges between SIC and discriminative edges of CNNGLasso.
    for top_edge in np.arange(100) + 1:
        print(top_edge)
        sic_sort_common = []
        for i in range(top_edge):
            if sic_sort_edges[i] in f_sort_edges[:top_edge]:
                sic_sort_common.append(sic_sort_edges[i])

    # Generate table
    roi_names = load_sorted_rois()

    table_str = ''
    headers = ['Brain Region A', 'Abbr.', 'Brain Region B', 'Abbr.']
    table_str += ' & '.join(header for header in headers) + '\\\\\r\n'

    top_edge = 10
    for part_name, data in zip(['Top 10 common edges between SIC and discriminative edges of CNNGLasso model',
                                'Top 10 discriminative edges of CNNGLasso model'],
                               [sic_sort_common, f_sort_edges]):
        # Header of sub part
        table_str += '\multicolumn{{4}}{{l}}{{{:s}}} \\\\\r\n'.format(
            part_name)

        # Add edges
        edge_count = 0
        for edge in data:
            if roi_names[edge[0]+1].abbreviation == roi_names[edge[1]+1].abbreviation:
                continue

            table_str += '{:s} & {:s}-{:s} & {:s} & {:s}-{:s} \\\\\r\n'.format(
                roi_names[edge[0]+1].roi_name,
                roi_names[edge[0]+1].abbreviation,
                roi_names[edge[0]+1].hemisphere,
                roi_names[edge[1]+1].roi_name,
                roi_names[edge[1]+1].abbreviation,
                roi_names[edge[1]+1].hemisphere,
            )
            edge_count += 1
            if edge_count > top_edge:
                break
    print(table_str)


def plot_propotion():
    # propotion = []
    # for top_edge in np.arange(4005) + 1:
    #     print(top_edge)
    #     sic_sort_common = []
    #     for i in range(top_edge):
    #         if sic_sort_edges[i] in f_sort_edges[:top_edge]:
    #             sic_sort_common.append(sic_sort_edges[i])
    #     propotion.append(len(sic_sort_common)/top_edge)
    # num_edges = np.arange(4005) + 1
    # sio.savemat(os.path.join(dir_path, 'propotion.mat'),
    #             {'num_edges': num_edges,
    #              'propotion': propotion})
    # print('Save propotion results.')

    propotion_data = sio.loadmat(os.path.join(DIR_PATH, 'propotion.mat'))
    num_edges = propotion_data['num_edges'][0]
    propotion = propotion_data['propotion'][0]
    color = cm.viridis(0.5)
    f, ax = plt.subplots(1, 1)
    ax.plot(num_edges, propotion, color=color)
    for prop in [0.7, 0.8, 0.9]:
        optimal_error = 1
        optimal_index = -1
        for index, _ in enumerate(num_edges):
            error = np.abs(prop - propotion[index])
            if error < optimal_error:
                optimal_error = error
                optimal_index = index

        ax.annotate('Proportion={:}, edges={:d}'.format(prop, num_edges[optimal_index]),
                    (num_edges[optimal_index], propotion[optimal_index]),
                    xytext=(0.3, prop), textcoords='axes fraction',
                    arrowprops=dict(facecolor='grey', color='grey'))
    ax.set_xlabel('Top edges')
    ax.set_ylabel('Proportion')
    f.savefig(os.path.join(
        DIR_PATH, 'Result/DeepLearning/Proportion.png'), dpi=1000)


def evalute_models(dataset: dict,
                   date: str,
                   clock: str,
                   ):
    FRAME.log.set_path(date=date, clock=clock)
    for time_dir in os.listdir(FRAME.log.dir_path):
        for fold_dir in os.listdir(os.path.join(FRAME.log.dir_path, time_dir)):
            # Preparing dataset
            data = dataset['{:s}'.format(fold_dir)]

            # Preparing log director
            FRAME.log.set_path(
                subfolder='{:s}/{:s}'.format(time_dir, fold_dir))
            # Preparing tensorflow graph
            FRAME.log.reset_graph()

            # Rebuild the structure and log
            FRAME.model = FRAME.models[FRAME.scheme](scheme=FRAME.scheme,
                                                     log=FRAME.log,
                                                     dir_path=FRAME.dir_path,
                                                     spe_pas=FRAME.spe_pas,
                                                     )
            FRAME.model.build_structure()
            result_fold = FRAME.model.predicting(data=data,
                                                 epoch=0,
                                                 restored_model='final')
            print(result_fold)


def gender_analysis(date: str,
                    clock: str,
                    ):
    # Preparing dataset
    for gender in ['male', 'female']:
        # dataset = frame.models[frame.scheme].load_dataset(
        #     scheme='gender {:s}/{:s}'.format(frame.scheme, gender),
        #     hdf5_file_path=frame.dataset_file_path,
        # )
        dataset = FRAME.models[FRAME.scheme].load_dataset(
            scheme=FRAME.scheme,
            hdf5_file_path=FRAME.dataset_file_path,
        )
        evalute_models(dataset=dataset,
                       date=date,
                       clock=clock)


if __name__ == "__main__":
    # get_trained_parameters(date='2020-01-07',
    #                        clock='15-41-26',
    #                        if_save=True)
    # statistical_analyse(date='2020-01-07',
    #                     clock='15-41-26')
    # generate_table3()
    plot_propotion()
