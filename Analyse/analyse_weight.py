import os
import sys
import numpy as np
import scipy.io as sio

# from AAL.ROI import load_index_roi
from circos.output import get_links
from ops.matrix import matrix_sort, matrix_significance_difference, vector_sort


def discriminant_power_analyse(weights: dict,
                               if_diagonal: bool = False,
                               if_symmetric: bool = True):
    F = np.array([1, -1])
    for layer_scope in ['hidden2', 'hidden1']:
        weight = weights[layer_scope]['weight'][0, 0]
        F = np.matmul(weight, F)

    # N2G
    weight = weights['N2G1']['weight'][0, 0]
    F = np.multiply(weight, F)
    F = np.squeeze(np.sum(np.absolute(F), axis=-1))

    # E2N
    try:
        weight = weights['E2N1']['weight_multiply'][0, 0]
    except Exception:
        weight = weights['E2N1']['weight'][0, 0]
    F = np.sum(np.abs(np.multiply(np.squeeze(weight), F)), axis=-1)

    if not if_diagonal:
        F = F - np.diag(np.diagonal(F))

    if if_symmetric:
        F = np.tril((F + F.T) / 2)

    return F


def SICE_weight_analyse(weight: dict):
    # weight = weight['E2N1']['weight_SICE_bn'][0, 0]
    output_channels = np.shape(weight)[-1]

    SICE_weights = {'weight_SICE_ASD': weight[..., 0:int(output_channels / 2)],
                    'weight_SICE_NC': weight[..., int(output_channels / 2):],
                    }
    return SICE_weights


def load_results(save_dir: str,
                 run_times: int = 20,
                 fold_num: int = 5):
    results = {'weight': {},
               'weight_SICE': {},
               'F': {},
               }
    for run_time in range(run_times):
        weight_time = {}
        weights_SICE_time = {}
        F_time = {}
        for fold_index in range(fold_num):
            save_path = os.path.join(save_dir, 'time {:d}/fold {:d}/parameters.mat'.format(run_time + 1,
                                                                                                fold_index + 1))
            try:
                weights = sio.loadmat(save_path)
            except Exception:
                continue

            # F = discriminant_power_analyse(weights)
            F = weights['F']
            weights_SICE = SICE_weight_analyse(weights['weight_SICE'])
            weight_time['fold {:d}'.format(fold_index + 1)] = weights
            F_time['fold {:d}'.format(fold_index + 1)] = np.expand_dims(F, axis=-1)
            weights_SICE_time['fold {:d}'.format(fold_index + 1)] = weights_SICE
        results['weight']['time {:d}'.format(run_time + 1)] = weight_time
        results['F']['time {:d}'.format(run_time + 1)] = F_time
        results['weight_SICE']['time {:d}'.format(run_time + 1)] = weights_SICE_time
    return results


def statistical_analyse_FC():
    save_path = 'F:/OneDriveOffL/Data/Data/BrainNetCNN/ALLASD_NETFC_SG_Pear.mat'
    data = sio.loadmat(save_path)

    # 0 = Control; 1 = Autism
    labels = data['phenotype'][:, 2]
    fcs = data['net']

    fc_asd = []
    fc_nc = []
    for fc, label in zip(fcs, labels):
        fc = np.expand_dims(fc, axis=0)
        if label == 0:
            fc_nc.append(fc)
        else:
            fc_asd.append(fc)
    fc_asd = np.concatenate(fc_asd, axis=0)
    fc_nc = np.concatenate(fc_nc, axis=0)
    p_value = matrix_significance_difference(fc_asd, fc_nc)
    significance = 1 - np.copy(p_value)
    significance[significance > 0.95] = 1
    significance[significance < 0.95] = 0
    return p_value, significance


def analyse_result_scheme(save_path):
    # results = load_results(save_dir=save_path)

    F = []
    weights_SICE_ASD = []
    weights_SICE_NC = []

    for run_time in range(10):
        # F_time = results['F']['time {:d}'.format(run_time + 1)]
        # weights_SICE_time = results['weight_SICE']['time {:d}'.format(run_time + 1)]
        for fold_index in np.arange(int=0, stop=5):
            try:
                path = os.path.join(save_path, 
                                    'time {:d}'.format(run_time + 1), 
                                    'fold {:d}'.format(fold_index + 1), 
                                    'parameters.mat'
                                    )
                weights = sio.loadmat(path)
            except Exception:
                continue

            F.append(np.expand_dims(weights['F'], axis=-1))
            weigth_SICEs = SICE_weight_analyse(weights['weight_SICE_bn'])
            weights_SICE_ASD.append(weigth_SICEs['weight_SICE_ASD'])
            weights_SICE_NC.append(weigth_SICEs['weight_SICE_NC'])

            # weight_SICE_fold = weights_SICE_time['fold {:d}'.format(fold_index + 1)]
            # weights_SICE_ASD.append(weight_SICE_fold['weight_SICE_ASD'])
            # weights_SICE_NC.append(weight_SICE_fold['weight_SICE_NC'])

    F = np.squeeze(np.concatenate(F, axis=-1))
    weights_SICE_ASD = np.squeeze(np.concatenate(weights_SICE_ASD, axis=-1))
    weights_SICE_NC = np.squeeze(np.concatenate(weights_SICE_NC, axis=-1))
    results = {'F': F,
               'weight_SICE_ASD': weights_SICE_ASD,
               'weight_SICE_NC': weights_SICE_NC,
               }
    return results


def analyse_weights(dir_path: str):
    # p_FC, _ = statistical_analyse_FC()

    save_paths = [
        # 'F:/OneDriveOffL/Data/Result/DeepLearning/2019-07-02/09-51/scheme BrainNetCNNEW',
        # 'F:/OneDriveOffL/Data/Result/DeepLearning/2019-07-02/14-59/scheme CNNWithGLasso',
        os.path.join(dir_path, 'Result/DeepLearning/CNNGLasso/2019-12-24/16-55'),
    ]
    for save_path in save_paths:
        results = analyse_result_scheme(save_path)

        # p_SIC = matrix_significance_difference(results['weight_SICE_ASD'],
        #                                        results['weight_SICE_NC'])

        F_mean = np.mean(results['F'], axis=-1)
        weights_SIC_ASD_mean = np.mean(results['weight_SICE_ASD'], axis=-1)
        weights_SIC_NC_mean = np.mean(results['weight_SICE_NC'], axis=-1)
        weight_SIC_diff = weights_SIC_ASD_mean - weights_SIC_NC_mean

        order, edges_F, abnormal_conn = matrix_sort(F_mean)
        order_roi, rois, abnormal_roi = vector_sort(np.sum(F_mean + F_mean.T, axis=0))

        top_SICE = 50
        _, edges_SIC_ASD, weights_SIC_ASD_mean = matrix_sort(weights_SIC_ASD_mean, top=top_SICE, if_diagonal=False)
        _, edges_SIC_NC, weights_SIC_NC_mean = matrix_sort(weights_SIC_NC_mean, top=top_SICE, if_diagonal=False)
        _, edges_SIC_diff, weight_SIC_diff = matrix_sort(weight_SIC_diff, top=top_SICE, if_diagonal=False)

        # labels = load_index_roi()
        # edge_names = [['{:d}{:s}'.format(edge[0] + 1, labels[edge[0] + 1].name),
        #                '{:d}{:s}'.format(edge[1] + 1, labels[edge[1] + 1].name)]
        #               for edge in edges_F]

        # edge_names_num = 0
        # edge_index = 1
        # for edge_name, edge in zip(edge_names, edges_F):
        #     if edge_index > edge_names_num:
        #         break
        #     print('Edge {:d}: {:}, value: {:f}'.format(edge_index,
        #                                                edge_name,
        #                                                abnormal_conn[tuple(edge)]
        #                                                ))
        #     edge_index += 1
        # print()
        # roi_names_num = 20
        # show_num = 1
        # index = 1
        # for roi_index in rois:
        #     if show_num > roi_names_num:
        #         break
        #     print('Index: {:d} ROI {:d}: {:} ({:}), value: {:f}'.format(index,
        #                                                                 roi_index + 1,
        #                                                                 labels[roi_index + 1].name,
        #                                                                 labels[roi_index + 1].abrv,
        #                                                                 abnormal_roi[roi_index],
        #                                                                 ))
        #     show_num += 1
        #     index += 1
        # print()

        # schemes = ['BrainNetCNNEW', 'CNNWithGLasso']
        # for scheme in schemes:
        #     if scheme in save_path:
        #         break

        results = {'weight': abnormal_conn + abnormal_conn.T,
                   'weight_SIC_ASD': weights_SIC_ASD_mean,
                   'weight_SIC_NC': weights_SIC_NC_mean,
                   'weight_SIC_diff': weight_SIC_diff,
                   }
        edges = {'weight': edges_F,
                 'weight_SIC_ASD': edges_SIC_ASD,
                 'weight_SIC_NC': edges_SIC_NC, 
                 'weight_SIC_diff': edges_SIC_diff, 
                 }
        # sio.savemat('F:/results_{:s}.mat'.format(scheme), results)

        circos_path = os.path.join(save_path, 'Circos')
        if not os.path.exists(circos_path):
            os.makedirs(circos_path)
        for result in results:
            output_path = os.path.join(circos_path, 'links.{:s}.txt'.format(result))
            get_links(adjacent_matrix=results[result],
                      edges=edges[result],
                      output_path=output_path)

        output_path = os.path.join(save_path, 'Edge')
        if not os.path.exists(output_path):
            os.makedirs(output_path)
        np.savetxt('{:s}/abnormal_edges.edge'.format(output_path), results['weight'])
        # np.savetxt('F:/weight_SIC_ASD_{:s}.edge'.format(scheme), weights_SIC_ASD_mean)
        np.savetxt('{:s}/weight_SIC_NC.edge'.format(output_path), results['weight_SIC_NC'])
        np.savetxt('{:s}/weight_SIC_diff.edge'.format(output_path), results['weight_SIC_diff'])


# p_value, sig = analyse_significance_difference()
# pass

# analyse_weights(dir_path=dir_path)