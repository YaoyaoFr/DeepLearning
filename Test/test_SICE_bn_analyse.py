import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt

path = 'F:/Results_time_{:d}_fold_{:d}.mat'
for fold_index in range(5):
    run_times = 0
    F_stable = np.zeros(shape=[90, 90])
    for run_time in range(10):
        file_path = path.format(run_time + 1, fold_index + 1)
        try:
            datas = sio.loadmat(file_path)
        except FileNotFoundError:
            continue
        F = datas['F']
        F_stable += F
        run_times += 1
        weight = np.abs(datas['weight_SICE_bn'])

        # threshold = 0.3

        weight_ASD = np.mean(weight[..., 0:8], axis=-1)
        weight_TC = np.mean(weight[..., 8:16], axis=-1)

        # weight_ASD[weight_ASD < threshold] = 0
        # weight_TC[weight_TC < threshold] = 0

        datas.update({'weight_SICE_ASD': weight_ASD,
                      'weight_SICE_TC': weight_TC})
        sio.savemat(file_path, datas)
    F_stable /= run_times
    F_stable = np.tril(F_stable, k=-1)
    row, col = np.shape(F_stable)
    F_tmp = np.zeros(shape=[row, col])
    top = 25
    edges = np.zeros(shape=[top, 2])
    for i in range(top):
        index = np.argmax(F_stable)
        m, n = divmod(index, col)
        F_tmp[m, n] = F_stable[m, n]
        F_tmp[n, m] = F_stable[m, n]
        F_stable[m, n] = 0
        edges[i, 0] = m
        edges[i, 1] = n
    sio.savemat('F:/F_stable_fold_{:d}.mat'.format(fold_index + 1),
                {'F_stable': F_tmp,
                 'Time': run_times,
                 'Top': top,
                 'edges': edges + 1})
