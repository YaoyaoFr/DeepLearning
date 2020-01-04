import os
import time
import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt

from Log.log import Log
from Model.Framework import Framework
from Dataset.utils import vector2onehot
from Dataset.utils import hdf5_handler


def save_weight_as_fig(weight):
    net_path = 'F:/OneDriveOffL/Data/Result/Net/fold 1/train'
    out_channels = np.shape(weight)[2]
    max = np.max(weight)
    min = np.min(weight)
    fig = plt.figure()
    for i in range(out_channels):
        w = np.abs(weight[..., i])
        w[np.eye(90) == 1] = 0
        plt.imshow(w)
        plt.colorbar()
        plt.savefig(os.path.join(net_path, 'CNNWithGlasso_weight_{:d}.jpg'.format(i + 1)))
        fig.clear()


net_path = 'F:/OneDriveOffL/Data/Result/Net/fold 1/train'

if True:
    version = 1
    if version == 1:
        data_file = hdf5_handler(b'F:/OneDriveOffL/Data/Data/DCAE_scheme.hdf5')
        data = data_file['scheme CNNWithGLasso/ABIDE/pearson correlation/fold 1']
        train_data = np.array(data['train data'])
        train_covariance = np.array(data['train covariance'])
        train_label = vector2onehot(np.array(data['train label']))
    elif version == 2:
        data = sio.loadmat('F:/OneDriveOffL/Data/Data/BrainNetCNN/ALLASD3_NETFC_SG_Pear.mat')
        train_data = np.expand_dims(data['net_train'], axis=-1)
        train_label = vector2onehot(data['phenotype_train'][:, 2])

    subjects = np.arange(200)
    train_data = train_data[subjects]
    train_label = train_label[subjects]
    train_covariance = train_covariance[subjects]

    # scheme = 'BrainNetCNN' 0.657
    # scheme = 'BrainNetCNNEW'
    scheme = 'CNNWithGLasso2'
    log = Log(restored_date='ParametersTuning/{:s}'.format(time.strftime('%Y-%m-%d', time.localtime(time.time()))),
              )
    # Training
    frame = Framework(scheme=scheme, log=log)

    # Rebuild the structure and log
    classifier = frame.structure['CNNWithGLasso'](scheme=frame.scheme, log=frame.log)
    with log.graph.as_default():
        classifier.build_structure()

    obj_values = []
    softmax_values = []
    for epoch in range(500):
        results_SICE = classifier.backpropagation_epoch(data=train_data,
                                                        label=train_label,
                                                        learning_rate=frame.train_pa['learning_rate'],
                                                        train_pa=frame.train_pa,
                                                        training=True,
                                                        save_info=False,
                                                        show_info=False,
                                                        minimizer=classifier.optimizer['minimizer_SICE'],
                                                        epoch=epoch + 1,
                                                        feed_dict_extra={classifier.input_placeholders[
                                                                             'covariance']: train_covariance},
                                                        )
        if True:
            print('Epoch {:4d}\t'
                  'SICE Penalty: {:4f}\t'
                  'L1 Penalty: {:4f}\t'
                  'Objective function: {:4f}\t'.
                  format(epoch + 1,
                         results_SICE['SICE Penalty'],
                         results_SICE['L1 Penalty'],
                         results_SICE['SICE Penalty'] + results_SICE['L1 Penalty'],
                         ))

        if (epoch + 1) % 1 == 0:
            tensors = classifier.sess.run(fetches=classifier.structure['tensors']['E2NGLasso1'],
                                          feed_dict={
                                              classifier.input_placeholders['input']: train_data,
                                              classifier.input_placeholders['output']: train_label,
                                              classifier.input_placeholders['training']: True,
                                              classifier.input_placeholders['covariance']: train_covariance,
                                          })

            weight = np.squeeze(tensors['weight'])
            obj_value = tensors['SICE regularizer']
            softmax_value = tensors['Regularizer softmax']
            obj_values.append(np.expand_dims(obj_value, axis=-1))
            softmax_values.append(np.expand_dims(softmax_value, axis=-1))
    obj_values = np.concatenate(obj_values, axis=-1)
    softmax_values = np.concatenate(softmax_values, axis=-1)

    out_channels = np.shape(weight)[2]
    obj_values_1 = obj_values[0, :int(out_channels / 2), :]
    obj_values_2 = obj_values[0, int(out_channels / 2):, :]
    fig = plt.figure()
    plt.plot(obj_values_1.T)
    plt.legend([str(i) for i in np.arange(int(out_channels / 2))])
    plt.show()
    fig = plt.figure()
    plt.plot(obj_values_2.T)
    plt.legend([str(i) for i in np.arange(int=int(out_channels / 2), stop=out_channels)])
    plt.show()

    softmax_values_1 = softmax_values[0, :int(out_channels / 2), :]
    softmax_values_2 = softmax_values[0, int(out_channels / 2):, :]
    fig = plt.figure()
    plt.plot(softmax_values_1.T)
    plt.legend([str(i) for i in np.arange(int(out_channels / 2))])
    plt.show()
    fig = plt.figure()
    plt.plot(softmax_values_2.T)
    plt.legend([str(i) for i in np.arange(int=int(out_channels / 2), stop=out_channels)])
    plt.show()

    print(obj_value)
    print(np.argmin(obj_value, axis=-1))

    max = np.max(weight)
    min = np.min(weight)
    fig = plt.figure()
    for i in range(out_channels):
        w = np.abs(weight[..., i])
        w[np.eye(90) == 1] = 0
        plt.imshow(w)
        plt.colorbar()
        plt.savefig(os.path.join(net_path, 'CNNWithGlasso_weight_{:d}.jpg'.format(i + 1)))
        fig.clear()

Ps = sio.loadmat(os.path.join(net_path, 'GMGLASS_SIC.mat'))['Ps']
SICE_subjects = [1, 2, 9, 10]
for i in range(90):
    for k in range(len(SICE_subjects)):
        Ps[i, i, k] = 0

max = np.max(Ps)
min = np.min(Ps)
fig = plt.figure()
for index, sub_ID in enumerate(SICE_subjects):
    w = np.abs(Ps[..., index])
    plt.imshow(w, vmin=min, vmax=max)
    plt.colorbar()
    plt.savefig(os.path.join(net_path, 'GMGLASS_SIC_{:d}.jpg'.format(sub_ID)))
    fig.clear()
