import os
import time
import numpy as np
from Log.log import Log
from Model.Framework import Framework
from Dataset.utils import vector2onehot

os.chdir('..')
scheme = 'CNNWithGLasso'
log = Log(restored_date='ParametersTuning/{:s}'.format(time.strftime('%Y-%m-%d', time.localtime(time.time()))),
          )
arch = Framework(scheme=scheme, log=log)

fold = arch.hdf5['scheme CNNWithGLasso/ABIDE/pearson correlation/fold 3']

data = {'train data': np.array(fold['train data']),
        'train covariance': np.array(fold['train covariance']),
        'train label': vector2onehot(np.array(fold['train label'])),
        'valid data': np.array(fold['valid data']),
        'valid covariance': np.array(fold['valid covariance']),
        'valid label': vector2onehot(np.array(fold['valid label'])),
        'test data': np.array(fold['test data']),
        'test covariance': np.array(fold['test covariance']),
        'test label': vector2onehot(np.array(fold['test label'])),
        }
arch.model = arch.structure[scheme](scheme=arch.scheme, log=arch.log)
classifier = arch.model
train_pa = classifier.train_pa

for lambda_ in [5]:
    classifier.basic_pa['lambda'] = lambda_
    lambda_str = str(lambda_).replace('.', '')
    with classifier.graph.as_default():
        classifier.build_structure()

    data_supervise = np.concatenate((data['train data'],
                                     # data['valid data'],
                                     # data['test data'],
                                     ),
                                    axis=0)
    extra_data_supervise = np.concatenate((data['train covariance'], ),
                                          axis=0)
    label_supervise = np.concatenate((data['train label'],
                                      # data['valid label'],
                                      # data['test label']
                                      ),
                                     axis=0)
    extra_feed_supervise = {classifier.input_placeholders['sample_covariance']: extra_data_supervise}

    data_unsupervise = np.concatenate((data['valid data'],
                                       data['test data']),
                                      axis=0)
    label_unsupervise = np.concatenate((data['valid label'],
                                        data['test label']),
                                       axis=0)
    extra_data_unsupervise = np.concatenate((data['valid covariance'], data['test covariance']), axis=0)
    extra_feed_unsupervise = {classifier.input_placeholders['sample_covariance']: extra_data_unsupervise}

    for i in range(1000):
        # print('Epoch {:d}'.format(i + 1))
        if (i + 1) % 20 == 0:
            classifier.save_GLasso_weights_to_figure(if_save=True)

        classifier.backpropagation_epoch(
            data=data_supervise,
            label=label_supervise,
            learning_rate=0.1,
            train_pa=train_pa,
            training=True,
            save_info=False,
            show_info=False,
            get_tensors=False,
            minimizer=classifier.optimizer['minimizer_SICE'],
            epoch=i + 1,
            feed_dict_extra=extra_feed_supervise,
        )
        classifier.backpropagation_epoch(
            data=data_unsupervise,
            label=label_unsupervise,
            learning_rate=0.01,
            train_pa=train_pa,
            training=False,
            save_info=False,
            show_info=False,
            get_tensors=False,
            minimizer=classifier.optimizer['minimizer_SICE'],
            epoch=i + 1,
            feed_dict_extra=extra_feed_unsupervise,
        )
