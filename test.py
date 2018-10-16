import sys
import time
import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt

from Data.utils_prepare_data import hdf5_handler
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