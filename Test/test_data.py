import numpy as np
from Dataset.utils import hdf5_handler

hdf5_path = b'F:/OneDriveOffL/Data/Data/DCAE_scheme.hdf5'
hdf5_file = hdf5_handler(hdf5_path)

data_BrainNetCNN = np.array(hdf5_file['scheme BrainNetCNN/ABIDE/pearson correlation/fold 1/train data'])
data_CNNWithGLasso = np.array(hdf5_file['scheme CNNWithGLasso/ABIDE/pearson correlation/fold 1/train data'])

data_size = np.size(data_BrainNetCNN, axis=0)
for i in range(data_size):
    data_temp1 = data_BrainNetCNN[i]
    data_temp2 = data_CNNWithGLasso[i]
    pass
pass


