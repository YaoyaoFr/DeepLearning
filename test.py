import numpy as np
import tensorflow as tf
from ops.SICE import generate_data
import scipy.io as sio
import matplotlib.pyplot as plt
from data.utils_prepare_data import hdf5_handler, create_dataset_hdf5

path = b'F:/OneDriveOffL/Data/Data/DCAE_results.hdf5'
file = hdf5_handler(path)

scheme_group = file.require_group('scheme GraphCNN')
time_group = scheme_group.require_group('time 4')
fold_group = time_group.require_group('fold 2')
