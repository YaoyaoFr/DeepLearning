import sys
import time
import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt

from Data.utils_prepare_data import load_nifti_data
def load_data():
    data_path = 'F:/OneDriveOffL/Data/Data/ABIDE/Results/fALFF_FunImgARCW/fALFFMap_0051585.nii'
    data = load_nifti_data(data_path=data_path, mask=True, normalization=True)
    fig = plt.figure()
    plt.imshow(data[:, 37, :])
    plt.show()

load_data()
import tensorflow as tf
from ftplib import FTP
from utils import *
from Visualize.visualize import *
from Data.utils_prepare_data import *
from utils import run_classifier
from Structure.scae_tl import main
from Structure.nn import StackedConvolutionAutoEncoder
from Structure.classfier import svm_classify, cnn_classify
from Data.load_data import *

from Data.prepare_data import *


def load_datas():
    hdf5 = load_datas()


from nilearn.datasets import fetch_atlas_harvard_oxford
from nilearn.plotting import plot_glass_brain
from nibabel import load
import matplotlib.pyplot as plt


def download_harvard_oxford_atlas():
    atlas_dir = 'F:/OneDriveOffL/Program/Python/'
    atlas = fetch_atlas_harvard_oxford(atlas_name='cort-maxprob-thr25-2mm',
                                       data_dir=atlas_dir)
    volume = load(atlas['maps'])
    plot_glass_brain(volume)
    plt.show()
