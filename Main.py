import sys
import time
import numpy as np
import tensorflow as tf
from utils import run_classifier
import matplotlib.pyplot as plt
from Structure.multiview_cnn import main
from Structure.nn import StackedConvolutionAutoEncoder, DeepNeuralNetwork
from Structure.classfier import SupportVectorMachine, svm_classify
from utils import onehot_to_vector
from Data.utils_prepare_data import load_nifti_data, hdf5_handler, repeatmap, data_normalization

main()