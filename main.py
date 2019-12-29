import os
import sys
import numpy as np

from os import path
from Model.Framework import Framework

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['CUDA_VISIBLE_DEVICES'] = '1'

dir_path = '/'.join(__file__.split('/')[:-4])


def main(schemes: str or list,
         start_time: int = 1,
         stop_time: int = 20,
         start_fold: int = 1,
         stop_fold: int = 5,
         spe_pas: dict = None,
         ):
    if isinstance(schemes, str):
        schemes = [schemes]

    for scheme in schemes:
        frame = Framework(scheme=scheme,
                          dir_path=dir_path,
                          spe_pas=spe_pas)
        frame.training(start_time=start_time, stop_time=stop_time)


if __name__ == '__main__':
    start_time = 69
    for x in range(7):
        spe_pas = {'SICE_lambda': '{:.2f}'.format(0.14 + 0.01 * x), 
                   'cross_validation': 'Monte Calor'}
        print(spe_pas)
        main(schemes='CNNGLasso', spe_pas=spe_pas, stop_time=100)
        start_time = 1
