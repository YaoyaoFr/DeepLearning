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
         spe_pas: dict = None,
         ):
    if isinstance(schemes, str):
        schemes = [schemes]
    print(dir_path)
    for scheme in schemes:
        frame = Framework(scheme=scheme,
                          dir_path=dir_path,
                          spe_pas=spe_pas,
                          )
        frame.training(start_time=start_time, 
                       stop_time=stop_time, 
                       )


if __name__ == '__main__':
    main(schemes='CNNGLasso')
