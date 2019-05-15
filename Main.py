import numpy as np

from Log.log import Log
from Analyse.result import Result
from Structure.architecture import Architecture
from data.utils_prepare_data import basic_path, hdf5_handler


def main():
    start_time = 13
    stop_time = 20

    training = True
    analyse = True
    save = True

    schemes = ['GraphCNN']

    for scheme in schemes:

        # Training
        if training:
            start_fold = 2
            arch = Architecture(scheme=scheme)
            for time in np.arange(start=start_time, stop=stop_time):
                arch.train_folds(start_fold=start_fold,
                                 run_time=time,
                                 show_info=True,
                                 save_result=save,
                                 )
                start_fold = 0

        # Analyse results
        if analyse:
            rt = Result(schemes=[scheme])
            result = rt.analyse_results()[scheme]
            print('Accuracy of scheme {:s}: mean:{:4f}\t max:{:4f}'.format(scheme,
                                                                           np.mean(result),
                                                                           np.max(np.mean(result, axis=1))))


main()
