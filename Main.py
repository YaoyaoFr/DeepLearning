import numpy as np

from Log.log import Log
from Structure.Framework import Framework
from NeuroimageDataProcessing.utils import basic_path, hdf5_handler


def main():
    start_time = 1
    stop_time = 10
    save = True

    log = Log()
    scheme = 'CNNSmallWorld'

    frame = Framework(scheme=scheme, log=log)
    # Training
    start_fold = 1
    end_fold = 5
    for time in np.arange(start=start_time, stop=stop_time + 1):
        frame.train_folds(start_fold=start_fold,
                          end_fold=end_fold,
                          run_time=time,
                          show_info=True,
                          save_result=save,
                          )
        start_fold = 1


def rerun():
    save = True

    log = Log()
    scheme = 'CNNWithGLasso'
    frame = Framework(scheme=scheme, log=log)
    train_folds = [[2, 3]]

    for train_fold in train_folds:
        frame.train_folds(run_time=train_fold[0],
                          start_fold=train_fold[1],
                          end_fold=train_fold[1],
                          save_result=save)


main()
# rerun()
