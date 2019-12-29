import time
import numpy as np

from Log.log import Log
from Model.Framework import Framework


def main():
    # scheme = 'BrainNetCNN' # 0.657
    # scheme = 'BrainNetCNNEW'  # 0.670
    # scheme = 'CNNGLasso'
    scheme = 'CNNSmallWorld'
    log = Log(restored_date='{:s}/{:s}'.format(scheme, time.strftime('%Y-%m-%d', time.localtime(time.time()))),
              )
    # Training
    frame = Framework(scheme=scheme, log=log)

    # Rebuild the structure and log
    run_times = 10
    start_fold = 1
    end_fold = 5
    accuracy_times = []
    for run_time in np.arange(start=1, stop=run_times + 1):
        accuracy_folds = []
        for fold_index in np.arange(start=start_fold, stop=end_fold + 1):
            frame.model = frame.structure[frame.basic_pa['scheme']](scheme=scheme, log=frame.log)

            fold = frame.hdf5['fold {:d}'.format(fold_index)]
            # Training
            results, max_epoch = frame.model.training(data=fold,
                                                      fold_index=fold_index,
                                                      run_time=run_time,
                                                      # restored_path=restored_path,
                                                      show_info=True)
            print('Time: {:d}\tFold: {:d}\tMax Accuracy is in epoch {:d}, train: {:f}\tvalid: {:f}\ttest: {:f}'.format(
                run_time,
                fold_index,
                max_epoch,
                results['Accuracy']['train'][max_epoch],
                results['Accuracy']['valid'][max_epoch],
                results['Accuracy']['test'][max_epoch],
            ))
            accuracy_folds.append(results['Accuracy']['test'][max_epoch])

        accuracy_folds = np.array(accuracy_folds)
        print('Time: {:d}\tAccuracies: {:}'.format(run_time, accuracy_folds))
        accuracy_times.append(accuracy_folds)
        accuracy_times_temp = np.array(accuracy_times)
        print('Accuracy: {:}\r\nMean: {:f}'.format(accuracy_times_temp, np.mean(accuracy_times_temp)))


main()
