import h5py
import numpy as np
import tensorflow as tf
from Log.log import Log
from Model.SVM import SupportVectorMachine


class SparseInverseCovarianceSVM(SupportVectorMachine):
    paper = 'Sparse network-based models for patient classification using fMRI'

    def __init__(self,
                 scheme: str,
                 dir_path: str,
                 log: Log = None,
                 graph: tf.Graph = None,
                 spe_pas: dict = None, 
                 ):
        SupportVectorMachine.__init__(self,
                                      scheme=scheme,
                                      dir_path=dir_path)

    def training(self,
                 data: dict,
                 run_time: int = 1,
                 fold_index: int = None,
                 show_info: bool = True,
                 ):
        """
        group scheme
            -group fold         str: ['fold 1', 'fold 2', ...]
                -group  alpha   str: ['0.01', '0.02', ...]
                    -data   train data
                    -data   train label
                    -data   valid data
                    -data   valid label
                    -data   test data
                    -data   test label
        """
        self.build_structure()
        result_types = {'Accuracy', 'Precision', 'Recall','Specificity', 'F1 Score'}
        result_datasets = {'train', 'valid', 'test'}
        results = {result_type: {result_dataset: [] for result_dataset in result_datasets} for 
                        result_type in result_types}
        for alpha in data:
            alpha_group = data[alpha]

            alpha_group = self.load_data(data=alpha_group)
            result_alpha = self.fit(data=alpha_group)
            if show_info:
                self.show_results(results=result_alpha,
                                  run_time=run_time,
                                  alpha=float(alpha),
                                  fold_index=fold_index)

            for result_type in result_types:
                for result_dataset in result_datasets:
                    results[result_type][result_dataset].append(result_alpha[result_dataset][result_type])
        return results

    
    @staticmethod
    def load_dataset(hdf5_file, 
                     scheme: str):
        dataset = {}

        scheme_group = hdf5_file['scheme {:s}'.format(scheme)]
        for fold_index in range(5):
            fold_dataset = {}
            fold_group = scheme_group['fold {:d}'.format(fold_index+1)]
            for alpha in fold_group:
                alpha_group = fold_group[alpha]
                alpha_dataset = {}

                for tag in ['train', 'valid', 'test']:
                    for data_type in ['data', 'label']:
                        str = '{:s} {:s}'.format(tag, data_type)
                        try:
                            alpha_dataset[str] = np.array(alpha_group[str])
                        except KeyError:
                            continue
                fold_dataset[alpha] = alpha_dataset
            dataset['fold {:d}'.format(fold_index + 1)] = fold_dataset
        hdf5_file.close()

        return dataset