from Structure.nn import *
from Structure.classfier import *
from Data.utils_prepare_data import *


class Architecture:
    scae = None
    classifier = None
    hdf5 = None
    scheme = None

    def __init__(self, scheme: int = 4, hdf5: h5py.Group = None):
        if hdf5 is None:
            hdf5_path = b'F:/OneDriveOffL/Data/Data/DCAE_scheme.hdf5'
            self.hdf5 = hdf5_handler(hdf5_path, 'a')

        log = Log()
        self.scae = StackedConvolutionAutoEncoder(log=log, scheme=scheme)

        # restored_path = 'F:/OneDriveOffL/Data/Result/DCAE/2018-09-18/22-48/fold 1/fine_tuning/model/train.model_300'
        # log.restore(restored_path=restored_path)

        self.classifier = DeepNeuralNetwork(log=log, scheme=scheme)

    def train_fold(self, fold: h5py.Group,
                   restored_path: str = None,
                   start_process: str = 'fine_tune_SCAE',
                   ):

        process_list = ['fine_tune_SCAE', 'pre_train_Classifier', 'fine_tune_Classifier']
        if start_process is not None:
            start_process = process_list.index(start_process)
        else:
            start_process = 0
        for process_index in np.arange(start=start_process, stop=len(process_list)):
            process_tag = process_list[process_index]
            print('Start process {:s}'.format(process_tag.replace('_', ' ')))
            if process_tag == 'pre_train_Classifier':
                self.classifier.pre_train_fold(fold=fold, restored_path=restored_path)
            elif process_tag == 'fine_tune_Classifier':
                self.scae.build_structure()
                self.classifier.fine_tune_fold(fold=fold,
                                               restored_path=restored_path,
                                               input_tensor=self.scae.structure['encoder_tensor'],
                                               input_place=self.scae.structure['input_place']
                                               ),
            elif process_tag == 'fine_tune_SCAE':
                self.scae.build_structure()
                self.scae.fine_tune_fold(fold=fold, restored_path=restored_path)
                self.scae.encode_fold(fold=fold)

            restored_path = None

    def train_folds(self,
                    folds: h5py.Group = None,
                    restored_path: str = None,
                    start_fold: int = 0,
                    end_fold: int = None,
                    start_process=None,
                    start_train_index=None,
                    ):
        if folds is None:
            folds = self.hdf5['scheme 4/falff']
        if end_fold is None:
            end_fold = 6

        # restored_path = 'F:/OneDriveOffL/Data/Result/DCAE/2018-09-22/23-18/fold 1/pre_train_Classifier/' \
        #                 'model/train.model_200'

        # Pre train

        self.scae.initialization()
        restored_path = 'F:/OneDriveOffL/Data/Result/DCAE/2018-10-02/21-50/fold 1/fine_tune_SCAE\model/train.model_20'
        # self.scae.pre_train_fold(fold=folds, restore_path=restored_path, start_index=start_train_index),
        for fold_index in np.arange(start=start_fold, stop=end_fold):
            self.classifier.initialization()
            fold = folds['fold {:d}'.format(fold_index)]
            self.train_fold(fold=fold,
                            restored_path=restored_path,
                            start_process=start_process,
                            )
            start_process = None
            # restored_path = None


def main():
    arch = Architecture()
    arch.train_folds(start_fold=1, start_process='pre_train_Classifier')


if __name__ == '__main__':
    main()
