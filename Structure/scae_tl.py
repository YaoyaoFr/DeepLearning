from Log.log import Log
from DeepNerualNetwork.DeepNeuralNetwork import DeepNeuralNetwork
from NeuroimageDataProcessing.utils import *


class Architecture:
    scae = None
    classifier = None
    hdf5 = None
    scheme = None

    def __init__(self, scheme: int = 5, hdf5: h5py.Group = None):
        self.log = Log()
        # self.scae = StackedConvolutionAutoEncoder(log=self.log, scheme=scheme)
        self.classifier = DeepNeuralNetwork(log=self.log, scheme=scheme)

        if hdf5 is None:
            hdf5_path = b'F:/OneDriveOffL/Data/Data/DCAE_scheme.hdf5'
            self.hdf5 = hdf5_handler(hdf5_path, 'a')

    def train_fold(self, fold: h5py.Group,
                   start_index: str = None,
                   ):

        # process_list = ['pre_train_SCAE', 'fine_tune_SCAE', 'pre_train_Classifier', 'fine_tune_Classifier']
        process_list = ['pre_train_Classifier', 'fine_tune_Classifier']
        for process_index in np.arange(start=1, stop=len(process_list)):
            process_tag = process_list[process_index]
            print('Start process {:s}'.format(process_tag.replace('_', ' ')))
            if process_tag == 'pre_train_SCAE':
                self.scae.train_fold(fold=fold, start_index=start_index),
            elif process_tag == 'fine_tune_SCAE':
                self.scae.build_structure()
                self.scae.fine_tune_fold(fold=fold)
                self.scae.encode_fold(fold=fold)
            elif process_tag == 'pre_train_Classifier':
                self.classifier.train_fold(fold=fold)
            elif process_tag == 'fine_tune_Classifier':
                self.scae.build_structure()
                self.classifier.fine_tune_fold(fold=fold,
                                               input_tensor=self.scae.structure['encoder_tensor'],
                                               input_place=self.scae.structure['input_place']
                                               ),

            restored_path = None

    def train_folds(self,
                    folds: h5py.Group = None,
                    scheme: int = 5,
                    end_fold: int = None,
                    ):
        if folds is None:
            folds = self.hdf5['scheme {:d}/ABIDE/falff'.format(scheme)]
        if end_fold is None:
            end_fold = 6

        restored_pas = self.log.get_restored_pa()
        self.scae.initialization()
        for fold_index in np.arange(start=restored_pas['fold'], stop=end_fold):
            self.classifier.initialization()
            fold = folds['fold {:d}'.format(fold_index)]
            self.train_fold(fold=fold,
                            process_index=restored_pas['process'],
                            start_index=restored_pas['indexes']
                            )
            restored_pas.update({
                'process': 0,
                'indexes': None
            })


def main():
    arch = Architecture()
    arch.train_folds()


if __name__ == '__main__':
    main()
