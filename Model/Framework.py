import re
import os
import h5py
import numpy as np

from Analyse.Result import Result
from Dataset.SchemeData import SchemeData
from Dataset.utils import hdf5_handler
from Log.log import Log
from Model.CNNGLasso import CNNGraphicalLasso
from Model.CNNSM import CNNSmallWorld
from Model.DTLNN import DeepTransferLearningNN
from Model.NN import NeuralNetwork
from Model.SAE import StackedAutoEncoders
from Model.SICSVM import SparseInverseCovarianceSVM
from Model.SVM import SupportVectorMachine
from Model.utils_model import save_weights
from Model.FCNN import FullyConnectedNeuralNetwork
from Schemes.xml_parse import parse_xml_file, parse_str


class Framework:
    scae = None
    model = None
    scheme = None
    spe_pas = None

    def __init__(self,
                 dir_path: str = 'F:/',
                 scheme: int or str = 'BrainNetCNN',
                 spe_pas: dict = None,
                 ):
        self.dir_path = dir_path
        self.project_path = os.path.join(dir_path, 'Program/Python/DeepLearning')
        self.result = Result(dir_path=dir_path)
        self.dataset_file_path = os.path.join(dir_path, 'Data/SchemeData.hdf5').encode()
        self.scheme = scheme
        self.spe_pas = spe_pas
        self.log = Log(scheme_folder=self.scheme, dir_path=dir_path)
        self.models = {
            'CNNGLasso': CNNGraphicalLasso,
            'BrainNetCNN': NeuralNetwork,
            'CNNElementWise': NeuralNetwork,
            'CNNSmallWorld': CNNSmallWorld,
            'DeepNeuralNetwork': NeuralNetwork,
            'DenoisedAutoEncoder': StackedAutoEncoders,
            'SICSVM': SparseInverseCovarianceSVM,
            'SVM': SupportVectorMachine,
            'DTLNN': DeepTransferLearningNN,
            'FCNN': FullyConnectedNeuralNetwork, 
        }
        self.update_parameters(spe_pas=spe_pas)


    def training(self,
                 start_time: int = 1,
                 stop_time: int = 100,
                 if_save: bool = True,
                 cross_validation: str = None):
        if cross_validation:
            self.cross_validation = cross_validation

        # Preparing dataset
        dataset = None
        if self.cross_validation == '5 fold':
            hdf5_file = hdf5_handler(self.dataset_file_path)
            dataset = self.models[self.scheme].load_dataset(hdf5_file=hdf5_file, scheme=self.scheme)

        # Training
        for run_time in np.arange(start=start_time, stop=stop_time + 1):
            print('Run times: {:d}...'.format(run_time))

            if self.cross_validation == 'Monte Calor':
                self.train_monte_calor(run_time=run_time,
                                       show_info=True,
                                       if_save=if_save,
                                       )
            elif self.cross_validation == '5 fold':
                self.train_folds(dataset=dataset, 
                                 run_time=run_time,
                                 show_info=True,
                                 if_save=if_save,
                                 )

        print('Training finished, the results are saved in {:s}. '.format(self.save_scheme_name))

    def train_monte_calor(self,
                          run_time: int = None,
                          show_info: bool = True,
                          if_save: bool = False,
                          ):
        # Preparing dataset
        sd = SchemeData(dir_path=self.dir_path)
        basic_pa = parse_str(self.current_xml)['parameters']['basic']
        data = sd.monte_calor_cross_validation(run_time=run_time,
                                               normalization=basic_pa['normalization'],
                                               dataset=basic_pa['dataset'],
                                               atlas=basic_pa['atlas'],
                                               feature=basic_pa['feature'])
        del sd

        # Preparing log director
        self.log.set_path(subfolder='time {:d}'.format(run_time))
        # Preparing tensorflow graph
        self.log.reset_graph()

        # Rebuild the structure and log
        self.model = self.models[self.scheme](scheme=self.scheme,
                                              log=self.log,
                                              dir_path=self.dir_path, 
                                              spe_pas=self.spe_pas)

        # Training
        results = self.model.training(data=data,
                                      run_time=run_time,
                                      show_info=show_info)
        # save_weights(self.model.op_layers, self.model.sess, save_dir=self.log.dir_path)

        if if_save:
            self.result.save_results(save_scheme_name=self.save_scheme_name,
                                     cross_validation=self.cross_validation,
                                     current_xml=self.current_xml,
                                     results=results,
                                     run_time=run_time)

    def train_folds(self,
                    dataset: dict, 
                    run_time: int = None,
                    start_fold: int = 1,
                    end_fold: int = 5,
                    fold_list: list = None,
                    if_save: bool = False,
                    show_info: bool = True,
                    ):
        if fold_list is None:
            fold_list = np.arange(start_fold, stop=end_fold + 1)

        for fold_index in fold_list:
            # Preparing dataset
            fold_group = dataset['fold {:d}'.format(fold_index)]

            # Preparing log director
            self.log.set_path(
                subfolder='time {:d}/fold {:d}'.format(run_time, fold_index))
            # Preparing tensorflow graph
            self.log.reset_graph()

            # Rebuild the structure and log
            self.model = self.models[self.scheme](scheme=self.scheme,
                                                  log=self.log,
                                                  dir_path=self.dir_path, 
                                                  spe_pas=self.spe_pas,
                                                  )

            results = self.model.training(data=fold_group,
                                          run_time=run_time,
                                          fold_index=fold_index,
                                          show_info=show_info)

            if if_save:
                self.result.save_results(save_scheme_name=self.save_scheme_name,
                                         cross_validation=self.cross_validation,
                                         current_xml=self.current_xml,
                                         results=results,
                                         run_time=run_time,
                                         fold_index=fold_index)

    def train_fold(self,
                   run_time: int,
                   fold_index: int,
                   folds: h5py.Group = None,
                   show_info: bool = True,
                   ):
        hdf5_file = hdf5_handler(self.dataset_file_path)
        dataset = self.models[self.scheme].load_dataset(hdf5_file=hdf5_file, scheme=self.scheme)
        self.log.set_path(
            subfolder='scheme {:s}/time {:d}/fold {:d}'.format(self.scheme,
                                                               run_time,
                                                               fold_index))

        # Get data fold
        fold = folds['fold {:d}'.format(fold_index)]

        # Rebuild the structure and log
        self.model = self.models[self.scheme](scheme=self.scheme, log=self.log)

        # Training
        results = self.model.training(data=fold,
                                      fold_index=fold_index,
                                      run_time=run_time,
                                      show_info=show_info)
        save_weights(self.model.op_layers, self.model.sess,
                     save_dir=self.log.dir_path)
        return results

    def update_parameters(self, 
                          spe_pas: dict = None):
        scheme_file_path = os.path.join(self.project_path, 'Schemes/{:s}.xml'.format(self.scheme))
        xml_str = open(scheme_file_path).read()

        if spe_pas is not None:
            for spe_pa in spe_pas:
                reg_str = '<{:s}>.*</{:s}>'.format(spe_pa, spe_pa)
                match_obj = re.search(pattern=reg_str, string=xml_str)
                new_str = '<{:s}>{:}</{:s}>'.format(spe_pa, spe_pas[spe_pa], spe_pa)
                xml_str = xml_str.replace(match_obj.group(), new_str)
            self.spe_pas = parse_str(xml_str)

        pas = parse_str(xml_str)
        self.cross_validation = pas['parameters']['basic']['cross_validation']

        # Saving group and parameters
        self.save_scheme_name, self.current_xml = self.result.set_saving_group(self.scheme, 
                                                                               xml_str, 
                                                                               self.cross_validation)
