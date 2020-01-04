"""
    Framework for running a model
"""

import os
import re

import h5py
import numpy as np

from Analyse.Result import Result
from Dataset.SchemeData import SchemeData
from Dataset.utils import hdf5_handler
from Log.log import Log
from Model.CNNGLasso import CNNGraphicalLasso
from Model.CNNSM import CNNSmallWorld
from Model.DTLNN import DeepTransferLearningNN
from Model.FCNN import FullyConnectedNeuralNetwork
from Model.NN import NeuralNetwork
from Model.SAE import StackedAutoEncoders
from Model.SICSVM import SparseInverseCovarianceSVM
from Model.SVM import SupportVectorMachine
from Model.utils_model import save_weights
from Schemes.xml_parse import parse_str


class Framework:
    scae = None
    model = None
    scheme = None
    spe_pas = None
    cross_validation = None

    def __init__(self,
                 dir_path: str = '/home/ai/data/yaoyao',
                 scheme: int or str = 'BrainNetCNN',
                 spe_pas: dict = None,
                 ):
        self.dir_path = dir_path
        self.project_path = os.path.join(
            dir_path, 'Program/Python/DeepLearning')
        self.result = Result(dir_path=dir_path)
        self.dataset_file_path = os.path.join(
            dir_path, 'Data/SchemeData.hdf5').encode()
        self.scheme = scheme
        self.spe_pas = spe_pas
        # Here should keep the date of log and save_scheme be the same.
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
        configurations = self.get_configurations(spe_pas=spe_pas)
        self.set_parameters(configurations=configurations)

    def training(self,
                 start_time: int = 1,
                 stop_time: int = 100,
                 if_save: bool = True):
        """Training the model for many times with different cross validation strategies.

        Keyword Arguments:
            start_time {int} --  (default: {1})
            stop_time {int} --  (default: {100})
            if_save {bool} --  (default: {True})
            cross_validation {str} -- [cross validation strategy: '5 fold' or 'Monte Calor'] 
                (default: {None})
        """
        # Training
        for run_time in np.arange(start=start_time, stop=stop_time + 1):
            print('Run times: {:d}...'.format(run_time))

            if self.cross_validation == 'Monte Calor':
                self.train_monte_calor(run_time=run_time,
                                       if_show=True,
                                       if_save=if_save,
                                       )
            elif self.cross_validation == '5 fold':
                self.train_folds(run_time=run_time,
                                 if_show=True,
                                 if_save=if_save,
                                 )

        print('Training finished, the results are saved in {:s} of file {:s}. '.format(
            self.save_scheme_name, self.result.result_file_path))

    def train_monte_calor(self,
                          run_time: int = None,
                          if_show: bool = True,
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
                                      if_show=if_show)
        # save_weights(self.model.op_layers, self.model.sess, save_dir=self.log.dir_path)

        if if_save:
            self.result.save_results(save_scheme_name=self.save_scheme_name,
                                     cross_validation=self.cross_validation,
                                     current_xml=self.current_xml,
                                     results=results,
                                     run_time=run_time)

    def train_folds(self,
                    run_time: int = None,
                    if_save: bool = False,
                    if_show: bool = True,
                    ):
        """Train model fold by fold. This part is mainly focus on load dataset from hdf5 file.
        
        Keyword Arguments:
            run_time {int} -- [description] (default: {None})
            if_save {bool} -- [description] (default: {False})
            show_info {bool} -- [description] (default: {True})
        """
        #Preparing dataset according to scheme
        dataset = self.models[self.scheme].load_dataset(
            hdf5_file_path=self.dataset_file_path, scheme=self.scheme)

        for fold_name, fold_data in dataset.items():
            self.train_fold(run_time=run_time,
                            fold_name=fold_name,
                            fold_data=fold_data,
                            if_save=if_save,
                            if_show=if_show)

    def train_fold(self,
                   run_time: int,
                   fold_name: str,
                   fold_data: dict = None,
                   if_show: bool = True,
                   if_save: bool = True,
                   ):
        """Train fold 

        Arguments:
            run_time {int} -- Current run time
            fold_data {dict} -- The fold dataset for training
            fold_name {str} -- String for fold, 'fold 1', 'fold 2', ...

        Keyword Arguments:
            if_show {bool} -- Whether show the results at the end of training (default: {True})
            if_save {bool} -- Whether save the results to hdf5 file

        Returns:
            [type] -- [description]
        """
        # Preparing log director
        self.log.set_path(
            subfolder='time {:d}/{:s}'.format(run_time, fold_name))
        # Preparing tensorflow graph
        self.log.reset_graph()

        #Preparing dataset according to scheme
        if fold_data is None:
            fold_data = self.models[self.scheme].load_dataset(
                hdf5_file_path=self.dataset_file_path, scheme=self.scheme)[fold_name]

        # Rebuild the structure and log
        self.model = self.models[self.scheme](scheme=self.scheme,
                                              log=self.log,
                                              dir_path=self.dir_path,
                                              spe_pas=self.spe_pas,
                                              )

        results = self.model.training(data=fold_data,
                                      run_time=run_time,
                                      fold_name=fold_name,
                                      show_info=if_show)

        if if_save:
            self.result.save_results(save_scheme_name=self.save_scheme_name,
                                     cross_validation=self.cross_validation,
                                     current_xml=self.current_xml,
                                     results=results,
                                     run_time=run_time,
                                     fold_name=fold_name)
        return results

    def get_configurations(self,
                           spe_pas: dict = None):
        scheme_file_path = os.path.join(self.project_path,
                                        'Schemes/{:s}.xml'.format(self.scheme))
        configurations = open(scheme_file_path).read()

        if spe_pas is not None:
            for spe_pa in spe_pas:
                reg_str = '<{:s}>.*</{:s}>'.format(spe_pa, spe_pa)
                match_obj = re.search(pattern=reg_str, string=configurations)
                new_str = '<{:s}>{:}</{:s}>'.format(
                    spe_pa, spe_pas[spe_pa], spe_pa)
                configurations = configurations.replace(
                    match_obj.group(), new_str)

        return configurations

    def set_parameters(self,
                       configurations: str):
        """Setting parameters by string.

        Arguments:
            configurations {str} -- [String with xml format]

        Raises:
            Warning: [Whether the cross validation strategy in the formatting xml files]
        """
        pas = parse_str(configurations)
        try:
            self.cross_validation = pas['parameters']['basic']['cross_validation']
        except KeyError:
            self.cross_validation = None
            raise Warning(
                'The cross validation should be given in configurations. ')

        # Saving group and parameters
        self.save_scheme_name, self.current_xml = self.result.set_saving_group(
            self.scheme,
            configurations,
            self.cross_validation)
    # To be continue

    def evalution_trained_models(self,
                                 exp_date: str,
                                 exp_clock: str,
                                 if_results_reset: bool = True,
                                 start_time: int = 1,
                                 stop_time: int = 10,
                                 if_save: bool = False,
                                 ):
        log = Log(dir_path=self.dir_path, scheme_folder=self.scheme)
        log.set_path(date=exp_date, clock=exp_clock)
        self.log = log

        # Preparing dataset
        sd = SchemeData(dir_path=self.dir_path)
        gender_hdf5 = hdf5_handler(sd.hdf5_path)[
            'scheme gender {:s}'.format(self.scheme)]

        for gender in ['male', 'femal']:
            gender_group = gender_hdf5.require_group(gender)
            dataset = self.models[self.scheme].load_dataset(
                scheme_group=gender_group)
            # Evaluation
            for run_time in np.arange(start=start_time, stop=stop_time + 1):
                print('Run times: {:d}...'.format(run_time))

                for fold_index in np.arange(5) + 1:
                    # Preparing dataset
                    data = dataset['fold {:d}'.format(fold_index)]

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
                    with self.model.log.graph.as_default():
                        self.model.build_structure()
                    self.model.log.restore(
                        restored_dir='optimal_model', restored_epoch='final')

                    result_fold = self.model.predicting(data=data, epoch=0)
                    print(result_fold)
