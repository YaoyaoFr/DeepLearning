import os
import time

import numpy
import xlwt
import h5py
import numpy as np

from scipy import stats
from Schemes.xml_parse import parse_xml_file, parse_str
from Dataset.utils import hdf5_handler, create_dataset_hdf5

"""
Results.hdf5
    - group experimentName  ['scheme BrainNetCNN', 'scheme CNNElementWise', ...]
        - group runTime ['time 1', 'time 2', ...]
            -attrs 'configuration'      str
            -attrs 'cross validation'   str
            
        if cross validation = k folds
            -group folds ['fold 1', 'fold 2', ...] (cross validation = k-folds)
                -data Accuracy          list
                -data Sensitivity       list
                ...
                
        if cross validation = 'Monte Carlor'
            -data Accuracy          list
            -data Sensitivity       list
            ...
"""


class Result:
    def __init__(self,
                 dir_path: str = None,
                 result_types: list = None,
                 result_datasets: list = None,
                 ):
        if not dir_path:
            dir_path = '/'.join(__file__.split('/')[:-5])

        self.dir_path = dir_path
        self.result_file_path = os.path.join(
            dir_path, 'Result/DeepLearning/Results.hdf5').encode()

        self.result_types = result_types if result_types is not None else \
            ['Accuracy', 'Cost', 'Cross Entropy', 'L1 Penalty', 'L2 Penalty', 'Precision',
             'Recall', 'Specificity', 'F1 Score']

        self.result_datasets = result_datasets if result_datasets is not None else \
            ['train', 'valid', 'test']

    def analyse_results(self,
                        experiments: list,
                        exp_times: dict = None,
                        optimize_type: str = 'Cross Entropy',
                        optimize_dataset: str = 'valid',
                        optimize_epoch: int = None,
                        objective_types: list = None,
                        objective_dataset: str = 'test',
                        show_result: bool = True,
                        top: int = 50,
                        ):
        if objective_types is None:
            objective_types = ['Accuracy', 'Specificity', 'Recall']

        result_file = hdf5_handler(self.result_file_path)

        exp_groups = []
        for exp_name in experiments:
            exp_group = result_file.require_group(exp_name)
            for time in exp_group:
                if exp_times:
                    for exp_time in exp_times[exp_name]:
                        if exp_time in time:
                            exp_groups.append(exp_group.require_group(time))
                else:
                    exp_groups.append(exp_group.require_group(time))

        results_schemes = {}
        for experiment_group in exp_groups:
            print('*********************************************************')
            self.analyse_experiment(experiment_group=experiment_group,
                                    objective_types=objective_types,
                                    objective_dataset=objective_dataset,
                                    optimize_type=optimize_type,
                                    optimize_dataset=optimize_dataset,
                                    optimize_epoch=optimize_epoch,
                                    show_result=show_result,
                                    top=top)

        result_file.close()
        return results_schemes

    def analyse_experiment(self,
                           experiment_group: h5py.Group,
                           optimize_type: str = 'Cross Entropy',
                           optimize_dataset: str = 'valid',
                           optimize_epoch: int = None,
                           objective_types: list = None,
                           objective_dataset: str = 'test',
                           show_result: bool = True,
                           top: int = 50,
                           ):
        if objective_types is None:
            objective_types = ['Accuracy', 'Specificity', 'Recall']

        pas = self.analyse_configuration(
            experiment_group=experiment_group, show_parameters=show_result)
        if 'strategy' in pas and pas['strategy'] == 'basic':
            optimize_type = None
            optimize_dataset = None
            optimize_epoch = np.inf

        cross_validation = experiment_group.attrs['cross validation']
        results = {objective_type: [] for objective_type in objective_types}
        for run_time in experiment_group:
            time_group = experiment_group.require_group(run_time)
            result = self.analyse_run_time(time_group=time_group,
                                           cross_validation=cross_validation,
                                           objective_dataset=objective_dataset,
                                           objective_types=objective_types,
                                           optimize_epoch=optimize_epoch,
                                           optimize_dataset=optimize_dataset,
                                           optimize_type=optimize_type,
                                           )
            for objective_type in objective_types:
                results[objective_type].append(result[objective_type])
        results = {objective_type: np.array(
            results[objective_type]) for objective_type in results}

        if show_result:
            print('Experiment: {:s}\t'
                  'run times: {:d}\t'
                  'cross validation: {:s}'.format(experiment_group.name, len(experiment_group), cross_validation))
            self.show_experiment_results(exp_results=results,
                                         cross_validation=cross_validation,
                                         top=top)

        return results

    def analyse_run_time(self,
                         time_group: h5py.Group,
                         cross_validation: str,
                         objective_dataset: str = 'test',
                         objective_types: list = None,
                         optimize_epoch: int = None,
                         optimize_type: str = 'Cross Entropy',
                         optimize_dataset: str = 'valid',
                         ):

        if not objective_types:
            objective_types = ['Accuracy', 'Specificity', 'Recall']
        optimize_fun = {'Accuracy': np.argmax,
                        'Cross Entropy': np.argmin,
                        }
        result = {objective_type: [] for objective_type in objective_types}

        if 'fold' in cross_validation:
            for fold in time_group:
                fold_group = time_group.require_group(fold)
                optimize_epoch_tmp = optimize_epoch

                if optimize_type and optimize_dataset:
                    optimize_type_dataset = self.load_result(group=fold_group,
                                                             result_type=optimize_type,
                                                             result_dataset=optimize_dataset)
                    if not optimize_epoch_tmp:
                        optimize_epoch_tmp = optimize_fun[optimize_type](
                            optimize_type_dataset)

                for objective_type in objective_types:
                    objective_type_dataset = self.load_result(group=fold_group,
                                                              result_type=objective_type,
                                                              result_dataset=objective_dataset)
                    if not optimize_epoch_tmp or optimize_epoch_tmp >= len(objective_type_dataset):
                        optimize_epoch_tmp = len(objective_type_dataset) - 1
                    result[objective_type].append(
                        objective_type_dataset[optimize_epoch_tmp])
            result = {objective_type: np.array(
                result[objective_type]) for objective_type in result}

        elif cross_validation == 'Monte Calor':
            if optimize_type and optimize_dataset:
                optimize_type_dataset = self.load_result(group=time_group,
                                                         result_type=optimize_type,
                                                         result_dataset=optimize_dataset)
                if not optimize_epoch:
                    optimize_epoch = optimize_fun[optimize_type](
                        optimize_type_dataset)

            for objective_type in objective_types:
                objective_type_dataset = self.load_result(group=time_group,
                                                          result_type=objective_type,
                                                          result_dataset=objective_dataset)
                if not optimize_epoch or optimize_epoch >= len(objective_type_dataset):
                    optimize_epoch = len(objective_type_dataset) - 1
                result[objective_type].append(
                    objective_type_dataset[optimize_epoch])

            result = {objective_type: np.array(
                result[objective_type]) for objective_type in result}

        return result

    @staticmethod
    def load_result(group: h5py.Group,
                    result_type: str = 'Accuracy',
                    result_dataset: str = 'train',
                    ):
        try:
            data = np.array(group[result_type][result_dataset])
            return data
        except Exception:
            return None

    @staticmethod
    def analyse_configuration(experiment_group: h5py.Group,
                              key_paras: list = None,
                              show_parameters: bool = True,
                              ):

        if not key_paras:
            key_paras = ['SICE_training', 'SICE_lambda', 'lambda', 'pretraining_cycle', 'pre_learning_rate',
                         'strategy', 'back_epoch', 'stop_accuracy', 'learning_rates', 'learning_rate',
                         'min_learning_rate', 'decay_rate']

        configuration = parse_str(experiment_group.attrs['configurations'])

        parameters = configuration['parameters']
        pas = {}
        for configs in parameters:
            pas.update(parameters[configs])

        # layers = configuration['layers']

        if show_parameters:
            configuration_info_list = ['{:s}: {:}'.format(key_pa, pas[key_pa])
                                       for key_pa in key_paras if key_pa in pas]
            configuration_info = '\r\n'.join(configuration_info_list)
            print(configuration_info)

        return pas

    @staticmethod
    def show_experiment_results(exp_results: dict,
                                cross_validation: str,
                                top: int = None):
        for result_type in exp_results:
            result = exp_results[result_type]
            if 'fold' in cross_validation:
                result = np.mean(result, axis=1)

            result = np.squeeze(result)
            info = '{:s}: mean: {:4f}\tstd: {:4f}\t'.format(result_type,
                                                            np.mean(result),
                                                            np.var(result),
                                                            )

            if top:
                result_sorted = np.copy(result)
                result_sorted.sort()
                info += 'top-{:d}: {:4f}\t'.format(top,
                                                   np.mean(result_sorted[-top:]))

            print(info)

    def set_saving_group(self,
                         scheme: str, 
                         current_xml: str,
                         cross_validation):
        result_hdf5 = hdf5_handler(self.result_file_path)
        scheme_group = result_hdf5.require_group(scheme)
        scheme_groups = [scheme_group[name] for name in scheme_group]

        for scheme_group in scheme_groups:
            try:
                xml_str = scheme_group.attrs['configurations']
                cross_valid = scheme_group.attrs['cross validation']
            except KeyError:
                xml_str = ''
                cross_valid = ''
                
            if current_xml == xml_str and cross_validation == cross_valid:
                print('Exist save scheme name is {:s}.'.format(
                    scheme_group.name))
                return scheme_group.name, xml_str

        current_time = time.strftime(
            '%Y-%m-%d-%H-%M-%S', time.localtime(time.time()))
        save_scheme_name = '{:s}/{:s}'.format(scheme, current_time)
        result_hdf5.close()
        print('New save scheme name is {:s}.'.format(save_scheme_name))
        return save_scheme_name, current_xml

    def save_results(self,
                     save_scheme_name,
                     cross_validation,
                     current_xml,
                     results,
                     run_time,
                     fold_index: int = None,
                     show_info: bool = False):
        result_hdf5 = hdf5_handler(self.result_file_path)
        exp_group = result_hdf5.require_group('{:s}'.format(save_scheme_name))

        if 'cross validation' not in exp_group.attrs:
            exp_group.attrs['cross validation'] = cross_validation
        if 'configurations' not in exp_group.attrs:
            exp_group.attrs['configurations'] = current_xml
        else:
            assert exp_group.attrs['configurations'] == current_xml, 'Configurations doesn\'t match.'

        time_group = exp_group.require_group('time {:d}'.format(run_time))

        # Get the fold group
        if fold_index is None:
            fold_group = time_group
        else:
            fold_group = time_group.require_group(
                'fold {:d}'.format(fold_index))

        # Save results to h5py file
        for result_type in results:
            result_group = fold_group.require_group(result_type)
            for result_dataset in results[result_type]:
                result = np.array(results[result_type][result_dataset])
                create_dataset_hdf5(group=result_group,
                                    data=result,
                                    name=result_dataset,
                                    show_info=show_info)

        result_hdf5.close()

    def copy_groups(self,
                    source_group: str,
                    objective_group: str,
                    times: list = None,
                    if_pop: bool = False,
                    ):
        hdf5_file = hdf5_handler(filename=self.result_file_path)
        source_group = hdf5_file.require_group(source_group)
        if not times:
            times = list(source_group)
        else:
            times = ['time {:d}'.format(time) for time in times]

        objective_group = hdf5_file.require_group(objective_group)
        for time in times:
            time_group = source_group.require_group(time)
            objective_time_group = objective_group.require_group(time)
            for fold in time_group:
                if not 'fold' in fold:
                    continue
                objective_time_group[fold] = time_group[fold]

                if if_pop:
                    time_group.pop(fold)

    def write_results_to_excel(self,
                               schemes: list = None,
                               out_file_name: str = None,
                               parameters: dict = None,
                               ):
        result_file = hdf5_handler(self.result_file_path)
        if not schemes:
            schemes = list(result_file)
        if not out_file_name:
            out_file_name = 'results.xls'
        wb_path = os.path.join(
            self.dir_path, 'Result/DeepLearning/', out_file_name)

        wb = xlwt.Workbook(encoding='utf-8')

        for scheme in schemes:
            print('Writing scheme {:s}'.format(scheme))
            scheme_group = result_file[scheme]
            exp_groups = [result_file[scheme].require_group(
                exp) for exp in scheme_group]

            scheme_sheet = wb.add_sheet(
                sheetname=scheme, cell_overwrite_ok=True)

            if not parameters:
                parameters = ['SICE_training', 'SICE_lambda', 'lambda', 'pretraining_cycle', 'pre_learning_rate',
                              'strategy', 'back_epoch', 'stop_accuracy', 'learning_rates', 'learning_rate',
                              'min_learning_rate', 'decay_rate', 'cross_validation', 'mean_acc', 'std_acc', 'mean_sen', 'std_sen',
                              'mean_spe', 'std_spe']
            for index, pa in enumerate(parameters):
                scheme_sheet.write(index+1, 0, pa)

            col = 1
            for exp in exp_groups:
                print('Writing experiment {:s}'.format(exp.name))
                pas = self.analyse_configuration(experiment_group=exp,
                                                 key_paras=parameters,
                                                 show_parameters=False,
                                                 )
                if 'cross_validation' not in pas and 'cross validation' in exp.attrs:
                    pas['cross_validation'] = exp.attrs['cross validation']

                exp_results = self.analyse_experiment(
                    experiment_group=exp,
                    show_result=False,
                )
                accuracy = np.squeeze(exp_results['Accuracy'])
                sensitivity = np.squeeze(exp_results['Recall'])
                specificity = np.squeeze(exp_results['Specificity'])

                try:
                    pas['mean_acc'] = np.mean(accuracy)
                    pas['std_acc'] = np.std(accuracy)
                    pas['mean_sen'] = np.mean(sensitivity)
                    pas['std_sen'] = np.std(sensitivity)
                    pas['mean_spe'] = np.mean(specificity)
                    pas['std_spe'] = np.std(specificity)
                except Exception:
                    pass

                scheme_sheet.write(0, col, exp.name)
                for index, pa in enumerate(parameters):
                    if pa in pas:
                        try:
                            scheme_sheet.write(index+1, col, pas[pa])
                        except Exception:
                            pass

                if np.linalg.matrix_rank(accuracy) > 1:
                    raw_shape = np.shape(accuracy)
                    accuracy = np.reshape(accuracy, newshape=[-1, ])
                    # print('Reshape Accuracy from {:} to shape [{:}, ]'.format(raw_shape, np.shape(accuracy)))

                try:
                    for index, acc in enumerate(accuracy):
                        scheme_sheet.write(
                            len(parameters) + 1 + index, col, acc)
                    col += 1
                except Exception:
                    print('Skip experiment result {:s}'.format(exp.name))
                    continue

        wb.save(wb_path)

    def clear_results(self):
        result_file = hdf5_handler(self.result_file_path)
        for scheme in result_file:
            scheme_group = result_file[scheme]
            for exp in result_file[scheme]:
                if 'configurations' not in scheme_group[exp].attrs:
                    scheme_group.pop(exp)
        result_file.close()

    def arrange_results(self, if_pop: bool = False):
        result_file = hdf5_handler(self.result_file_path)

        for scheme in result_file:
            if 'scheme' in scheme:
                strs = scheme.split(' ')
                if len(strs) == 3:
                    scheme_name = strs[1]
                    date = strs[2]
                elif len(strs) == 2:
                    scheme_name = strs[1]
                    date = '2019'

                scheme_group = result_file.require_group(name=scheme_name)
                try:
                    scheme_group[date] = result_file[scheme]
                    print(
                        'Remove {:s} to {:s}/{:s}'.format(scheme, scheme_name, date))
                except Exception:
                    print('{:s}/{:s} already exist.'.format(scheme_group.name, date))

    @staticmethod
    def clear_empty(self, clear_date: str = None,
                    minimum_run_times: int = 0):
        result_file = hdf5_handler(b'F:/OneDrive/Data/Data/DCAE_results.hdf5')
        for scheme in result_file:
            if clear_date is not None and clear_date in scheme:
                continue

            scheme_group = result_file[scheme]
            if len(scheme_group) <= minimum_run_times:
                result_file.pop(scheme)
                print('Pop {:s} with run times {:d}'.format(
                    scheme, len(scheme_group)))
            if len(scheme_group) == minimum_run_times + 1:
                times = list(scheme_group)
                if len(scheme_group[times[0]]) < 5:
                    result_file.pop(scheme)
                    print('Pop {:s} with run times 1 and fold num {:d}'.format(
                        scheme, len(scheme_group[times[0]])))

    def statistical_analysis(self, experiments: list):
        hdf5 = hdf5_handler(self.result_file_path)
        accuracies = []
        for exp in experiments:
            exp_group = hdf5[exp]
            exp_results = self.analyse_experiment(
                experiment_group=exp_group,
                show_result=False,
            )
            accuracy = np.squeeze(exp_results['Accuracy'])

            if np.linalg.matrix_rank(accuracy) > 1:
                raw_shape = np.shape(accuracy)
                accuracy = np.reshape(accuracy, newshape=[-1, ])

            accuracies.append(accuracy)

        exp_nums = len(experiments)
        p_values = np.zeros(shape=[exp_nums, exp_nums])
        for i in range(exp_nums):
            for j in range(i):
                t, p = stats.ttest_ind(accuracies[i], accuracies[j])
                p_values[i, j] = p

        print(p_values)
        hdf5.close()


# clear_empty()
# clear_results(clear_date='scheme CNNElementWise')


def analyse(dir_path: str,
            experiments: list):
    rt = Result()

    # rt.arrange_results()
    # rt.clear_results()
    # exp_times = {'CNNGLasso': ['2019-12-24-14-59-20']}
    # rt.analyse_results(top=10, experiments=experiments)
    rt.write_results_to_excel(schemes=experiments)

    # rt.statistical_analysis(experiments=experiments)


if __name__ == '__main__':
    rt = Result()
    rt.write_results_to_excel(schemes=['CNNGLasso'])

# analyse(dir_path=dir_path, experiments=[
#     'CNNWithGLasso/2019-07-15-21-24-08',
#     '/BrainNetCNNEW/2019-05-30-16-47',
#     'BrainNetCNN/2019-12-15-14-34-13',
#     'DTLNN/2019-12-24-10-43-54',
#     'DenoisedAutoEncoder/2019-12-25-08-37-45',
#     ])
# analyse(dir_path=dir_path, experiments=[
#     'BrainNetCNNEW', 'CNNGLasso', 'CNNWithGLasso', 'BrainNetCNN', 'DTLNN', 'DenoisedAutoEncoder'
# ])
# rt.analyse_results(experiments=['CNNGLasso'])
# analyse(dir_path=dir_path, experiments=['SICSVM'])