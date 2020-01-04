import os
import time
import operator
import scipy.io as sio

# Plot
import re
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.style as style
import matplotlib.cm as cm
import shutil

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


def screen(exp_groups: list,
           filters: list,
           ):
    operator_dict = {'<': operator.lt,
                     '<=': operator.le,
                     '>': operator.gt,
                     '>=': operator.ge,
                     '==': operator.eq,
                     }

    final_exp_groups = []
    for exp in exp_groups:
        exp_pas = Result.analyse_configuration(
            experiment_group=exp, show_parameters=False)
        exp_pas['run_times'] = len(exp)
        exp_pas['cross_validation'] = exp.attrs['cross validation']
        flag = True
        for filter in filters:
            strs = filter.split(' ')
            key = strs[0]
            ope = strs[1]
            value = ' '.join(strs[2:])
            try:
                value = float(value)
            except Exception:
                pass

            try:
                if not operator_dict[ope](exp_pas[key], value):
                    flag = False
                    break
            except Exception:
                print('The parameter {:} can not be checked.'.format(key))
                flag = False
                break

        if flag:
            final_exp_groups.append(exp)
    return final_exp_groups


class Result:
    def __init__(self,
                 dir_path: str = None,
                 result_file_path: str = None,
                 result_types: list = None,
                 result_datasets: list = None,
                 ):
        if not dir_path:
            dir_path = '/'.join(__file__.split('/')[:-5])

        self.dir_path = dir_path

        if not result_file_path:
            self.result_file_path = os.path.join(
                dir_path, 'Result/DeepLearning/Results.hdf5').encode()
        else:
            self.result_file_path = result_file_path.encode()

        self.result_types = result_types if result_types is not None else \
            ['Accuracy', 'Cost', 'Cross Entropy', 'L1 Penalty', 'L2 Penalty', 'Precision',
             'Recall', 'Specificity', 'F1 Score']

        self.result_datasets = result_datasets if result_datasets is not None else \
            ['train', 'valid', 'test']

    def analyse_results(self,
                        schemes: list = None,
                        experiment_names: list = None,
                        filters: list = None,
                        sort_pa: str = None,
                        optimize_type: str = 'Cross Entropy',
                        optimize_dataset: str = 'valid',
                        optimize_epoch: int = None,
                        objective_types: list = None,
                        objective_dataset: str = 'test',
                        show_result: bool = True,
                        top: int = 10,
                        ):
        if objective_types is None:
            objective_types = ['Accuracy', 'Specificity', 'Recall']

        hdf5 = hdf5_handler(self.result_file_path)

        exp_groups = []
        if schemes is not None:
            for scheme_name in schemes:
                scheme_group = hdf5[scheme_name]
                for exp_name in scheme_group:
                    exp_groups.append(scheme_group[exp_name])

        if experiment_names is not None:
            for exp_name in experiment_names:
                exp_groups.append(hdf5[exp_name])

        if filters is not None:
            exp_groups = screen(exp_groups=exp_groups, filters=filters)

        if sort_pa is not None:
            _, exp_groups = self.sort(exp_groups=exp_groups, sort_pa=sort_pa)

        results = []
        for experiment_group in exp_groups:
            print('*********************************************************')
            results.append(self.analyse_experiment(
                experiment_group=experiment_group,
                objective_types=objective_types,
                objective_dataset=objective_dataset,
                optimize_type=optimize_type,
                optimize_dataset=optimize_dataset,
                optimize_epoch=optimize_epoch,
                show_result=show_result,
                top=top))

        hdf5.close()
        return results

    def analyse_experiment(self,
                           experiment_group: h5py.Group = None,
                           experiment_name: str = None,
                           optimize_type: str = 'Cross Entropy',
                           optimize_dataset: str = 'valid',
                           optimize_epoch: int = None,
                           objective_types: list = None,
                           objective_dataset: str = 'test',
                           show_result: bool = True,
                           top: int = 10,
                           ):
        if objective_types is None:
            objective_types = ['Accuracy', 'Specificity', 'Recall']

        hdf5 = None
        if not experiment_group and experiment_name is not None:
            hdf5 = hdf5_handler(self.result_file_path)
            experiment_group = hdf5[experiment_name]

        pas = self.analyse_configuration(experiment_group=experiment_group,
                                         show_parameters=show_result)
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

        if hdf5 is not None:
            hdf5.close()

        return {'results': results,
                'parameters': pas}

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
            fold_nums = int(cross_validation.split(' ')[0])
            for fold_index in range(fold_nums):
                fold_group = time_group.require_group(
                    'fold {:d}'.format(fold_index + 1))
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
        print(exp_results['Accuracy'])

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
                         cross_validation,
                         if_reset: bool = False):
        result_hdf5 = hdf5_handler(self.result_file_path)
        scheme_group = result_hdf5.require_group(scheme)
        scheme_groups = [scheme_group[name] for name in scheme_group]

        if not if_reset:
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
                     fold_name: str = None,
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
        if fold_name is None:
            fold_group = time_group
        else:
            fold_group = time_group.require_group(fold_name)

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
                               sort_pa: str = None,
                               filters: list = None,
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

            # Filter the experimenets according to some rules
            if filters is not None:
                exp_groups = screen(exp_groups=exp_groups, filters=filters)

            # Writing to sheets
            scheme_sheet = wb.add_sheet(
                sheetname=scheme, cell_overwrite_ok=True)

            # Writing headers
            if not parameters:
                parameters = ['SICE_training', 'SICE_lambda', 'lambda', 'pretraining_cycle', 'pre_learning_rate',
                              'strategy', 'back_epoch', 'stop_accuracy', 'learning_rates', 'learning_rate',
                              'min_learning_rate', 'decay_rate', 'cross_validation', 'mean_acc', 'std_acc', 'mean_sen', 'std_sen',
                              'mean_spe', 'std_spe']
            for index, pa in enumerate(parameters):
                scheme_sheet.write(index+1, 0, pa)

            # Sort the results accoding to value of parameter
            exp_pas, exp_groups = self.sort(exp_groups=exp_groups,
                                            sort_pa=sort_pa,
                                            parameters=parameters)

            # Writing results
            col = 1
            for exp, pas in zip(exp_groups, exp_pas):
                print('Writing experiment {:s}'.format(exp.name))

                if 'cross_validation' not in pas and 'cross validation' in exp.attrs:
                    pas['cross_validation'] = exp.attrs['cross validation']

                try:
                    exp_results = self.analyse_experiment(
                        experiment_group=exp,
                        show_result=False,
                    )['results']
                except Exception:
                    continue
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

                # Write parameters
                scheme_sheet.write(0, col, exp.name)
                for index, pa in enumerate(parameters):
                    if pa in pas:
                        try:
                            scheme_sheet.write(index+1, col, pas[pa])
                        except Exception:
                            pass

                # Reshape matrix like (20, 5) to (100, 1)
                if np.linalg.matrix_rank(accuracy) > 1:
                    accuracy = np.reshape(accuracy, newshape=[-1, ])

                try:
                    for index, acc in enumerate(accuracy):
                        scheme_sheet.write(
                            len(parameters) + 1 + index, col, acc)
                    col += 1
                except Exception:
                    print('Skip experiment result {:s}'.format(exp.name))
                    continue

        wb.save(wb_path)

    def sort(self, exp_groups: list, sort_pa: str = None, parameters: list = None):
        exp_pas = []
        sort_pas = []
        for exp in exp_groups:
            exp_pa = self.analyse_configuration(experiment_group=exp,
                                                key_paras=parameters,
                                                show_parameters=False,
                                                )
            exp_pas.append(exp_pa)
            try:
                sort_pas.append(float(exp_pa[sort_pa]))
            except Exception:
                print(
                    'Parse sort parameter in experiment {:s} error'.format(exp.name))
                sort_pas.append(np.inf)

        if sort_pa is not None:
            pas, exp_groups, exp_pas = [list(x) for x in zip(
                *sorted(zip(sort_pas, exp_groups, exp_pas), key=lambda pair: pair[0]))]

        return exp_pas, exp_groups

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

    def gender_analyse(self,
                       exp_names: list,
                       figure_name: str = 'gender_results.png'):

        plt.style.use('ggplot')

        dir_path = os.path.join(
            '/home/ai/data/yaoyao/Result/DeepLearning', exp_names[2])
        male_results = sio.loadmat(os.path.join(dir_path, 'male_results.mat'))
        female_results = sio.loadmat(
            os.path.join(dir_path, 'female_results.mat'))
        data_type = 'Accuracy'
        male_data = male_results['male_{:s}'.format(data_type)]
        female_data = female_results['female_{:s}'.format(data_type)]

        fig = plt.figure(figsize=(30, 10))
        width = 0.35
        N = 5
        ind = np.arange(N)

        results = {}
        for exp_str, exp_name in zip(['femal', 'male'], exp_names[:2]):
            results[exp_str] = self.analyse_experiment(
                experiment_name=exp_name, show_result=False)['results'][data_type]

        male_mean = np.mean(results['male'], axis=0)
        male_std = np.std(results['male'], axis=0)
        female_mean = np.mean(results['femal'], axis=0)
        female_std = np.std(results['femal'], axis=0)
        ax1 = fig.add_subplot(121)
        ax1.bar(ind+width, female_mean, width, yerr=female_std, label='female')
        ax1.bar(ind, male_mean, width, yerr=male_std, label='male')
        plt.xticks(ind + width / 2,
                   tuple(['fold {:d}'.format(n+1) for n in range(N)]))
        plt.legend(loc='best', fontsize='xx-large')

        male_mean = np.mean(male_data, axis=0)
        male_std = np.std(male_data, axis=0)
        female_mean = np.mean(female_data, axis=0)
        female_std = np.std(female_data, axis=0)

        ax2 = fig.add_subplot(122)
        ax2.bar(ind+width, female_mean, width, yerr=female_std, label='female')
        ax2.bar(ind, male_mean, width, yerr=male_std, label='male')
        plt.xticks(ind + width / 2,
                   tuple(['fold {:d}'.format(n+1) for n in range(N)]))
        plt.legend(loc='best', fontsize='xx-large')

        save_dir_path = os.path.join(self.dir_path, 'Result/DeepLearning')
        plt.savefig(os.path.join(save_dir_path, figure_name), dpi=1000)

    def lambda_GLasso(self,
                      x_axis: str,
                      y_axes: str or list = 'Accuracy',
                      optimal_axis: str = 'Accuracy',
                      schemes: list = None,
                      experiment_names: list = None,
                      filters: list = None,
                      sort_pa: str = None,
                      figure_name: str = 'Results.png',
                      ):

        results = self.analyse_results(schemes=schemes,
                                       experiment_names=experiment_names,
                                       filters=filters,
                                       sort_pa=sort_pa,
                                       show_result=False,
                                       )

        # Figure style setup
        sns.set_style('whitegrid')
        style.use('ggplot')
        f, ax = plt.subplots(1, 1, figsize=(10, 7))

        # Set x axis
        x_axis_dict = {'SICE_lambda': '$\lambda_{GLasso}$'}
        ax.set_xlabel(x_axis_dict[x_axis])

        if isinstance(y_axes, str):
            y_axes = [y_axes]

        plt.ylim(bottom=0.63, top=0.68)
        lines = []
        for y_index, y_axis in enumerate(y_axes):
            color = cm.viridis((y_index + 1) / (len(y_axes) + 1))
            iter = [result['parameters'][x_axis] for result in results]
            returnavg = [np.mean(result['results'][y_axis])
                         for result in results]
            returnstd = [np.var(result['results'][y_axis])
                         for result in results]

            line, = ax.plot(iter, returnavg, color=color)
            lines.append(line)
            r1 = list(map(lambda x: x[0]-x[1], zip(returnavg, returnstd)))
            r2 = list(map(lambda x: x[0]+x[1], zip(returnavg, returnstd)))
            ax.fill_between(iter, r1, r2, color=color, alpha=0.2)

            # Optimal parameter chosen
            if y_axis == optimal_axis:
                optimal_index = np.argmax(returnavg)
                ax.annotate('Optimal {:s} = {:.2f}'.format(x_axis_dict[x_axis], iter[optimal_index]),
                            (iter[optimal_index], returnavg[optimal_index]),
                            xytext=(0.3, 0.8), textcoords='axes fraction',
                            arrowprops=dict(facecolor='grey', color='grey'))

        ax.legend(tuple(lines), tuple(y_axes), fontsize='xx-large')

        save_dir_path = os.path.join(self.dir_path, 'Result/DeepLearning')
        f.savefig(os.path.join(save_dir_path, figure_name), dpi=1000)


if __name__ == '__main__':
    filters = ['run_times == 100',
               'cross_validation == Monte Calor',
               'SICE_lambda >= 0.01',
               'SICE_lambda <= 0.2',
               ]

    old_path = '/home/ai/data/yaoyao/Result/DeepLearning_old/DCAE_results.hdf5'
    rt = Result(result_file_path=old_path)
    # rt = Result()

    # rt.write_results_to_excel(schemes=['CNNGLasso', 'CNNWithGLasso', 'DTLNN', 'FCNN', 'BrainNetCNN'],
                              #   sort_pa='SICE_lambda',
                            #   filters=filters,
                            #   )
    # results = rt.analyse_results(schemes=['CNNGLasso'],
    #                              filters=filters,
    #                              sort_pa='SICE_lambda',
    #                              optimize_epoch=-1,
    #                              show_result=True,
    #                              )

    # Plot
    # rt.lambda_GLasso(x_axis='SICE_lambda',
    #                  y_axes=['Accuracy', 'Precision', 'Recall'],
    #                  schemes=['CNNGLasso'],
    #                  sort_pa='SICE_lambda',
    #                  filters=filters,
    #                  figure_name='optimal_lambda_SICE.png')

    exp_name = 'CNNGLasso/2020-01-01-03-40-29'
    rt.analyse_experiment(experiment_name=exp_name)

    exp_names = [
        'CNNGLasso/2019-12-31-20-48-41',
        'CNNGLasso/2020-01-01-00-11-36',
        'CNNGLasso/2020-01-01/11-19-38',
    ]
    # rt.gender_analyse(exp_names=exp_names)


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
# analyse(dir_path=dir_path, experiments=['SICSVM'])
