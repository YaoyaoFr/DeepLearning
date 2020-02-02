"""Early stop

Returns:
    [type] -- [description]
"""
import os
import shutil
import collections
import numpy as np

from Log.log import Log

PARAMETERS = ['strategy',
              'training_cycle',
              'save_cycle',
              'tolerance_all',
              'learning_rate',
              'min_learning_rate',
              'decay_rate',
              'decay_step'
              'back_epoch',
              'stop_accuracy',
              'optimize_type',
              'optimize_dataset',
              ]

REQUIRED_ATTRS = {'basic': ['learning_rate',
                            'training_cycle',
                            'save_cycle'],
                  'early_stop': ['save_cycle',
                                 'tolerance_all',
                                 'learning_rate',
                                 'decay_rate',
                                 'decay_step'],
                  'restore': ['save_cycle',
                              'tolerance_all',
                              'learning_rate',
                              'decay_rate',
                              'back_epoch',
                              'min_learning_rate'
                              ],
                  }


class EarlyStop:
    """Class for early stop
    """
    epoch = 1
    optimal_epoch = 1
    overfitting_count = 0

    def __init__(self,
                 log: Log,
                 data: dict,
                 results: dict,
                 pas: dict) -> int:
        self.log = log
        self.results = {result: {tag: [] for tag in ['train', 'valid', 'test']
                                 if '{:s} data'.format(tag) in data}
                        for result in results}
        self.optimal_epochs = []
        self.parameters = collections.OrderedDict(
            {p: None for p in PARAMETERS})
        self.parameters.update(pas)

        assert self.parameters['strategy'] is not None, 'The early stop strategy must be fed.'

        # Check the reqired parameters
        missed_pas = []
        for p in REQUIRED_ATTRS[self.parameters['strategy']]:
            if self.parameters[p] is None:
                missed_pas.append(p)

        assert len(missed_pas) == 0, 'If strategy is {:s}, the parameters {:} should be fed.'.format(
            self.parameters['strategy'], missed_pas)

    def next(self,
             results: dict,
             ):
        """ 1. Show results of this epoch. 
            2. Cache the results.
            3. Judge if stop according to strategy and results.

        Arguments:
            results {dict} -- [description]

        Returns:
            epoch [int] -- current epoch
        """
        self.show_results_epoch(results=results)

        if self.epoch % self.parameters['save_cycle'] == 0:
            self.log.save_model(epoch=self.epoch, show_info=False)

        # Update results
        for result_tag in results:
            for result_type in results[result_tag]:
                result = results[result_tag][result_type]
                self.results[result_type][result_tag].append(result)

        if self.epoch == 1:
            self.optimal_epoch = self.epoch
            self.optimal_epochs.append(self.optimal_epoch)
            self.log.save_model(epoch=self.epoch, show_info=True)
            self.epoch += 1
        else:
            strategy_dict = {
                'basic': self.basic,
                'early_stop': self.early_stop,
                'restore': self.restore,
            }
            strategy_dict[self.parameters['strategy']](results=results)

        return self.epoch

    def basic(self,
              results: dict):
        """The base early stop strategy. Go to next step directly

        Arguments:
            results {dict} -- [description]
        """
        self.optimal_epoch = self.epoch
        self.epoch += 1

        if 'decay_step' in self.parameters and 'decay_rate' in self.parameters:
            decay_step = self.parameters['decay_step']
            decay_rate = self.parameters['decay_rate']
            if (self.epoch + 1) % decay_step == 0:
                self.parameters['learning_rate'] *= decay_rate
                print('Change learning rate to {:.5f}.'.format(
                    self.parameters['learning_rate']))

        return

    def early_stop(self,
                   results: dict):
        """Early stop according to optimal_type and optimal_dataset in results

        Arguments:
            results {dict} -- [description]
        """
        # Skip early stop while the training accuracy is too low
        if self.parameters['stop_accuracy'] is not None:
            if results['train']['Accuracy'] < self.parameters['stop_accuracy']:
                self.epoch += 1
                return

        # Parameters used in this strategy
        decay_step = self.parameters['decay_step']
        decay_rate = self.parameters['decay_rate']
        optimize_dataset = self.parameters['optimize_data']
        optimize_type = self.parameters['optimize_type']
        tolerance_all = self.parameters['tolerance_all']
        training_cycle = self.parameters['training_cycle']

        if (self.epoch + 1) % decay_step == 0:
            self.parameters['learning_rate'] *= decay_rate
            print('Change learning rate to {:.5f}.'.format(
                self.parameters['learning_rate']))

        if results[optimize_dataset][optimize_type] < np.min(
                np.array(self.results[optimize_type][optimize_dataset])[:-1]):
            self.optimal_epoch = self.epoch
            self.overfitting_count = 0
        else:
            self.overfitting_count += 1

        # When tolerance greater less than tolerance all
        if self.overfitting_count >= tolerance_all:
            self.epoch = training_cycle

        self.epoch += 1

    def restore(self,
                results: dict):
        """Restore model when overfitting has happend

        Arguments:
            results {dict} -- [description]
        """
        # Skip early stop while the training accuracy is too low
        if self.parameters['stop_accuracy'] is not None:
            if results['train']['Accuracy'] < self.parameters['stop_accuracy']:
                self.epoch += 1
                return

        # Parameters used in this strategy
        decay_rate = self.parameters['decay_rate']
        optimize_dataset = self.parameters['optimize_dataset']
        optimize_type = self.parameters['optimize_type']
        tolerance_all = self.parameters['tolerance_all']
        training_cycle = self.parameters['training_cycle']
        min_learning_rate = self.parameters['min_learning_rate']
        back_epoch = self.parameters['back_epoch']

        if results[optimize_dataset][optimize_type] < np.min(
                np.array(self.results[optimize_type][optimize_dataset])[:-1]):
            self.optimal_epoch = self.epoch
            self.optimal_epochs.append(self.epoch)
            self.log.save_model(epoch=self.epoch)
            self.overfitting_count = 0
        else:
            self.overfitting_count += 1

        # When tolerance greater less than tolerance all
        if self.overfitting_count >= tolerance_all:
            self.parameters['learning_rate'] *= decay_rate
            print('Change learning rate to {:.5f}.'.format(
                self.parameters['learning_rate']))

            # Stop restore
            if self.parameters['learning_rate'] < min_learning_rate:
                self.epoch = training_cycle
            else:
                back_epoch = - \
                    back_epoch if len(self.optimal_epochs) >= back_epoch else 0
                self.optimal_epoch = self.optimal_epochs[back_epoch]
                if back_epoch + 1 < 0:
                    self.optimal_epochs = self.optimal_epochs[:back_epoch + 1]
                self.epoch = self.optimal_epoch

                self.log.restore_model(restored_epoch=self.optimal_epoch)
                self.overfitting_count = 0
                self.results = {result: {tag: self.results[result][tag][:self.optimal_epoch]
                                         for tag in self.results[result]}
                                for result in self.results}

        self.epoch += 1

    def show_results_epoch(self,
                           results: dict,
                           ):
        """Show results in this epoch

        Arguments:
            results {dict} -- [description]
        """
        info = 'Epoch: {:d}\t'.format(self.epoch)

        result_tags = [tag for tag in [
            'train', 'valid', 'test'] if tag in results]
        for tag in result_tags:
            try:
                info += '{:s} CE: {:.5f}\taccuracy: {:.4f}\t'.format(tag,
                                                                     results[tag]['Cross Entropy'],
                                                                     results[tag]['Accuracy'])
            except KeyError:
                info += '{:s} MSE: {:.5f}\t'.format(tag,
                                                    results[tag]['MSE'])

        print(info)

    def show_results(self,
                     run_time: int,
                     fold_name: str = None):
        """Show results at the end of training process

        Arguments:
            run_time {int} -- 

        Keyword Arguments:
            fold_name {str} --  (default: {None})
        """
        info = 'Time: {:d}\t'.format(run_time)
        if fold_name is not None:
            info += '{:s}\t'.format(fold_name)
        for tag in ['Accuracy', 'Cross Entropy']:
            try:
                info += '{:s}: {:.5f}\t'.format(tag,
                                                self.results[tag]['test'][self.optimal_epoch - 1])
            except Exception:
                pass
        print(info)

    def clear_models(self,
                     save_first_model: bool = True,
                     save_optimal_model: bool = True,
                     save_final_model: bool = True,
                     ):
        """Save the first, optimal and final saved models.

        Keyword Arguments:
            save_first_model {bool} -- [description] (default: {True})
            save_optimal_model {bool} -- [description] (default: {True})
            save_final_model {bool} -- [description] (default: {True})
        """
        optimal_dir = os.path.join(self.log.dir_path, 'optimal_model')
        if not os.path.exists(optimal_dir):
            os.mkdir(optimal_dir)

        if save_final_model:
            final_path = os.path.join(optimal_dir, 'train_model_final')
            self.log.save_model(save_path=final_path)

        if save_first_model:
            self.log.restore_model(restored_epoch=1)
            first_path = os.path.join(optimal_dir, 'train_model_first')
            self.log.save_model(save_path=first_path)

        if save_optimal_model:
            optimal_dir = os.path.join(self.log.dir_path, 'optimal_model')
            self.log.restore_model(restored_epoch=self.optimal_epoch)
            optimal_path = os.path.join(optimal_dir, 'train_model_optimal')
            self.log.save_model(save_path=optimal_path)

        rm_dir = os.path.join(self.log.dir_path, 'model')
        shutil.rmtree(rm_dir)
