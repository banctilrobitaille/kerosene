# -*- coding: utf-8 -*-
# Copyright 2019 Kerosene Authors. All Rights Reserved.
#
# Licensed under the MIT License;
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://opensource.org/licenses/MIT
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
from typing import List

from kerosene.events import Event
from kerosene.events.handlers.visdom.data import VisdomData, PlotType, PlotFrequency
from kerosene.events.preprocessors.base_preprocessor import EventPreprocessor
from kerosene.training.trainers import Trainer, ModelTrainer


class PlotAllModelStateVariables(EventPreprocessor):

    def __init__(self, model_name=None):
        self._model_name = model_name

    def __call__(self, event: Event, trainer: Trainer) -> List[VisdomData]:
        model_states = list(filter(self.filter_by_name, trainer.model_trainers))

        if event == Event.ON_EPOCH_END:
            return self.flatten(
                list(map(lambda model_state: self.create_epoch_visdom_data(trainer.epoch, model_state), model_states)))
        elif event == Event.ON_TRAIN_BATCH_END:
            return self.flatten(
                list(map(
                    lambda model_state: self.create_train_batch_visdom_data(trainer.current_train_step, model_state),
                    model_states)))
        elif event == Event.ON_VALID_BATCH_END:
            return self.flatten(
                list(map(
                    lambda model_state: self.create_valid_batch_visdom_data(trainer.current_valid_step, model_state),
                    model_states)))

    @staticmethod
    def create_epoch_visdom_data(epoch, model_trainer: ModelTrainer):
        data = PlotLosses.create_epoch_visdom_data(epoch, model_trainer)
        data.extend(PlotMetrics.create_epoch_visdom_data(epoch, model_trainer))

        return data

    @staticmethod
    def create_train_batch_visdom_data(step, model_trainer: ModelTrainer):
        data = PlotLosses.create_train_batch_visdom_data(step, model_trainer)
        data.extend(PlotMetrics.create_train_batch_visdom_data(step, model_trainer))

        return data

    @staticmethod
    def create_valid_batch_visdom_data(step, model_trainer: ModelTrainer):
        data = PlotLosses.create_valid_batch_visdom_data(step, model_trainer)
        data.extend(PlotMetrics.create_valid_batch_visdom_data(step, model_trainer))

        return data

    def filter_by_name(self, model_trainer: ModelTrainer):
        return True if self._model_name is None else model_trainer.name == self._model_name

    def flatten(self, list_of_visdom_data):
        return [item for sublist in list_of_visdom_data for item in sublist]


class PlotLosses(EventPreprocessor):

    def __init__(self, model_name=None):
        self._model_name = model_name

    def __call__(self, event: Event, trainer: Trainer) -> List[VisdomData]:
        model_states = list(filter(self.filter_by_name, trainer.model_trainers))

        if event == Event.ON_EPOCH_END:
            return self.flatten(
                list(map(lambda model_state: self.create_epoch_visdom_data(trainer.epoch, model_state), model_states)))
        elif event == Event.ON_TRAIN_BATCH_END:
            return self.flatten(
                list(map(
                    lambda model_state: self.create_train_batch_visdom_data(trainer.current_train_step, model_state),
                    model_states)))
        elif event == Event.ON_VALID_BATCH_END:
            return self.flatten(
                list(map(
                    lambda model_state: self.create_valid_batch_visdom_data(trainer.current_valid_step, model_state),
                    model_states)))

    @staticmethod
    def create_epoch_visdom_data(epoch, model_trainer: ModelTrainer):
        return [VisdomData(model_trainer.name, "Training Loss", PlotType.LINE_PLOT, PlotFrequency.EVERY_EPOCH,
                           [[epoch, epoch]], [[model_trainer.train_loss, model_trainer.valid_loss]],
                           params={'opts': {'xlabel': str(PlotFrequency.EVERY_EPOCH),
                                            'ylabel': "Loss",
                                            'title': "{} {} per {}".format(model_trainer.name, "Loss",
                                                                           str(PlotFrequency.EVERY_EPOCH)),
                                            'legend': ["Training",
                                                       "Validation"]}})]

    @staticmethod
    def create_train_batch_visdom_data(step, model_trainer: ModelTrainer):
        return [VisdomData(model_trainer.name, "Training Loss", PlotType.LINE_PLOT, PlotFrequency.EVERY_STEP,
                           [step], model_trainer.step_train_loss,
                           params={'opts': {'xlabel': str(PlotFrequency.EVERY_STEP),
                                            'ylabel': "Loss",
                                            'title': "{} {} per {}".format(model_trainer.name, "Training Loss",
                                                                           str(PlotFrequency.EVERY_EPOCH)),
                                            'legend': [model_trainer.name]}})]

    @staticmethod
    def create_valid_batch_visdom_data(step, model_trainer: ModelTrainer):
        return [VisdomData(model_trainer.name, "Validation Loss", PlotType.LINE_PLOT, PlotFrequency.EVERY_STEP,
                           [step], model_trainer.step_valid_loss,
                           params={'opts': {'xlabel': str(PlotFrequency.EVERY_STEP),
                                            'ylabel': "Loss",
                                            'title': "{} {} per {}".format(model_trainer.name, "Validation Loss",
                                                                           str(PlotFrequency.EVERY_EPOCH)),
                                            'legend': [model_trainer.name]}})]

    def filter_by_name(self, model_trainer: ModelTrainer):
        return True if self._model_name is None else model_trainer.name == self._model_name

    def flatten(self, list_of_visdom_data):
        return [item for sublist in list_of_visdom_data for item in sublist]


class PlotMetrics(EventPreprocessor):

    def __init__(self, model_name=None):
        self._model_name = model_name

    def __call__(self, event: Event, model_trainer: Trainer) -> List[VisdomData]:
        model_states = list(filter(self.filter_by_name, model_trainer.model_trainers))

        if event == Event.ON_EPOCH_END:
            return self.flatten(
                list(
                    map(lambda model_state: self.create_epoch_visdom_data(model_trainer.epoch, model_state),
                        model_states)))
        elif event == Event.ON_TRAIN_BATCH_END:
            return self.flatten(
                list(map(lambda model_state: self.create_train_batch_visdom_data(model_trainer.current_train_step,
                                                                                 model_state),
                         model_states)))
        elif event == Event.ON_VALID_BATCH_END:
            return self.flatten(
                list(map(lambda model_state: self.create_valid_batch_visdom_data(model_trainer.current_valid_step,
                                                                                 model_state),
                         model_states)))

    @staticmethod
    def create_epoch_visdom_data(epoch, model_trainer: ModelTrainer):
        return [
            VisdomData(model_trainer.name, "Training Metric", PlotType.LINE_PLOT, PlotFrequency.EVERY_EPOCH,
                       [[epoch, epoch]],
                       [[model_trainer.train_metric, model_trainer.valid_metric]],
                       params={'opts': {'xlabel': str(PlotFrequency.EVERY_EPOCH), 'ylabel': "Metric",
                                        'title': "{} {} per {}".format(model_trainer.name, "Metric",
                                                                       str(PlotFrequency.EVERY_EPOCH)),
                                        'legend': ["Training",
                                                   "Validation"]}})]

    @staticmethod
    def create_train_batch_visdom_data(step, model_trainer: ModelTrainer):
        return [VisdomData(model_trainer.name, "Training Metric", PlotType.LINE_PLOT, PlotFrequency.EVERY_STEP,
                           [step], model_trainer.step_train_metric,
                           params={'opts': {'xlabel': str(PlotFrequency.EVERY_STEP),
                                            'ylabel': "Metric",
                                            'title': "{} {} per {}".format(model_trainer.name, "Training Metric",
                                                                           str(PlotFrequency.EVERY_STEP)),
                                            'legend': [model_trainer.name]}})]

    @staticmethod
    def create_valid_batch_visdom_data(step, model_trainer: ModelTrainer):
        return [VisdomData(model_trainer.name, "Validation Metric", PlotType.LINE_PLOT, PlotFrequency.EVERY_STEP,
                           [step], model_trainer.step_valid_metric,
                           params={'opts': {'xlabel': str(PlotFrequency.EVERY_STEP),
                                            'ylabel': "Metric",
                                            'title': "{} {} per {}".format(model_trainer.name, "Validation Metric",
                                                                           str(PlotFrequency.EVERY_STEP)),
                                            'legend': [model_trainer.name]}})]

    def filter_by_name(self, model_trainer: ModelTrainer):
        return True if self._model_name is None else model_trainer.name == self._model_name

    def flatten(self, list_of_visdom_data):
        return [item for sublist in list_of_visdom_data for item in sublist]


class PlotCustomVariables(EventPreprocessor):
    def __init__(self, variable_name, plot_type: PlotType, params):
        self._variable_name = variable_name
        self._plot_type = plot_type
        self._params = params

    def __call__(self, event: Event, trainer: Trainer) -> List[VisdomData]:

        if event == Event.ON_EPOCH_END:
            return self.create_epoch_visdom_data(trainer)
        elif event == Event.ON_100_TRAIN_STEPS:
            return self.create_train_batch_visdom_data(trainer)
        elif event == Event.ON_TRAIN_BATCH_END:
            return self.create_train_batch_visdom_data(trainer)
        elif event == Event.ON_VALID_BATCH_END:
            return self.create_valid_batch_visdom_data(trainer)

    def create_epoch_visdom_data(self, trainer: Trainer):
        return [VisdomData(trainer.name, self._variable_name, self._plot_type, PlotFrequency.EVERY_EPOCH,
                           [trainer.epoch], trainer.custom_variables[self._variable_name], self._params)]

    def create_train_batch_visdom_data(self, trainer: Trainer):
        return [VisdomData(trainer.name, self._variable_name, self._plot_type, PlotFrequency.EVERY_STEP,
                           [trainer.current_train_step], trainer.custom_variables[self._variable_name], self._params)]

    def create_valid_batch_visdom_data(self, trainer: Trainer):
        return [VisdomData(trainer.name, self._variable_name, self._plot_type, PlotFrequency.EVERY_STEP,
                           [trainer.current_valid_step], trainer.custom_variables[self._variable_name], self._params)]


class PlotLR(EventPreprocessor):

    def __call__(self, event: Event, trainer: Trainer) -> List[VisdomData]:
        if event == Event.ON_EPOCH_END:
            return list(map(lambda model_state: self.create_epoch_visdom_data(trainer.epoch, model_state),
                            trainer.model_trainers))

    @staticmethod
    def create_epoch_visdom_data(epoch, model_trainer: ModelTrainer):
        return VisdomData(model_trainer.name, "Learning Rate", PlotType.LINE_PLOT, PlotFrequency.EVERY_EPOCH, [epoch],
                          model_trainer.optimizer_lr,
                          params={'opts': {'xlabel': str(PlotFrequency.EVERY_EPOCH),
                                           'ylabel': "Learning Rate",
                                           'title': "{} {} per {}".format(model_trainer.name, "Learning Rate",
                                                                          str(PlotFrequency.EVERY_EPOCH)),
                                           'legend': [model_trainer.name]}})
