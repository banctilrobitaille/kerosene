from typing import List

from kerosene.events import Event
from kerosene.events.handlers.visdom.data import VisdomData, PlotType, PlotFrequency
from kerosene.events.preprocessors.base_preprocessor import EventPreprocessor
from kerosene.training.state import TrainerState, ModelTrainerState


class PlotAllModelStateVariables(EventPreprocessor):

    def __init__(self, model_name=None):
        self._model_name = model_name

    def __call__(self, event: Event, state: TrainerState) -> List[VisdomData]:
        model_states = list(filter(self.filter_by_name, state.model_trainer_states))

        if event == Event.ON_EPOCH_END:
            return self.flatten(
                list(map(lambda model_state: self.create_epoch_visdom_data(state.epoch, model_state), model_states)))
        elif event == Event.ON_TRAIN_BATCH_END:
            return self.flatten(
                list(map(lambda model_state: self.create_train_batch_visdom_data(state.train_step, model_state),
                         model_states)))
        elif event == Event.ON_VALID_BATCH_END:
            return self.flatten(
                list(map(lambda model_state: self.create_valid_batch_visdom_data(state.valid_step, model_state),
                         model_states)))

    @staticmethod
    def create_epoch_visdom_data(epoch, state: ModelTrainerState):
        data = PlotLosses.create_epoch_visdom_data(epoch, state)
        data.extend(PlotMetrics.create_epoch_visdom_data(epoch, state))

        return data

    @staticmethod
    def create_train_batch_visdom_data(step, state: ModelTrainerState):
        data = PlotLosses.create_train_batch_visdom_data(step, state)
        data.extend(PlotMetrics.create_train_batch_visdom_data(step, state))

        return data

    @staticmethod
    def create_valid_batch_visdom_data(step, state: ModelTrainerState):
        data = PlotLosses.create_valid_batch_visdom_data(step, state)
        data.extend(PlotMetrics.create_valid_batch_visdom_data(step, state))

        return data

    def filter_by_name(self, model_trainer_state: ModelTrainerState):
        return True if self._model_name is None else model_trainer_state.name == self._model_name

    def flatten(self, list_of_visdom_data):
        return [item for sublist in list_of_visdom_data for item in sublist]


class PlotLosses(EventPreprocessor):

    def __init__(self, model_name=None):
        self._model_name = model_name

    def __call__(self, event: Event, state: TrainerState) -> List[VisdomData]:
        model_states = list(filter(self.filter_by_name, state.model_trainer_states))

        if event == Event.ON_EPOCH_END:
            return self.flatten(
                list(map(lambda model_state: self.create_epoch_visdom_data(state.epoch, model_state), model_states)))
        elif event == Event.ON_TRAIN_BATCH_END:
            return self.flatten(
                list(map(lambda model_state: self.create_train_batch_visdom_data(state.train_step, model_state),
                         model_states)))
        elif event == Event.ON_VALID_BATCH_END:
            return self.flatten(
                list(map(lambda model_state: self.create_valid_batch_visdom_data(state.valid_step, model_state),
                         model_states)))

    @staticmethod
    def create_epoch_visdom_data(epoch, state: ModelTrainerState):
        return [VisdomData(state.name, "Training Loss", PlotType.LINE_PLOT, PlotFrequency.EVERY_EPOCH,
                           [[epoch, epoch]], [[state.train_loss, state.valid_loss]],
                           params={'opts': {'xlabel': str(PlotFrequency.EVERY_EPOCH),
                                            'ylabel': "Loss",
                                            'title': "{} {} per {}".format(state.name, "Loss",
                                                                           str(PlotFrequency.EVERY_EPOCH)),
                                            'legend': ["Training",
                                                       "Validation"]}})]

    @staticmethod
    def create_train_batch_visdom_data(step, state: ModelTrainerState):
        return [VisdomData(state.name, "Training Loss", PlotType.LINE_PLOT, PlotFrequency.EVERY_STEP,
                           [step], state.step_train_loss,
                           params={'opts': {'xlabel': str(PlotFrequency.EVERY_STEP),
                                            'ylabel': "Loss",
                                            'title': "{} {} per {}".format(state.name, "Training Loss",
                                                                           str(PlotFrequency.EVERY_EPOCH)),
                                            'legend': [state.name]}})]

    @staticmethod
    def create_valid_batch_visdom_data(step, state: ModelTrainerState):
        return [VisdomData(state.name, "Validation Loss", PlotType.LINE_PLOT, PlotFrequency.EVERY_STEP,
                           [step], state.step_valid_loss,
                           params={'opts': {'xlabel': str(PlotFrequency.EVERY_STEP),
                                            'ylabel': "Loss",
                                            'title': "{} {} per {}".format(state.name, "Validation Loss",
                                                                           str(PlotFrequency.EVERY_EPOCH)),
                                            'legend': [state.name]}})]

    def filter_by_name(self, model_trainer_state: ModelTrainerState):
        return True if self._model_name is None else model_trainer_state.name == self._model_name

    def flatten(self, list_of_visdom_data):
        return [item for sublist in list_of_visdom_data for item in sublist]


class PlotMetrics(EventPreprocessor):

    def __init__(self, model_name=None):
        self._model_name = model_name

    def __call__(self, event: Event, state: TrainerState) -> List[VisdomData]:
        model_states = list(filter(self.filter_by_name, state.model_trainer_states))

        if event == Event.ON_EPOCH_END:
            return self.flatten(
                list(
                    map(lambda model_state: self.create_epoch_visdom_data(state.epoch, model_state), model_states)))
        elif event == Event.ON_TRAIN_BATCH_END:
            return self.flatten(
                list(map(lambda model_state: self.create_train_batch_visdom_data(state.train_step, model_state),
                         model_states)))
        elif event == Event.ON_VALID_BATCH_END:
            return self.flatten(
                list(map(lambda model_state: self.create_valid_batch_visdom_data(state.valid_step, model_state),
                         model_states)))

    @staticmethod
    def create_epoch_visdom_data(epoch, state: ModelTrainerState):
        return [
            VisdomData(state.name, "Training Metric", PlotType.LINE_PLOT, PlotFrequency.EVERY_EPOCH, [[epoch, epoch]],
                       [[state.train_metric, state.valid_metric]],
                       params={'opts': {'xlabel': str(PlotFrequency.EVERY_EPOCH), 'ylabel': "Metric",
                                        'title': "{} {} per {}".format(state.name, "Metric",
                                                                       str(PlotFrequency.EVERY_EPOCH)),
                                        'legend': ["Training",
                                                   "Validation"]}})]

    @staticmethod
    def create_train_batch_visdom_data(step, state: ModelTrainerState):
        return [VisdomData(state.name, "Training Metric", PlotType.LINE_PLOT, PlotFrequency.EVERY_STEP,
                           [step], state.step_train_metric,
                           params={'opts': {'xlabel': str(PlotFrequency.EVERY_STEP),
                                            'ylabel': "Metric",
                                            'title': "{} {} per {}".format(state.name, "Training Metric",
                                                                           str(PlotFrequency.EVERY_STEP)),
                                            'legend': [state.name]}})]

    @staticmethod
    def create_valid_batch_visdom_data(step, state: ModelTrainerState):
        return [VisdomData(state.name, "Validation Metric", PlotType.LINE_PLOT, PlotFrequency.EVERY_STEP,
                           [step], state.step_valid_metric,
                           params={'opts': {'xlabel': str(PlotFrequency.EVERY_STEP),
                                            'ylabel': "Metric",
                                            'title': "{} {} per {}".format(state.name, "Validation Metric",
                                                                           str(PlotFrequency.EVERY_STEP)),
                                            'legend': [state.name]}})]

    def filter_by_name(self, model_trainer_state: ModelTrainerState):
        return True if self._model_name is None else model_trainer_state.name == self._model_name

    def flatten(self, list_of_visdom_data):
        return [item for sublist in list_of_visdom_data for item in sublist]


class PlotCustomVariables(EventPreprocessor):
    def __init__(self, variable_name, plot_type: PlotType, params):
        self._variable_name = variable_name
        self._plot_type = plot_type
        self._params = params

    def __call__(self, event: Event, state: TrainerState) -> List[VisdomData]:

        if event == Event.ON_EPOCH_END:
            return self.create_epoch_visdom_data(state)
        elif event == Event.ON_100_TRAIN_STEPS:
            return self.create_train_batch_visdom_data(state)
        elif event == Event.ON_TRAIN_BATCH_END:
            return self.create_train_batch_visdom_data(state)
        elif event == Event.ON_VALID_BATCH_END:
            return self.create_valid_batch_visdom_data(state)

    def create_epoch_visdom_data(self, state: TrainerState):
        return [VisdomData(state.name, self._variable_name, self._plot_type, PlotFrequency.EVERY_EPOCH,
                           [state.epoch], state.custom_variables[self._variable_name], self._params)]

    def create_train_batch_visdom_data(self, state: TrainerState):
        return [VisdomData(state.name, self._variable_name, self._plot_type, PlotFrequency.EVERY_STEP,
                           [state.train_step], state.custom_variables[self._variable_name], self._params)]

    def create_valid_batch_visdom_data(self, state: TrainerState):
        return [VisdomData(state.name, self._variable_name, self._plot_type, PlotFrequency.EVERY_STEP,
                           [state.valid_step], state.custom_variables[self._variable_name], self._params)]


class PlotLR(EventPreprocessor):

    def __call__(self, event: Event, state: TrainerState) -> List[VisdomData]:
        if event == Event.ON_EPOCH_END:
            return list(map(lambda model_state: self.create_epoch_visdom_data(state.epoch, model_state),
                            state.model_trainer_states))

    @staticmethod
    def create_epoch_visdom_data(epoch, state: ModelTrainerState):
        return VisdomData(state.name, "Learning Rate", PlotType.LINE_PLOT, PlotFrequency.EVERY_EPOCH, [epoch],
                          state.optimizer_lr,
                          params={'opts': {'xlabel': str(PlotFrequency.EVERY_EPOCH),
                                           'ylabel': "Learning Rate",
                                           'title': "{} {} per {}".format(state.name, "Learning Rate",
                                                                          str(PlotFrequency.EVERY_EPOCH)),
                                           'legend': [state.name]}})
