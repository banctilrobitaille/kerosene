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
        return [VisdomData(state.name, "Training Loss", PlotType.LINE_PLOT, PlotFrequency.EVERY_EPOCH,
                           state.train_loss, epoch),
                VisdomData(state.name, "Validation Loss", PlotType.LINE_PLOT, PlotFrequency.EVERY_EPOCH,
                           state.valid_loss, epoch),
                VisdomData(state.name, "Training Metric", PlotType.LINE_PLOT, PlotFrequency.EVERY_EPOCH,
                           state.train_metric, epoch),
                VisdomData(state.name, "Validation Metric", PlotType.LINE_PLOT, PlotFrequency.EVERY_EPOCH,
                           state.valid_metric, epoch)]

    @staticmethod
    def create_train_batch_visdom_data(step, state: ModelTrainerState):
        return [VisdomData(state.name, "Training Loss", PlotType.LINE_PLOT, PlotFrequency.EVERY_STEP,
                           state.train_loss, step),
                VisdomData(state.name, "Training Metric", PlotType.LINE_PLOT, PlotFrequency.EVERY_STEP,
                           state.train_metric, step)]

    @staticmethod
    def create_valid_batch_visdom_data(step, state: ModelTrainerState):
        return [VisdomData(state.name, "Validation Loss", PlotType.LINE_PLOT, PlotFrequency.EVERY_STEP,
                           state.valid_loss, step),
                VisdomData(state.name, "Validation Metric", PlotType.LINE_PLOT, PlotFrequency.EVERY_STEP,
                           state.valid_metric, step)]

    def filter_by_name(self, model_trainer_state: ModelTrainerState):
        return True if self._model_name is None else model_trainer_state.name == self._model_name

    def flatten(self, list_of_visdom_data):
        return [item for sublist in list_of_visdom_data for item in sublist]
