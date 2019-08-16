from functools import reduce
from typing import List

from kerosene.events import Event
from kerosene.events.handlers.visdom.data import VisdomData, PlotType, PlotFrequency
from kerosene.events.preprocessors.base_preprocessor import EventPreprocessor
from kerosene.training.state import TrainerState, ModelTrainerState


class PlotAllModelStateVariables(EventPreprocessor):

    def __init__(self, model_name=None):
        self._model_name = model_name

    def __call__(self, event: Event, state: TrainerState) -> List[VisdomData]:
        model_state = list(filter(self.filter_by_name, state.model_trainer_states))

        if event == Event.ON_EPOCH_END:
            return reduce(lambda x, y: self.create_epoch_visdom_data(state.epoch, x).extend(
                self.create_epoch_visdom_data(state.epoch, y)), model_state)

    @staticmethod
    def create_epoch_visdom_data(epoch, state: ModelTrainerState):
        return [VisdomData(state.name, "Training Loss", PlotType.LINE_PLOT, PlotFrequency.EVERY_EPOCH,
                           state.train_loss, epoch),
                VisdomData(state.name, "Validation Loss", PlotType.LINE_PLOT, PlotFrequency.EVERY_EPOCH,
                           state.valid_loss, epoch),
                VisdomData(state.name, "Training Metric", PlotType.LINE_PLOT, PlotFrequency.EVERY_EPOCH,
                           state.train_metric, epoch),
                VisdomData(state.name, "Validation Metric", PlotType.LINE_PLOT, PlotFrequency.EVERY_EPOCH,
                           state.valid_metric, epoch)
                ]

    def filter_by_name(self, model_trainer_state: ModelTrainerState):
        return True if self._model_name is None else model_trainer_state.name == self._model_name
