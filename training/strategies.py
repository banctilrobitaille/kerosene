from abc import ABC, abstractmethod
from typing import Union, List

from training.trainers import ModelTrainer


class TrainingStrategy(ABC):

    def __init__(self, model_trainers: Union[List[ModelTrainer], ModelTrainer]):
        self._model_trainers = model_trainers if isinstance(model_trainers, list) else [model_trainers]

    @property
    def current_model_trainers_states(self):
        return [model_trainer.state for model_trainer in self._model_trainers]

    def _train_step(self, *inputs):
        self._model_trainers = list(map(lambda model_trainer: model_trainer.train(), self._model_trainers))

    def _validate_step(self, *inputs):
        pass

    def backward(self, losses):
        for loss_id, loss in enumerate(losses):
            if APEX_AVAILABLE:
                with amp.scale_loss(loss, self._model_trainers[loss_id].optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward()

    @abstractmethod
    def train_step(self, input, target):
        raise NotImplementedError()

    @abstractmethod
    def validate_step(self, input, target):
        raise NotImplementedError()
