from abc import abstractmethod
from typing import Union, List, Callable

import torch
from ignite.metrics import Metric
from torch import nn
from torch.utils.data import DataLoader

from kerosene.events import Event
from kerosene.events.generators import EventGenerator
from kerosene.events.handlers import HandlerPreprocessor
from kerosene.metrics.gauges import AverageGauge
from kerosene.training.state import TrainerState, ModelTrainerState

try:
    from apex import amp

    APEX_AVAILABLE = True
except ModuleNotFoundError:
    APEX_AVAILABLE = False


class ModelTrainer(nn.Module):

    def __init__(self, model_name, model, criterion, optimizer, scheduler, metric_computer: Metric):
        super().__init__()
        self._model_name = model_name
        self._model = model
        self._criterion = criterion
        self._optimizer = optimizer
        self._scheduler = scheduler
        self._metric_computer = metric_computer

        self._step_train_loss = 0
        self._step_valid_loss = 0
        self._step_train_metric = 0
        self._step_valid_metric = 0

        self._train_loss = AverageGauge()
        self._valid_loss = AverageGauge()
        self._train_metric = AverageGauge()
        self._valid_metric = AverageGauge()

    @property
    def optimizer_state(self):
        return self._optimizer.state_dict()

    @property
    def model_state(self):
        return self._model.state_dict()

    @property
    def criterion(self):
        return self._criterion

    @property
    def scheduler(self):
        return self._scheduler

    @property
    def state(self):
        return ModelTrainerState(self._train_loss.compute(),
                                 self._valid_loss.compute(),
                                 self._train_metric.compute(),
                                 self._valid_metric.compute(),
                                 self.model_state,
                                 self.optimizer_state)

    def forward(self, *input):
        return self._model.forward(*input)

    def eval(self):
        self._model.eval()

    def train(self, mode=True):
        self._model.train()

    def step(self):
        self._optimizer.step()

    def zero_grad(self):
        self._optimizer.zero_grad()

    def compute_train_loss(self, pred, target):
        self._step_train_loss = self._criterion(pred, target)
        self._train_loss.update(self._step_train_loss)

        return amp.scale_loss(self._step_train_loss, self._optimizer) if APEX_AVAILABLE else self._step_train_loss

    def compute_valid_loss(self, pred, target):
        self._step_valid_loss = self._criterion(pred, target)
        self._valid_loss.update(self._step_valid_loss)

        return amp.scale_loss(self._step_valid_loss, self._optimizer) if APEX_AVAILABLE else self._step_valid_loss

    def compute_train_metric(self, pred, target):
        self._metric_computer.update((pred, target))
        self._step_train_metric = self._metric_computer.compute()
        self._train_metric.update(self._step_train_metric)
        self._metric_computer.reset()

        return self._step_train_metric

    def compute_valid_metric(self, pred, target):
        self._metric_computer.update((pred, target))
        self._step_valid_metric = self._metric_computer.compute()
        self._valid_metric.update(self._step_valid_metric)
        self._metric_computer.reset()

        return self._step_valid_metric

    def reset(self):
        self._metric_computer.reset()
        self._train_loss.reset()
        self._valid_loss.reset()
        self._train_metric.reset()
        self._valid_metric.reset()


class Trainer(EventGenerator):

    def __init__(self, train_data_loader: DataLoader, valid_data_loader: DataLoader,
                 model_trainers: Union[List[ModelTrainer], ModelTrainer], configuration):
        super().__init__()
        self._train_data_loader = train_data_loader
        self._valid_data_loader = valid_data_loader
        self._model_trainers = model_trainers if isinstance(model_trainers, list) else [model_trainers]

        self._nb_epochs = configuration.nb_epochs
        self._current_train_batch = 0
        self._current_valid_batch = 0
        self._current_epoch = 0

    @property
    def nb_epochs(self):
        return self._nb_epochs

    @property
    def nb_of_train_batch(self):
        return len(self._train_data_loader)

    @property
    def nb_of_valid_batch(self):
        return len(self._valid_data_loader)

    @property
    def current_train_step(self):
        return self._current_epoch * len(self._train_data_loader) + self._current_train_batch

    @property
    def current_valid_step(self):
        return self._current_epoch * len(self._valid_data_loader) + self._current_valid_batch

    @property
    def state(self):
        return TrainerState(self._current_epoch,
                            self.current_train_step,
                            self.current_valid_step,
                            self._train_data_loader,
                            self._valid_data_loader,
                            [model_trainer.state for model_trainer in self._model_trainers])

    @abstractmethod
    def train_step(self, inputs, target):
        raise NotImplementedError()

    @abstractmethod
    def validate_step(self, inputs, target):
        raise NotImplementedError()

    def _reset_model_trainers(self):
        for model_trainer in self._model_trainers:
            model_trainer.reset()

    def train(self):
        self.fire(Event.ON_TRAINING_BEGIN)

        for self._current_epoch in range(0, self._nb_epochs):
            self.fire(Event.ON_EPOCH_BEGIN)
            self._reset_model_trainers()
            self.fire(Event.ON_TRAIN_EPOCH_BEGIN)
            self._train_epoch()
            self.fire(Event.ON_TRAIN_EPOCH_END)
            self.fire(Event.ON_VALID_EPOCH_BEGIN)
            self._validate_epoch()
            self.fire(Event.ON_VALID_EPOCH_END)
            self.fire(Event.ON_EPOCH_END)

        self.fire(Event.ON_TRAINING_END)

    def _train_epoch(self):

        self._model_trainers = [model_trainer.train() for model_trainer in self._model_trainers]

        for self._current_train_batch, (inputs, target) in enumerate(self._train_data_loader):
            self.fire(Event.ON_TRAIN_BATCH_BEGIN)
            target = target.cuda(non_blocking=True)
            self.train_step(inputs, target)
            self.fire(Event.ON_TRAIN_BATCH_END)

    def _validate_epoch(self):

        self._model_trainers = [model_trainer.eval() for model_trainer in self._model_trainers]

        with torch.no_grad():
            for self._current_valid_batch, (inputs, target) in enumerate(self._valid_data_loader):
                self.fire(Event.ON_VALID_BATCH_BEGIN)
                target = target.cuda(non_blocking=True)
                self.validate_step(inputs, target)
                self.fire(Event.ON_VALID_BATCH_END)

    def with_event_handler(self, handler, event: Event, preprocessor: Callable):
        if event in self._event_handlers.keys():
            self._event_handlers[event].append(HandlerPreprocessor(handler, preprocessor))
        else:
            self._event_handlers[event] = [HandlerPreprocessor(handler, preprocessor)]

        return self
