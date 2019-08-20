from abc import abstractmethod
from typing import Union, List, Callable

import torch
from ignite.metrics import Metric
from torch import nn
from torch.utils.data import DataLoader

from kerosene.config.trainers import ModelTrainerConfiguration, RunConfiguration
from kerosene.events import Event
from kerosene.events.generators.base_generator import EventGenerator
from kerosene.events.preprocessors.base_preprocessor import HandlerPreprocessor, Identity
from kerosene.metrics.gauges import AverageGauge
from kerosene.metrics.metrics import MetricFactory
from kerosene.models.models import ModelFactory
from kerosene.nn.criterions import CriterionFactory
from kerosene.optim.optimizers import OptimizerFactory
from kerosene.optim.schedulers import SchedulerFactory
from kerosene.training.state import TrainerState, ModelTrainerState
from kerosene.utils.distributed import on_single_device

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
        self._optimizer = optimizer
        self._criterion = criterion
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
    def model(self):
        return self._model

    @model.setter
    def model(self, value):
        self._model = value

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
        return ModelTrainerState(self._model_name,
                                 self._train_loss.compute(),
                                 self._valid_loss.compute(),
                                 self._train_metric.compute(),
                                 self._valid_metric.compute(),
                                 self._model,
                                 self._optimizer)

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

        return self._step_train_loss

    def compute_valid_loss(self, pred, target):
        self._step_valid_loss = self._criterion(pred, target)
        self._valid_loss.update(self._step_valid_loss)

        return self._step_valid_loss

    def backward(self, loss):
        if APEX_AVAILABLE:
            with amp.scale_loss(loss, self._optimizer) as scaled_loss:
                scaled_loss.backward()
        else:
            loss.backward()

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

    def configure(self, run_config: RunConfiguration):
        if on_single_device(run_config.devices):
            self._model.cuda(device=run_config.devices[0])
        else:
            # TODO implement distributed training
            raise NotImplementedError("Distributed training is not yet supported")

        if APEX_AVAILABLE and run_config.use_amp:
            self._model, self._optimizer = amp.initialize(
                self._model, self._optimizer, opt_level=run_config.amp_opt_level)


class Trainer(EventGenerator):

    def __init__(self, train_data_loader: DataLoader, valid_data_loader: DataLoader,
                 model_trainers: Union[List[ModelTrainer], ModelTrainer], run_config: RunConfiguration):
        super().__init__()
        self._train_data_loader = train_data_loader
        self._valid_data_loader = valid_data_loader
        self._model_trainers = model_trainers if isinstance(model_trainers, list) else [model_trainers]

        self._current_train_batch = 0
        self._current_valid_batch = 0
        self._current_epoch = 0

        self._run_config = run_config

        for model_trainer in self._model_trainers:
            model_trainer.configure(self._run_config)

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

    def train(self, nb_epoch):
        self.fire(Event.ON_TRAINING_BEGIN)

        for self._current_epoch in range(0, nb_epoch):
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

        for model_trainer in self._model_trainers:
            model_trainer.train()

        for self._current_train_batch, (inputs, target) in enumerate(self._train_data_loader):
            self.fire(Event.ON_TRAIN_BATCH_BEGIN)
            if on_single_device(self._run_config.devices):

                inputs = [single_input.cuda(self._run_config.devices[0]) for single_input in inputs] if isinstance(
                    inputs, list) else inputs.cuda(self._run_config.devices[0])

                target = [single_target.cuda(self._run_config.devices[0]) for single_target in target] if isinstance(
                    target, list) else target.cuda(self._run_config.devices[0])
            else:
                # TODO Implement distributed training
                raise NotImplementedError("Distributed training not implemented yet !")
            self.train_step(inputs, target)
            self.fire(Event.ON_TRAIN_BATCH_END)

    def _validate_epoch(self):

        for model_trainer in self._model_trainers:
            model_trainer.eval()

        with torch.no_grad():
            for self._current_valid_batch, (inputs, target) in enumerate(self._valid_data_loader):
                self.fire(Event.ON_VALID_BATCH_BEGIN)
                if on_single_device(self._run_config.devices):
                    if on_single_device(self._run_config.devices):
                        inputs = [single_input.cuda(self._run_config.devices[0]) for single_input in
                                  inputs] if isinstance(inputs, list) else inputs.cuda(self._run_config.devices[0])

                        target = [single_target.cuda(self._run_config.devices[0]) for single_target in
                                  target] if isinstance(target, list) else target.cuda(self._run_config.devices[0])
                else:
                    # TODO Implement distributed training
                    raise NotImplementedError("Distributed training not implemented yet !")
                self.validate_step(inputs, target)
                self.fire(Event.ON_VALID_BATCH_END)

    def with_event_handler(self, handler, event: Event, preprocessor: Callable = Identity()):
        if event in self._event_handlers.keys():
            self._event_handlers[event].append(HandlerPreprocessor(handler, preprocessor))
        else:
            self._event_handlers[event] = [HandlerPreprocessor(handler, preprocessor)]

        return self


class SimpleTrainer(Trainer):

    def train_step(self, inputs, target):
        model = self._model_trainers[0]

        pred = model.forward(inputs)
        model.compute_train_metric(pred, target)
        loss = model.compute_train_loss(pred, target)

        model.zero_grad()
        loss.backward()
        model.step()

    def validate_step(self, inputs, target):
        model = self._model_trainers[0]

        pred = model.forward(inputs)
        model.compute_train_metric(pred, target)
        model.compute_train_loss(pred, target)


class ModelTrainerFactory(object):
    def __init__(self, model_factory: ModelFactory, optimizer_factory=OptimizerFactory(),
                 scheduler_factory=SchedulerFactory(), criterion_factory=CriterionFactory(),
                 metric_factory=MetricFactory()):
        self._model_factory = model_factory
        self._optimizer_factory = optimizer_factory
        self._scheduler_factory = scheduler_factory
        self._criterion_factory = criterion_factory
        self._metric_factory = metric_factory

    def create(self, model_trainer_config: ModelTrainerConfiguration):
        model = self._model_factory.create(model_trainer_config.model_type,
                                           model_trainer_config.model_params)
        optimizer = self._optimizer_factory.create(model_trainer_config.optimizer_type,
                                                   model.parameters(),
                                                   model_trainer_config.optimizer_params)
        scheduler = self._scheduler_factory.create(model_trainer_config.scheduler_type, optimizer,
                                                   model_trainer_config.scheduler_params)
        criterion = self._criterion_factory.create(model_trainer_config.criterion_type,
                                                   model_trainer_config.criterion_params)
        metric = self._metric_factory.create(model_trainer_config.metric_type, model_trainer_config.metric_params)

        return ModelTrainer(model_trainer_config.model_name, model, criterion, optimizer, scheduler, metric)
