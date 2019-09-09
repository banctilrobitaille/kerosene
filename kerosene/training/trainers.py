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
import logging
from abc import abstractmethod
from typing import Union, List

import torch
from ignite.metrics import Metric
from torch.utils.data import DataLoader

from kerosene.config.trainers import ModelTrainerConfiguration, RunConfiguration
from kerosene.events import Event, BaseEvent
from kerosene.events.generators.base_generator import EventGenerator
from kerosene.metrics.gauges import AverageGauge
from kerosene.metrics.metrics import MetricFactory
from kerosene.models.models import ModelFactory
from kerosene.nn.apex import ApexModule, ApexLoss
from kerosene.nn.criterions import CriterionFactory
from kerosene.optim.optimizers import OptimizerFactory
from kerosene.optim.schedulers import SchedulerFactory
from kerosene.training import Status


class ModelTrainer(ApexModule):
    LOGGER = logging.getLogger("ModelTrainer")

    def __init__(self, model_name, model, criterion, optimizer, scheduler, metric_computer: Metric):
        super().__init__(model, optimizer)
        self._model_name = model_name

        self._criterion = criterion
        self._scheduler = scheduler
        self._metric_computer = metric_computer

        self._step_train_loss = torch.tensor([0.0])
        self._step_valid_loss = torch.tensor([0.0])
        self._step_train_metric = torch.tensor([0.0])
        self._step_valid_metric = torch.tensor([0.0])

        self._train_loss = AverageGauge()
        self._valid_loss = AverageGauge()
        self._train_metric = AverageGauge()
        self._valid_metric = AverageGauge()

        self._status = Status.INITIALIZED

    @property
    def name(self):
        return self._model_name

    @property
    def step_train_loss(self):
        return torch.tensor([self._step_train_loss]).cpu()

    @property
    def step_valid_loss(self):
        return torch.tensor([self._step_valid_loss]).cpu()

    @property
    def step_train_metric(self):
        return torch.tensor([self._step_train_metric]).cpu()

    @property
    def step_valid_metric(self):
        return torch.tensor([self._step_valid_metric]).cpu()

    @property
    def train_loss(self):
        return torch.tensor([self._train_loss.compute()]).cpu()

    @property
    def valid_loss(self):
        return torch.tensor([self._valid_loss.compute()]).cpu()

    @property
    def train_metric(self):
        return torch.tensor([self._train_metric.compute()]).cpu()

    @property
    def valid_metric(self):
        return torch.tensor([self._valid_metric.compute()]).cpu()

    @property
    def optimizer_state(self):
        return self._optimizer.state_dict()

    @property
    def optimizer_lr(self):
        for param_group in self._optimizer.param_groups:
            return torch.tensor([param_group["lr"]])

    @property
    def model_state(self):
        return self._model.state_dict()

    @property
    def criterion(self):
        return self._criterion

    @property
    def scheduler(self):
        return self._scheduler

    def is_active(self):
        return self._status is not Status.FINALIZE

    def named_parameters(self, prefix: str = ..., recurse: bool = ...):
        return self.model.named_parameters()

    def forward(self, *input):
        return self._model.forward(*input)

    def eval(self):
        self._status = Status.VALID
        self._model.eval()

    def train(self, mode=True):
        self._status = Status.TRAIN
        self._model.train()

    def step(self):
        if self._status is Status.TRAIN:
            self._optimizer.step()

    def scheduler_step(self):
        if self._scheduler is not None:
            self._scheduler.step(self._train_loss.compute())

    def zero_grad(self):
        self._optimizer.zero_grad()

    def compute_train_loss(self, pred, target) -> Union[ApexLoss, torch.Tensor]:
        self._step_train_loss = self._criterion(pred, target)
        self._train_loss.update(self._step_train_loss)

        return ApexLoss(self._amp_id, self._step_train_loss, self._optimizer) if self.use_amp else self._step_train_loss

    def compute_valid_loss(self, pred, target) -> Union[ApexLoss, torch.Tensor]:
        self._step_valid_loss = self._criterion(pred, target)
        self._valid_loss.update(self._step_valid_loss)

        return ApexLoss(self._amp_id, self._step_valid_loss, self._optimizer) if self.use_amp else self._step_valid_loss

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

    def finalize(self):
        self._status = Status.FINALIZE


class Trainer(EventGenerator):
    LOGGER = logging.getLogger("Trainer")

    def __init__(self, name, train_data_loader: DataLoader, valid_data_loader: DataLoader,
                 model_trainers: Union[List[ModelTrainer], ModelTrainer],
                 run_config: RunConfiguration = RunConfiguration()):
        super().__init__()
        self._name = name
        self._train_data_loader = train_data_loader
        self._valid_data_loader = valid_data_loader
        self._model_trainers = model_trainers if isinstance(model_trainers, list) else [model_trainers]

        self._current_train_batch = 0
        self._current_valid_batch = 0
        self._current_epoch = 0

        self._run_config = run_config

        self._custom_variables = {}

        for amp_id, model_trainer in enumerate(self._model_trainers):
            model_trainer.initialize(amp_id, len(self._model_trainers), self._run_config)

        self._status = Status.INITIALIZED

    @property
    def name(self):
        return self._name

    @property
    def train_data_loader(self):
        return self._train_data_loader

    @property
    def valid_data_loader(self):
        return self._valid_data_loader

    @property
    def model_trainers(self):
        return self._model_trainers

    @property
    def current_train_step(self):
        return self._current_epoch * len(self._train_data_loader) + self._current_train_batch

    @property
    def current_valid_step(self):
        return self._current_epoch * len(self._valid_data_loader) + self._current_valid_batch

    @property
    def epoch(self):
        return self._current_epoch

    @property
    def custom_variables(self):
        return self._custom_variables

    @property
    def nb_of_train_batch(self):
        return len(self._train_data_loader)

    @property
    def nb_of_valid_batch(self):
        return len(self._valid_data_loader)

    @abstractmethod
    def train_step(self, inputs, target):
        raise NotImplementedError()

    @abstractmethod
    def validate_step(self, inputs, target):
        raise NotImplementedError()

    @abstractmethod
    def scheduler_step(self):
        raise NotImplementedError()

    @property
    def state(self):
        return self

    def is_active(self):
        return self._status is not Status.FINALIZE

    def _reset_model_trainers(self):
        for model_trainer in self._model_trainers:
            model_trainer.reset()

    def _at_least_one_model_is_active(self):
        return len(list(filter(lambda model_trainer: model_trainer.is_active(), self._model_trainers))) > 0

    def train(self, nb_epoch):
        self.fire(Event.ON_TRAINING_BEGIN)

        for self._current_epoch in range(0, nb_epoch):
            if self._at_least_one_model_is_active():
                self._on_epoch_begin()
                self.fire(Event.ON_TRAIN_EPOCH_BEGIN)
                self._train_epoch()
                self.fire(Event.ON_TRAIN_EPOCH_END)
                self.fire(Event.ON_VALID_EPOCH_BEGIN)
                self._validate_epoch()
                self.fire(Event.ON_VALID_EPOCH_END)
                self._on_epoch_end()
            else:
                self.finalize()

        self.fire(Event.ON_TRAINING_END)

    def _train_epoch(self):
        for model_trainer in self._model_trainers:
            model_trainer.train()

        for self._current_train_batch, (inputs, target) in enumerate(self._train_data_loader):
            self.fire(Event.ON_TRAIN_BATCH_BEGIN)

            inputs = [single_input.to(self._run_config.device, non_blocking=True) for single_input in
                      inputs] if isinstance(inputs, list) else inputs.to(self._run_config.device, non_blocking=True)

            target = [single_target.to(self._run_config.device, non_blocking=True) for single_target in
                      target] if isinstance(target, list) else target.to(self._run_config.device, non_blocking=True)

            self.train_step(inputs, target)
            if self._current_train_batch % 100 == 0:
                self.fire(Event.ON_100_TRAIN_STEPS)
            self.fire(Event.ON_TRAIN_BATCH_END)
            self.fire(Event.ON_BATCH_END)

    def _validate_epoch(self):

        for model_trainer in self._model_trainers:
            model_trainer.eval()

        with torch.no_grad():
            for self._current_valid_batch, (inputs, target) in enumerate(self._valid_data_loader):
                self.fire(Event.ON_VALID_BATCH_BEGIN)

                inputs = [single_input.to(self._run_config.device, non_blocking=True) for single_input in
                          inputs] if isinstance(inputs, list) else inputs.to(self._run_config.device, non_blocking=True)

                target = [single_target.to(self._run_config.device, non_blocking=True) for single_target in
                          target] if isinstance(target, list) else target.to(self._run_config.device, non_blocking=True)

                self.validate_step(inputs, target)
                self.fire(Event.ON_VALID_BATCH_END)
                self.fire(Event.ON_BATCH_END)

    def finalize(self):
        self._status = Status.FINALIZE

    def with_event_handler(self, handler, event: BaseEvent):
        if event in self._event_handlers.keys():
            self._event_handlers[event].append(handler)
        else:
            self._event_handlers[event] = [handler]

        return self

    def _on_epoch_begin(self):
        self.fire(Event.ON_EPOCH_BEGIN)
        self._reset_model_trainers()
        self.on_epoch_begin()

    def _on_epoch_end(self):
        self.fire(Event.ON_EPOCH_END)
        self.scheduler_step()
        self.on_epoch_end()


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
        model.compute_valid_metric(pred, target)
        model.compute_valid_loss(pred, target)

    def scheduler_step(self):
        self._model_trainers[0].scheduler_step()

    def on_epoch_begin(self):
        pass

    def on_epoch_end(self):
        pass


class ModelTrainerFactory(object):
    def __init__(self, model=None, model_factory: ModelFactory = None, optimizer_factory=OptimizerFactory(),
                 scheduler_factory=SchedulerFactory(), criterion_factory=CriterionFactory(),
                 metric_factory=MetricFactory()):
        self._model = model
        self._model_factory = model_factory
        self._optimizer_factory = optimizer_factory
        self._scheduler_factory = scheduler_factory
        self._criterion_factory = criterion_factory
        self._metric_factory = metric_factory

        assert (self._model is not None) or (
                self._model_factory is not None), "A model or a model factory must be provided !"

    def create(self, model_trainer_config: ModelTrainerConfiguration, run_config):
        torch.cuda.set_device(run_config.device)

        model = self._model if self._model is not None else self._model_factory.create(model_trainer_config.model_type,
                                                                                       model_trainer_config.model_params)
        optimizer = self._optimizer_factory.create(model_trainer_config.optimizer_type,
                                                   model.parameters(),
                                                   model_trainer_config.optimizer_params)
        scheduler = self._scheduler_factory.create(model_trainer_config.scheduler_type, optimizer,
                                                   model_trainer_config.scheduler_params)
        criterion = self._criterion_factory.create(model_trainer_config.criterion_type,
                                                   model_trainer_config.criterion_params).to(run_config.device)

        metric = self._metric_factory.create(model_trainer_config.metric_type, model_trainer_config.metric_params)

        return ModelTrainer(model_trainer_config.model_name, model, criterion, optimizer, scheduler, metric)
