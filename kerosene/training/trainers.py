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
from kerosene.utils.devices import on_cpu


class ModelTrainer(ApexModule):
    LOGGER = logging.getLogger("ModelTrainer")

    def __init__(self, model_name, model, criterion, optimizer, scheduler, metric_computer: Metric,
                 max_grad_norm: Union[None, float]):
        super().__init__(model, optimizer)
        self._model_name = model_name

        self._criterion = criterion
        self._scheduler = scheduler
        self._metric_computer = metric_computer
        self._max_grad_norm = max_grad_norm

        self._step_train_loss = torch.tensor([0.0])
        self._step_valid_loss = torch.tensor([0.0])
        self._step_test_loss = torch.tensor([0.0])
        self._step_train_metric = torch.tensor([0.0])
        self._step_valid_metric = torch.tensor([0.0])
        self._step_test_metric = torch.tensor([0.0])

        self._train_loss = AverageGauge()
        self._valid_loss = AverageGauge()
        self._test_loss = AverageGauge()
        self._train_metric = AverageGauge()
        self._valid_metric = AverageGauge()
        self._test_metric = AverageGauge()

        self._status = Status.INITIALIZATION

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
    def step_test_loss(self):
        return torch.tensor([self._step_test_loss]).cpu()

    @property
    def step_train_metric(self):
        return torch.tensor([self._step_train_metric]).cpu()

    @property
    def step_valid_metric(self):
        return torch.tensor([self._step_valid_metric]).cpu()

    @property
    def step_test_metric(self):
        return torch.tensor([self._step_test_metric]).cpu()

    @property
    def train_loss(self):
        return torch.tensor([self._train_loss.compute()]).cpu()

    @property
    def valid_loss(self):
        return torch.tensor([self._valid_loss.compute()]).cpu()

    @property
    def test_loss(self):
        return torch.tensor([self._test_loss.compute()]).cpu()

    @property
    def train_metric(self):
        return torch.tensor([self._train_metric.compute()]).cpu()

    @property
    def valid_metric(self):
        return torch.tensor([self._valid_metric.compute()]).cpu()

    @property
    def test_metric(self):
        return torch.tensor([self._test_metric.compute()]).cpu()

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
        return self._status is not Status.FINALIZING

    def named_parameters(self, prefix: str = ..., recurse: bool = ...):
        return self.model.named_parameters()

    def forward(self, *input):
        return self._model.forward(*input)

    def train(self, mode=True):
        self._status = Status.TRAINING
        self._model.train()

    def eval(self):
        self._status = Status.VALIDATING
        self._model.eval()

    def test(self):
        self._status = Status.TESTING
        self._model.eval()

    def step(self):
        if self._status is Status.TRAINING:
            if self._max_grad_norm:
                torch.nn.utils.clip_grad_norm_(self._model.parameters(), self._max_grad_norm)
            self._optimizer.step()

    def scheduler_step(self):
        if self._scheduler is not None:
            self._scheduler.step(self._train_loss.compute())

    def zero_grad(self):
        self._optimizer.zero_grad()

    def compute_and_update_train_loss(self, pred, target) -> Union[ApexLoss, torch.Tensor]:
        self._step_train_loss = self._criterion(pred, target)
        self._train_loss.update(self._step_train_loss)

        return ApexLoss(self._amp_id, self._step_train_loss, self._optimizer) if self.use_amp else self._step_train_loss

    def compute_and_update_valid_loss(self, pred, target) -> Union[ApexLoss, torch.Tensor]:
        self._step_valid_loss = self._criterion(pred, target)
        self._valid_loss.update(self._step_valid_loss)

        return ApexLoss(self._amp_id, self._step_valid_loss, self._optimizer) if self.use_amp else self._step_valid_loss

    def compute_and_update_test_loss(self, pred, target) -> Union[ApexLoss, torch.Tensor]:
        self._step_test_loss = self._criterion(pred, target)
        self._test_loss.update(self._step_test_loss)

        return ApexLoss(self._amp_id, self._step_test_loss, self._optimizer) if self.use_amp else self._step_test_loss

    def compute_loss(self, pred, target) -> Union[ApexLoss, torch.Tensor]:
        loss = self._criterion(pred, target)

        return ApexLoss(self._amp_id, loss, self._optimizer) if self.use_amp else loss

    def update_train_loss(self, loss):
        self._step_train_loss = float(loss) if not isinstance(loss, ApexLoss) else float(loss.loss)
        self._train_loss.update(self._step_train_loss)

    def update_valid_loss(self, loss):
        self._step_valid_loss = float(loss) if not isinstance(loss, ApexLoss) else float(loss.loss)
        self._valid_loss.update(self._step_valid_loss)

    def update_test_loss(self, loss):
        self._step_test_loss = float(loss) if not isinstance(loss, ApexLoss) else float(loss.loss)
        self._test_loss.update(self._step_test_loss)

    def compute_metric(self, pred, target):
        self._metric_computer.update((pred, target))
        metric = self._metric_computer.compute()
        self._metric_computer.reset()

        return metric

    def update_train_metric(self, metric):
        self._step_train_metric = metric
        self._train_metric.update(self._step_train_metric)

    def update_valid_metric(self, metric):
        self._step_valid_metric = metric
        self._valid_metric.update(self._step_valid_metric)

    def update_test_metric(self, metric):
        self._step_test_metric = metric
        self._test_metric.update(self._step_test_metric)

    def reset(self):
        self._metric_computer.reset()
        self._train_loss.reset()
        self._valid_loss.reset()
        self._test_loss.reset()
        self._train_metric.reset()
        self._valid_metric.reset()
        self._test_metric.reset()

    def finalize(self):
        self._status = Status.FINALIZING


class Trainer(EventGenerator):
    LOGGER = logging.getLogger("Trainer")

    def __init__(self, name, train_data_loader: Union[None, DataLoader], valid_data_loader: Union[None, DataLoader],
                 test_data_loader: Union[None, DataLoader], model_trainers: Union[List[ModelTrainer], ModelTrainer],
                 run_config: RunConfiguration = RunConfiguration()):
        super().__init__()
        self._name = name
        self._train_data_loader = train_data_loader
        self._valid_data_loader = valid_data_loader
        self._test_data_loader = test_data_loader
        self._model_trainers = model_trainers if isinstance(model_trainers, list) else [model_trainers]

        self._current_train_batch = 0
        self._current_valid_batch = 0
        self._current_test_batch = 0
        self._current_epoch = 0

        self._run_config = run_config

        self._custom_variables = {}

        for amp_id, model_trainer in enumerate(self._model_trainers):
            model_trainer.initialize(amp_id, len(self._model_trainers), self._run_config)

        self._status = Status.INITIALIZATION

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
    def current_test_step(self):
        return self._current_epoch * len(self._test_data_loader) + self._current_test_batch

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
    def test_step(self, inputs, target):
        raise NotImplementedError()

    @abstractmethod
    def scheduler_step(self):
        raise NotImplementedError()

    @property
    def state(self):
        return self

    @property
    def status(self):
        return self._status

    def is_active(self):
        return self._status is not Status.FINALIZING

    def _reset_model_trainers(self):
        for model_trainer in self._model_trainers:
            model_trainer.reset()

    def _at_least_one_model_is_active(self):
        return len(list(filter(lambda model_trainer: model_trainer.is_active(), self._model_trainers))) > 0

    def train(self, nb_epoch):
        self._on_training_begin()
        self.fire(Event.ON_TRAINING_BEGIN)

        for self._current_epoch in range(0, nb_epoch):
            if self._at_least_one_model_is_active():
                self._on_epoch_begin()
                self.fire(Event.ON_EPOCH_BEGIN)
                self._on_train_epoch_begin()
                self.fire(Event.ON_TRAIN_EPOCH_BEGIN)
                self._train_epoch()
                self._on_train_epoch_end()
                self.fire(Event.ON_TRAIN_EPOCH_END)
                self._on_validation_epoch_begin()
                self.fire(Event.ON_VALID_EPOCH_BEGIN)
                self._validate_epoch()
                self._on_validation_epoch_end()
                self.fire(Event.ON_VALID_EPOCH_END)
                self._on_test_epoch_begin()
                self.fire(Event.ON_TEST_EPOCH_BEGIN)
                self._test_epoch()
                self._on_test_epoch_end()
                self.fire(Event.ON_TEST_EPOCH_END)
                self.fire(Event.ON_TEST_EPOCH_END)
                self._on_epoch_end()
                self.fire(Event.ON_EPOCH_END)
            else:
                self.finalize()

        self._on_training_end()

    def _train_epoch(self):

        for model_trainer in self._model_trainers:
            model_trainer.train()

        for self._current_train_batch, (inputs, target) in enumerate(self._train_data_loader):
            self.on_train_batch_begin()

            self.fire(Event.ON_TRAIN_BATCH_BEGIN)

            inputs = [single_input.to(self._run_config.device, non_blocking=True) for single_input in
                      inputs] if isinstance(inputs, list) else inputs.to(self._run_config.device, non_blocking=True)

            target = [single_target.to(self._run_config.device, non_blocking=True) for single_target in
                      target] if isinstance(target, list) else target.to(self._run_config.device, non_blocking=True)

            self.train_step(inputs, target)
            self.on_train_batch_end()
            self.fire(Event.ON_TRAIN_BATCH_END)
            self.on_batch_end()
            self.fire(Event.ON_BATCH_END)

    def _validate_epoch(self):

        for model_trainer in self._model_trainers:
            model_trainer.eval()

        with torch.no_grad():
            for self._current_valid_batch, (inputs, target) in enumerate(self._valid_data_loader):
                self.on_valid_batch_begin()
                self.fire(Event.ON_VALID_BATCH_BEGIN)

                inputs = [single_input.to(self._run_config.device, non_blocking=True) for single_input in
                          inputs] if isinstance(inputs, list) else inputs.to(self._run_config.device, non_blocking=True)

                target = [single_target.to(self._run_config.device, non_blocking=True) for single_target in
                          target] if isinstance(target, list) else target.to(self._run_config.device, non_blocking=True)

                self.validate_step(inputs, target)
                self.on_valid_batch_end()
                self.fire(Event.ON_VALID_BATCH_END)
                self.on_batch_end()
                self.fire(Event.ON_BATCH_END)

            self._current_valid_batch = 0

    def _test_epoch(self):

        for model_trainer in self._model_trainers:
            model_trainer.eval()

        with torch.no_grad():
            for self._current_test_batch, (inputs, target) in enumerate(self._test_data_loader):
                self.on_test_batch_begin()
                self.fire(Event.ON_TEST_BATCH_BEGIN)

                inputs = [single_input.to(self._run_config.device, non_blocking=True) for single_input in
                          inputs] if isinstance(inputs, list) else inputs.to(self._run_config.device, non_blocking=True)

                target = [single_target.to(self._run_config.device, non_blocking=True) for single_target in
                          target] if isinstance(target, list) else target.to(self._run_config.device, non_blocking=True)

                self.test_step(inputs, target)
                self.on_test_batch_end()
                self.fire(Event.ON_TEST_BATCH_END)
                self.on_batch_end()
                self.fire(Event.ON_BATCH_END)

            self._current_valid_batch = 0
            self._current_test_batch = 0

    def _finalize(self):
        self._status = Status.FINALIZING
        self.finalize()
        self._status = Status.FINALIZED

    def with_event_handler(self, handler, event: BaseEvent):
        if event in self._event_handlers.keys():
            self._event_handlers[event].append(handler)
        else:
            self._event_handlers[event] = [handler]

        return self

    def _on_training_begin(self):
        self.on_training_begin()
        self._status = Status.READY

    def _on_training_end(self):
        self.on_training_end()

    def _on_epoch_begin(self):
        self._reset_model_trainers()
        self.on_epoch_begin()

    def _on_epoch_end(self):
        self.scheduler_step()
        self.on_epoch_end()

    def _on_train_epoch_begin(self):
        self._status = Status.TRAINING
        self.on_train_epoch_begin()

    def _on_train_epoch_end(self):
        self.on_train_epoch_end()

    def _on_validation_epoch_begin(self):
        self._status = Status.VALIDATING
        self.on_validation_epoch_begin()

    def _on_validation_epoch_end(self):
        self.on_validation_epoch_end()

    def _on_test_epoch_begin(self):
        self._status = Status.TESTING
        self.on_test_epoch_begin()

    def _on_test_epoch_end(self):
        self.on_test_epoch_end()


class SimpleTrainer(Trainer):

    def train_step(self, inputs, target):
        model = self._model_trainers[0]

        pred = model.forward(inputs)
        model.compute_train_metric(pred, target)
        loss = model.compute_and_update_train_loss(pred, target)

        model.zero_grad()
        loss.backward()
        model.step()

    def validate_step(self, inputs, target):
        model = self._model_trainers[0]

        pred = model.forward(inputs)
        model.compute_valid_metric(pred, target)
        model.compute_and_update_valid_loss(pred, target)

    def test_step(self, inputs, target):
        model = self._model_trainers[0]

        pred = model.forward(inputs)
        model.compute_valid_metric(pred, target)
        model.compute_and_update_valid_loss(pred, target)

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
        if not on_cpu(run_config.device):
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

        return ModelTrainer(model_trainer_config.model_name, model, criterion, optimizer, scheduler, metric,
                            model_trainer_config.max_grad_norm)
