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
from typing import Dict, Union, List, Optional

import torch
from ignite.metrics import Metric
from torch.utils.data import DataLoader

from kerosene.configs.configs import ModelConfiguration, RunConfiguration
from kerosene.events import BaseEvent, Monitor, Frequency, Phase
from kerosene.events.publishers.base_publisher import EventPublisher
from kerosene.metrics.gauges import AverageGauge
from kerosene.metrics.metrics import MetricFactory, to_tensor
from kerosene.models.models import ModelFactory
from kerosene.nn.apex import ApexModule, ApexLoss
from kerosene.nn.criterions import CriterionFactory
from kerosene.nn.utils.gradients import GradientClippingStrategy, GradientClippingStrategyFactory
from kerosene.optim.optimizers import OptimizerFactory
from kerosene.optim.schedulers import SchedulerFactory
from kerosene.training import Status
from kerosene.training.events import BatchEventPublisherMixin, EpochEventPublisherMixin, \
    TrainingPhaseEventPublisherMixin
from kerosene.utils.devices import on_gpu


class ModelTrainer(ApexModule):
    LOGGER = logging.getLogger("ModelTrainer")

    def __init__(self, model_name, model, criterions, optimizer, scheduler, metric_computers: Dict[str, Metric],
                 gradient_clipping_strategy: GradientClippingStrategy = None):
        super(ModelTrainer, self).__init__(model, optimizer)
        self._model_name = model_name

        self._criterions = criterions
        self._scheduler = scheduler
        self._metric_computers = metric_computers
        self._gradient_clipping_strategy = gradient_clipping_strategy

        self._step_train_loss = {criterion: torch.Tensor().new_zeros((1,)) for criterion in criterions.keys()}
        self._step_valid_loss = {criterion: torch.Tensor().new_zeros((1,)) for criterion in criterions.keys()}
        self._step_test_loss = {criterion: torch.Tensor().new_zeros((1,)) for criterion in criterions.keys()}
        self._step_train_metrics = {metric: torch.Tensor().new_zeros((1,)) for metric in metric_computers.keys()}
        self._step_valid_metrics = {metric: torch.Tensor().new_zeros((1,)) for metric in metric_computers.keys()}
        self._step_test_metrics = {metric: torch.Tensor().new_zeros((1,)) for metric in metric_computers.keys()}

        self._train_loss = {criterion: AverageGauge() for criterion in criterions.keys()}
        self._valid_loss = {criterion: AverageGauge() for criterion in criterions.keys()}
        self._test_loss = {criterion: AverageGauge() for criterion in criterions.keys()}
        self._train_metrics = {metric_name: AverageGauge() for metric_name in metric_computers.keys()}
        self._valid_metrics = {metric_name: AverageGauge() for metric_name in metric_computers.keys()}
        self._test_metrics = {metric_name: AverageGauge() for metric_name in metric_computers.keys()}

        self._status = Status.INITIALIZED

    @property
    def name(self):
        return self._model_name

    @property
    def step_train_loss(self):
        return {loss_name: to_tensor(step_train_loss).cpu() for loss_name, step_train_loss in
                self._step_train_loss.items()}

    @property
    def step_valid_loss(self):
        return {loss_name: to_tensor(step_valid_loss).cpu() for loss_name, step_valid_loss in
                self._step_valid_loss.items()}

    @property
    def step_test_loss(self):
        return {loss_name: to_tensor(step_test_loss).cpu() for loss_name, step_test_loss in
                self._step_test_loss.items()}

    @property
    def step_train_metrics(self):
        return {metric_name: to_tensor(step_train_metric).cpu() for metric_name, step_train_metric in
                self._step_train_metrics.items()}

    @property
    def step_valid_metrics(self):
        return {metric_name: to_tensor(step_valid_metric).cpu() for metric_name, step_valid_metric in
                self._step_valid_metrics.items()}

    @property
    def step_test_metrics(self):
        return {metric_name: to_tensor(step_test_metric).cpu() for metric_name, step_test_metric in
                self._step_test_metrics.items()}

    @property
    def train_loss(self):
        return {loss_name: to_tensor(train_loss.compute()).cpu() for loss_name, train_loss in
                self._train_loss.items()}

    @property
    def valid_loss(self):
        return {loss_name: to_tensor(train_loss.compute()).cpu() for loss_name, train_loss in
                self._valid_loss.items()}

    @property
    def test_loss(self):
        return {loss_name: to_tensor(train_loss.compute()).cpu() for loss_name, train_loss in
                self._test_loss.items()}

    @property
    def train_metrics(self):
        return {metric_name: to_tensor(train_metric.compute()).cpu() for metric_name, train_metric in
                self._train_metrics.items()}

    @property
    def valid_metrics(self):
        return {metric_name: to_tensor(valid_metric.compute()).cpu() for metric_name, valid_metric in
                self._valid_metrics.items()}

    @property
    def test_metrics(self):
        return {metric_name: to_tensor(test_metric.compute()).cpu() for metric_name, test_metric in
                self._test_metrics.items()}

    @property
    def optimizer_state(self):
        return self._optimizer.state_dict()

    @property
    def optimizer_lr(self):
        for param_group in self._optimizer.param_groups:
            return torch.tensor([param_group["lr"]])

    @optimizer_lr.setter
    def optimizer_lr(self, lr):
        for param_group in self._optimizer.param_groups:
            param_group["lr"] = lr

    @property
    def model_state(self):
        return self._model.state_dict()

    @property
    def criterions(self):
        return self._criterions

    @property
    def scheduler(self):
        return self._scheduler

    def step_monitors(self):
        return {Phase.TRAINING: {Monitor.METRICS: self.step_train_metrics,
                                 Monitor.LOSS: {name: value.item() for name, value in self.step_train_loss.items()}},
                Phase.VALIDATION: {Monitor.METRICS: self.step_valid_metrics,
                                   Monitor.LOSS: {name: value.item() for name, value in self.step_valid_loss.items()}},
                Phase.TEST: {Monitor.METRICS: self.step_test_metrics,
                             Monitor.LOSS: {name: value.item() for name, value in self.step_test_loss.items()}}}

    def epoch_monitors(self):
        return {Phase.TRAINING: {Monitor.METRICS: self.train_metrics, Monitor.LOSS: self.train_loss},
                Phase.VALIDATION: {Monitor.METRICS: self.valid_metrics, Monitor.LOSS: self.valid_loss},
                Phase.TEST: {Monitor.METRICS: self.test_metrics, Monitor.LOSS: self.test_loss}}

    def to_state_dict(self, frequency: Frequency = Frequency.EPOCH,
                      training_phase: Union[List[Phase], Phase] = Phase.ALL):
        training_phase = [training_phase] if not isinstance(training_phase, list) else training_phase
        monitors = self.step_monitors() if frequency == Frequency.STEP else self.epoch_monitors()

        return {"Model": self.model_state,
                "Optimizer": self.optimizer_state,
                "Monitors": dict(map(lambda phase: (phase, monitors[phase]), training_phase))}

    def is_active(self) -> bool:
        return self._status is not Status.FINALIZED

    def named_parameters(self, prefix: str = ..., recurse: bool = ...) -> tuple:
        return self.model.named_parameters()

    def forward(self, *input):
        return self._model.forward(*input)

    def train(self, mode=True) -> None:
        self._status = Status.TRAINING
        self._model.train()

    def eval(self) -> None:
        self._status = Status.VALIDATING
        self._model.eval()

    def test(self) -> None:
        self._status = Status.TESTING
        self._model.eval()

    def step(self):
        if self._status is Status.TRAINING:
            if self._should_clip_gradients():
                self._gradient_clipping_strategy.clip(self._model.parameters())
            self._optimizer.step()

    def scheduler_step(self) -> None:
        if self._scheduler is not None:
            self._scheduler.step(sum(list(map(lambda loss: loss.compute(), self._train_loss.values()))))

    def to(self, device: Optional[Union[int, torch.device]] = ..., dtype=..., non_blocking: bool = ...):
        self.model.to(device)

        for criterion in self.criterions.values():
            criterion.to(device)

    def zero_grad(self) -> None:
        self._optimizer.zero_grad()

    def freeze(self) -> None:
        for param in self.parameters():
            param.requires_grad = False

    def unfreeze(self) -> None:
        for param in self.parameters():
            param.requires_grad = True

    def compute_and_update_train_loss(self, name, pred, target) -> Union[ApexLoss, torch.Tensor]:
        self._step_train_loss[name] = self._criterions[name](pred, target)
        self._train_loss[name].update(self._step_train_loss[name].item())

        return ApexLoss(self._amp_id, self._step_train_loss[name], self._optimizer) if self.use_amp else \
            self._step_train_loss[name]

    def compute_and_update_train_losses(self, pred, target) -> Dict[str, Union[ApexLoss, torch.Tensor]]:
        losses = {}
        for name, criterion in self._criterions.items():
            self._step_train_loss[name] = criterion(pred, target)
            self._train_loss[name].update(self._step_train_loss[name].item())
            losses[name] = ApexLoss(self._amp_id, self._step_train_loss[name],
                                    self._optimizer) if self.use_amp else self._step_train_loss[name]
        return losses

    def compute_and_update_valid_loss(self, name, pred, target) -> Union[ApexLoss, torch.Tensor]:
        self._step_valid_loss[name] = self._criterions[name](pred, target)
        self._valid_loss[name].update(self._step_valid_loss[name].item())

        return ApexLoss(self._amp_id, self._step_valid_loss[name], self._optimizer) if self.use_amp else \
            self._step_valid_loss[name]

    def compute_and_update_valid_losses(self, pred, target) -> Dict[str, Union[ApexLoss, torch.Tensor]]:
        losses = {}
        for name, criterion in self._criterions.items():
            self._step_valid_loss[name] = criterion(pred, target)
            self._valid_loss[name].update(self._step_valid_loss[name].item())
            losses[name] = ApexLoss(self._amp_id, self._step_valid_loss[name],
                                    self._optimizer) if self.use_amp else self._step_valid_loss[name]
        return losses

    def compute_and_update_test_loss(self, name, pred, target) -> Union[ApexLoss, torch.Tensor]:
        self._step_test_loss[name] = self._criterions[name](pred, target)
        self._test_loss[name].update(self._step_test_loss[name].item())

        return ApexLoss(self._amp_id, self._step_test_loss[name], self._optimizer) if self.use_amp else \
            self._step_test_loss[name]

    def compute_and_update_test_losses(self, pred, target) -> Dict[str, Union[ApexLoss, torch.Tensor]]:
        losses = {}
        for name, criterion in self._criterions.items():
            self._step_test_loss[name] = criterion(pred, target)
            self._test_loss[name].update(self._step_test_loss[name].item())
            losses[name] = ApexLoss(self._amp_id, self._step_test_loss[name],
                                    self._optimizer) if self.use_amp else self._step_test_loss[name]
        return losses

    def compute_loss(self, name, pred, target) -> Union[ApexLoss, torch.Tensor]:
        loss = self._criterions[name](pred, target)

        return ApexLoss(self._amp_id, loss, self._optimizer) if self.use_amp else loss

    def compute_losses(self, pred, target) -> Dict[str, Union[ApexLoss, torch.Tensor]]:
        losses = {}
        for name, criterion in self._criterions.items():
            loss = criterion(pred, target)
            losses[name] = ApexLoss(self._amp_id, loss, self._optimizer) if self.use_amp else loss
        return losses

    def update_train_loss(self, name, loss) -> None:
        self._step_train_loss[name] = loss if not isinstance(loss, ApexLoss) else loss.loss
        self._train_loss[name].update(self._step_train_loss[name].item())

    def update_train_losses(self, losses) -> None:
        for name, value in losses.items():
            self._step_train_loss[name] = value if not isinstance(value, ApexLoss) else value.loss
            self._train_loss[name].update(self._step_train_loss[name].item())

    def update_valid_loss(self, name, loss) -> None:
        self._step_valid_loss[name] = loss if not isinstance(loss, ApexLoss) else loss.loss
        self._valid_loss[name].update(self._step_valid_loss[name].item())

    def update_valid_losses(self, losses) -> None:
        for name, value in losses.items():
            self._step_valid_loss[name] = value if not isinstance(value, ApexLoss) else value.loss
            self._valid_loss[name].update(self._step_valid_loss[name].item())

    def update_test_loss(self, name, loss) -> None:
        self._step_test_loss[name] = loss if not isinstance(loss, ApexLoss) else loss.loss
        self._test_loss[name].update(self._step_test_loss[name].item())

    def update_test_losses(self, losses) -> None:
        for name, value in losses.items():
            self._step_test_loss[name] = value if not isinstance(value, ApexLoss) else value.loss
            self._test_loss[name].update(self._step_test_loss[name].item())

    def compute_metrics(self, pred, target):
        metrics = dict()
        for metric_name, metric_computer in self._metric_computers.items():
            metric_computer.update((pred, target))
            metric = metric_computer.compute()
            metric_computer.reset()
            metrics[metric_name] = metric

        return metrics

    def update_train_metrics(self, metrics: dict):
        self._step_train_metrics = metrics
        for train_metric, step_train_metric in zip(list(self._train_metrics.values()),
                                                   list(self._step_train_metrics.values())):
            train_metric.update(step_train_metric)

    def update_valid_metrics(self, metric: dict):
        self._step_valid_metrics = metric
        for valid_metric, step_valid_metric in zip(list(self._valid_metrics.values()),
                                                   list(self._step_valid_metrics.values())):
            valid_metric.update(step_valid_metric)

    def update_test_metrics(self, metric: dict):
        self._step_test_metrics = metric
        for test_metric, step_test_metric in zip(list(self._test_metrics.values()),
                                                 list(self._step_test_metrics.values())):
            test_metric.update(step_test_metric)

    def reset(self):
        [metric_computer.reset() for metric_computer in self._metric_computers.values()]
        [metric.reset() for metric_name, metric in self._train_metrics.items()]
        [metric.reset() for metric_name, metric in self._valid_metrics.items()]
        [metric.reset() for metric_name, metric in self._test_metrics.items()]
        [loss.reset() for loss_name, loss in self._train_loss.items()]
        [loss.reset() for loss_name, loss in self._valid_loss.items()]
        [loss.reset() for loss_name, loss in self._test_loss.items()]

    def finalize(self) -> None:
        self._status = Status.FINALIZED

    def _should_clip_gradients(self):
        return self._gradient_clipping_strategy is not None


class Trainer(BatchEventPublisherMixin, EpochEventPublisherMixin, TrainingPhaseEventPublisherMixin, EventPublisher):
    LOGGER = logging.getLogger("Trainer")

    def __init__(self, name, train_data_loader: DataLoader, valid_data_loader: DataLoader,
                 test_data_loader: Union[DataLoader, None], model_trainers: Union[List[ModelTrainer], ModelTrainer],
                 run_config: RunConfiguration):
        super().__init__()
        self._name = name

        self._train_data_loader = train_data_loader
        self._valid_data_loader = valid_data_loader
        self._test_data_loader = test_data_loader

        self._model_trainers = model_trainers if isinstance(model_trainers, list) else [model_trainers]
        self._device = run_config.device

        self._current_train_batch = 0
        self._current_valid_batch = 0
        self._current_test_batch = 0
        self._current_epoch = 0

        self._custom_variables = {}

        for amp_id, model_trainer in enumerate(self._model_trainers):
            if on_gpu(self._device):
                model_trainer.to(self._device)
            model_trainer.initialize(amp_id, len(self._model_trainers), run_config.use_amp, run_config.amp_opt_level,
                                     self._device)

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
    def test_data_loader(self):
        return self._test_data_loader

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
        return self._current_epoch * len(
            self._test_data_loader) + self._current_test_batch if self._test_data_loader is not None else 0

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

    @property
    def nb_on_test_batch(self):
        return len(self._test_data_loader)

    @property
    def sender(self):
        return self

    @property
    def status(self):
        return self._status

    @status.setter
    def status(self, value):
        self._status = value

    def step_monitors(self, phase: Phase):
        if phase != Phase.ALL:
            monitors = dict(
                map(lambda model: (model.name, {phase: model.step_monitors()[phase]}), self._model_trainers))
        else:
            monitors = dict(
                map(lambda model: (model.name, model.step_monitors()), self._model_trainers))

        return monitors

    def epoch_monitors(self, phase: Phase):
        if phase != Phase.ALL:
            monitors = dict(
                map(lambda model: (model.name, {phase: model.epoch_monitors()[phase]}), self._model_trainers))
        else:
            monitors = dict(
                map(lambda model: (model.name, model.epoch_monitors()), self._model_trainers))

        return monitors

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

    def is_active(self):
        return self._status is not Status.FINALIZED

    def train(self, nb_epoch):
        self._on_training_begin()

        for self._current_epoch in range(0, nb_epoch):
            if self._at_least_one_model_is_active():
                self._on_training_begin()
                self._on_epoch_begin()
                self._on_train_epoch_begin()
                self._train_epoch()
                self._on_train_epoch_end()
                self._on_epoch_end()
                self._on_training_end()
                self._on_valid_begin()
                self._on_epoch_begin()
                self._on_valid_epoch_begin()
                self._validate_epoch()
                self._on_valid_epoch_end()
                self._on_epoch_end()
                self._on_valid_end()
                self._on_test_begin()
                self._on_epoch_begin()
                self._on_test_epoch_begin()
                self._test_epoch()
                self._on_test_epoch_end()
                self._on_epoch_end()
                self._on_test_end()
            else:
                self._finalize()

    def with_event_handler(self, handler, event: BaseEvent):
        if event in self._event_handlers.keys():
            self._event_handlers[event].append(handler)
        else:
            self._event_handlers[event] = [handler]

        return self

    def _train_epoch(self):
        for model_trainer in self._model_trainers:
            model_trainer.train()

        for self._current_train_batch, (inputs, target) in enumerate(self._train_data_loader):
            self._on_batch_begin()
            self._on_train_batch_begin()

            inputs = [single_input.to(self._device, non_blocking=True) for single_input in
                      inputs] if isinstance(inputs, list) else inputs.to(self._device, non_blocking=True)

            target = [single_target.to(self._device, non_blocking=True) for single_target in
                      target] if isinstance(target, list) else target.to(self._device, non_blocking=True)

            self.train_step(inputs, target)
            self._on_train_batch_end()
            self._on_batch_end()

    def _validate_epoch(self):
        for model_trainer in self._model_trainers:
            model_trainer.eval()

        with torch.no_grad():
            for self._current_valid_batch, (inputs, target) in enumerate(self._valid_data_loader):
                self._on_valid_batch_begin()

                inputs = [single_input.to(self._device, non_blocking=True) for single_input in
                          inputs] if isinstance(inputs, list) else inputs.to(self._device, non_blocking=True)

                target = [single_target.to(self._device, non_blocking=True) for single_target in
                          target] if isinstance(target, list) else target.to(self._device, non_blocking=True)

                self.validate_step(inputs, target)
                self._on_valid_batch_end()
                self._on_batch_end()

    def _test_epoch(self):
        if self._test_data_loader is not None:
            for model_trainer in self._model_trainers:
                model_trainer.eval()

            with torch.no_grad():
                for self._current_test_batch, (inputs, target) in enumerate(self._test_data_loader):
                    self._on_test_batch_begin()

                    inputs = [single_input.to(self._device, non_blocking=True) for single_input in
                              inputs] if isinstance(inputs, list) else inputs.to(self._device, non_blocking=True)

                    target = [single_target.to(self._device, non_blocking=True) for single_target in
                              target] if isinstance(target, list) else target.to(self._device, non_blocking=True)

                    self.test_step(inputs, target)
                    self._on_test_batch_end()
                    self._on_batch_end()

    def _reset_model_trainers(self):
        for model_trainer in self._model_trainers:
            model_trainer.reset()

    def _at_least_one_model_is_active(self):
        return any(model_trainer.is_active for model_trainer in self._model_trainers)


class SimpleTrainer(Trainer):

    def train_step(self, inputs, target):
        model = self._model_trainers[0]

        pred = model.forward(inputs)
        metric = model.compute_metrics(pred, target)
        model.update_train_metrics(metric)
        losses = model.compute_and_update_train_losses(pred, target)

        model.zero_grad()
        for loss in losses.values():
            loss.backward()
        model.step()

    def validate_step(self, inputs, target):
        model = self._model_trainers[0]

        pred = model.forward(inputs)
        metric = model.compute_metrics(pred, target)
        model.update_valid_metrics(metric)
        model.compute_and_update_valid_losses(pred, target)

    def test_step(self, inputs, target):
        model = self._model_trainers[0]

        pred = model.forward(inputs)
        metric = model.compute_metrics(pred, target)
        model.update_test_metrics(metric)
        model.compute_and_update_test_losses(pred, target)

    def scheduler_step(self):
        self._model_trainers[0].scheduler_step()


class ModelTrainerFactory(object):
    def __init__(self, model=None, model_factory: ModelFactory = None, optimizer_factory=OptimizerFactory(),
                 scheduler_factory=SchedulerFactory(), criterion_factory=CriterionFactory(),
                 metric_factory=MetricFactory(), gradient_clipping_strategy_factory=GradientClippingStrategyFactory()):
        self._model = model
        self._model_factory = model_factory
        self._optimizer_factory = optimizer_factory
        self._scheduler_factory = scheduler_factory
        self._criterion_factory = criterion_factory
        self._metric_factory = metric_factory
        self._gradient_clipping_strategy_factory = gradient_clipping_strategy_factory

        assert (self._model is not None) or (
                self._model_factory is not None), "A model or a model factory must be provided !"

    def create(self, model_trainer_configs: Union[List[ModelConfiguration], ModelConfiguration]):
        model_trainer_configs = [model_trainer_configs] if isinstance(model_trainer_configs,
                                                                      ModelConfiguration) else model_trainer_configs
        model_trainers = list(
            map(lambda model_trainer_config: self._create(model_trainer_config), model_trainer_configs))

        return model_trainers if len(model_trainers) > 1 else model_trainers[0]

    def _create(self, model_config: ModelConfiguration):
        model = self._model if self._model is not None else self._model_factory.create(model_config.type,
                                                                                       model_config.params)
        optimizer = self._optimizer_factory.create(model_config.optimizer_type,
                                                   model.parameters(),
                                                   model_config.optimizer_params)
        scheduler = self._scheduler_factory.create(model_config.scheduler_type, optimizer,
                                                   model_config.scheduler_params)

        criterions = {
            criterion_config.name: self._criterion_factory.create(criterion_config.type, criterion_config.params) for
            criterion_config in model_config.criterions_configs}

        metrics = {
            metric_config.name: self._metric_factory.create(metric_config.type, metric_config.params) for
            metric_config in model_config.metrics_configs}

        gradient_clipping_strategy = self._gradient_clipping_strategy_factory.create(
            model_config.gradient_clipping_strategy, model_config.gradient_clipping_params)

        return ModelTrainer(model_config.name, model, criterions, optimizer, scheduler, metrics,
                            gradient_clipping_strategy)
