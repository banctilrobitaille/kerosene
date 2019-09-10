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
import torch


class TrainerState(object):
    def __init__(self, name, train_data_loader, valid_data_loader, epoch=0, train_step=0, valid_step=0,
                 custom_variables=None, model_trainer_states=()):
        self._name = name
        self._epoch = epoch
        self._train_step = train_step
        self._valid_step = valid_step
        self._train_data_loader = train_data_loader
        self._valid_data_loader = valid_data_loader
        self._custom_variables = custom_variables
        self._model_trainer_states = model_trainer_states

    @property
    def name(self):
        return self._name

    @property
    def epoch(self):
        return self._epoch

    @property
    def train_step(self):
        return self._train_step

    @property
    def valid_step(self):
        return self._valid_step

    @property
    def train_data_loader(self):
        return self._train_data_loader

    @property
    def valid_data_loader(self):
        return self._valid_data_loader

    @property
    def custom_variables(self):
        return self._custom_variables

    @property
    def model_trainer_states(self):
        return self._model_trainer_states

    def with_epoch(self, epoch):
        self._epoch = epoch
        return self

    def with_train_step(self, train_step):
        self._train_step = train_step
        return self

    def with_valid_step(self, valid_step):
        self._valid_step = valid_step
        return self

    def with_custom_variables(self, custom_variables):
        self._custom_variables = custom_variables
        return self

    def with_model_trainers_state(self, model_trainers_state):
        self._model_trainer_states = model_trainers_state
        return self

    def __str__(self):
        return "Training state: Epoch: {} | Training step: {} | Validation step: {} \n".format(
            self._epoch, self._train_step, self._valid_step)


class ModelTrainerState(object):
    def __init__(self, name, train_loss=torch.tensor([0.0]), step_train_loss=torch.tensor([0.0]),
                 valid_loss=torch.tensor([0.0]), step_valid_loss=torch.tensor([0.0]),
                 train_metric=torch.tensor([0.0]), step_train_metric=torch.tensor([0.0]),
                 valid_metric=torch.tensor([0.0]), step_valid_metric=torch.tensor([0.0]),
                 model=None, optimizer=None):
        self._name = name
        self._train_loss = train_loss
        self._step_train_loss = step_train_loss

        self._valid_loss = valid_loss
        self._step_valid_loss = step_valid_loss

        self._train_metric = train_metric
        self._step_train_metric = step_train_metric

        self._valid_metric = valid_metric
        self._step_valid_metric = step_valid_metric

        self._model = model
        self._optimizer = optimizer

    @property
    def name(self):
        return self._name

    @property
    def train_loss(self):
        return self._train_loss

    @property
    def step_train_loss(self):
        return self._step_train_loss

    @property
    def valid_loss(self):
        return self._valid_loss

    @property
    def step_valid_loss(self):
        return self._step_valid_loss

    @property
    def train_metric(self):
        return self._train_metric

    @property
    def step_train_metric(self):
        return self._step_train_metric

    @property
    def valid_metric(self):
        return self._valid_metric

    @property
    def step_valid_metric(self):
        return self._step_valid_metric

    @property
    def model_state(self):
        return self._model.state_dict()

    @property
    def optimizer_state(self):
        return self._optimizer.state_dict()

    @property
    def optimizer_lr(self):
        for param_group in self._optimizer.param_groups:
            return torch.tensor([param_group["lr"]])

    def with_name(self, name):
        self._name = name
        return self

    def with_train_loss(self, train_loss):
        self._train_loss = train_loss
        return self

    def with_step_train_loss(self, step_train_loss):
        self._step_train_loss = step_train_loss
        return self

    def with_valid_loss(self, valid_loss):
        self._valid_loss = valid_loss
        return self

    def with_step_valid_loss(self, step_valid_loss):
        self._step_valid_loss = step_valid_loss
        return self

    def with_train_metric(self, train_metric):
        self._train_metric = train_metric
        return self

    def with_step_train_metric(self, step_train_metric):
        self._step_train_metric = step_train_metric
        return self

    def with_valid_metric(self, valid_metric):
        self._valid_metric = valid_metric
        return self

    def with_step_valid_metric(self, step_valid_metric):
        self._step_valid_metric = step_valid_metric
        return self
