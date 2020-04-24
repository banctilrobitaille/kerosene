# -*- coding: utf-8 -*-
# Copyright 2019 Pierre-Luc Delisle. All Rights Reserved.
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

from typing import List

import torch
from torch.utils.data import DataLoader

from kerosene.configs.configs import RunConfiguration
from kerosene.training.trainers import ModelTrainer
from kerosene.training.trainers import Trainer
from kerosene.utils.devices import on_single_device


class MNISTTrainer(Trainer):

    def __init__(self, training_config, model_trainers: List[ModelTrainer],
                 train_data_loader: DataLoader, valid_data_loader: DataLoader, run_config: RunConfiguration):
        super(MNISTTrainer, self).__init__("MNISTTrainer", train_data_loader, valid_data_loader, model_trainers,
                                           run_config)

        self._training_config = training_config

    def train_step(self, inputs, target):
        model = self._model_trainers[0]

        pred = model.forward(inputs)
        model.compute_train_metric(pred, target)
        loss = model.compute_and_update_train_loss(pred, target)

        model.zero_grad()
        loss.backward()

        if not on_single_device(self._run_config.devices):
            self.average_gradients(model)

        model.step()

    def validate_step(self, inputs, target):
        model = self._model_trainers[0]

        pred = model.forward(inputs)
        model.compute_valid_metric(pred, target)
        model.compute_and_update_valid_loss(pred, target)

    def scheduler_step(self):
        self._model_trainers[0].scheduler_step()

    @staticmethod
    def average_gradients(model):
        size = float(torch.distributed.get_world_size())
        for param in model.parameters():
            torch.distributed.all_reduce(param.grad.data, op=torch.distributed.ReduceOp.SUM)
            param.grad.data /= size

    def on_epoch_begin(self):
        pass

    def on_epoch_end(self):
        pass
