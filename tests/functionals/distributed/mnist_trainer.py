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

import os
from typing import List

import numpy as np
import torch
from kerosene.config.trainers import RunConfiguration
from kerosene.training.trainers import ModelTrainer
from kerosene.training.trainers import Trainer
from torch.utils.data import DataLoader


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
        loss = model.compute_train_loss(pred, target)

        model.zero_grad()
        loss.backward()
        model.step()

    def validate_step(self, inputs, target):
        pass

    def scheduler_step(self):
        pass

    @staticmethod
    def merge_tensors(tensor_0, tensor_1):
        return torch.cat((tensor_0, tensor_1), dim=0)

    @staticmethod
    def _reduce_tensor(tensor):
        rt = tensor.clone()
        torch.distributed.all_reduce(rt, op=torch.distributed.ReduceOp.SUM)
        rt /= int(os.environ['WORLD_SIZE'])
        return rt

    def on_epoch_begin(self):
        pass

    def on_epoch_end(self):
        pass
