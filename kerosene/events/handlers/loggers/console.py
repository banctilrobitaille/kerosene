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
from abc import ABC

from kerosene.events import BaseEvent
from kerosene.events.handlers.base_handler import EventHandler
from kerosene.training.trainers import Trainer


class BaseConsoleLogger(EventHandler, ABC):
    LOGGER = logging.getLogger("ConsoleLogger")


class PrintTrainingStatus(BaseConsoleLogger):
    def __call__(self, event: BaseEvent, trainer: Trainer):
        return self.LOGGER.info("Training state: Epoch: {} | Training step: {} | Validation step: {} \n".format(
            trainer.epoch, trainer.current_train_step, trainer.current_valid_step))
