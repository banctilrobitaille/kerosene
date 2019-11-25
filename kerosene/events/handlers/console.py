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
from enum import Enum
from typing import Dict, Union

from kerosene.events import BaseEvent, Event
from kerosene.events.exceptions import UnsupportedEventException
from kerosene.events.handlers.base_handler import EventHandler
from kerosene.training import Status, BaseStatus
from kerosene.training.trainers import Trainer


class ConsoleColors(Enum):
    PURPLE = '\033[95m'
    BLUE = '\033[94m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    ENDC = '\033[0m'
    NONE = ""

    def __str__(self):
        return self.value


class StatusConsoleColorPalette(Dict[Status, ConsoleColors]):
    DEFAULT = {
        Status.TRAINING: ConsoleColors.GREEN,
        Status.VALIDATING: ConsoleColors.BLUE,
        Status.TESTING: ConsoleColors.PURPLE
    }


class BaseConsoleLogger(EventHandler, ABC):
    LOGGER = logging.getLogger("ConsoleLogger")

    def __init__(self, every=1):
        super(BaseConsoleLogger, self).__init__(every)


class ColoredConsoleLogger(BaseConsoleLogger, ABC):

    def __init__(self, every=1, colors: Dict[object, ConsoleColors] = None):
        super().__init__(every)
        self._colors = colors if colors is not None else {}

    def color(self, text, color_key):
        return str(self._colors.get(color_key, ConsoleColors.NONE)) + str(text) + str(ConsoleColors.ENDC)


class PrintTrainingStatus(ColoredConsoleLogger):
    SUPPORTED_EVENTS = [Event.ON_BATCH_END, Event.ON_EPOCH_END, Event.ON_TRAIN_BATCH_END, Event.ON_VALID_BATCH_END]

    def __init__(self, every=1, colors: Union[Dict[BaseStatus, ConsoleColors], StatusConsoleColorPalette] = None):
        super().__init__(every, colors)

    def __call__(self, event: BaseEvent, trainer: Trainer):
        if event not in self.SUPPORTED_EVENTS:
            raise UnsupportedEventException(event, self.SUPPORTED_EVENTS)

        if self.should_handle_epoch_data(event, trainer):
            self.print_status(trainer.status, trainer.epoch, trainer.current_train_step, trainer.current_valid_step)
        elif self.should_handle_train_step_data(event, trainer):
            self.print_status(trainer.status, trainer.epoch, trainer.current_train_step, trainer.current_valid_step)
        elif self.should_handle_valid_step_data(event, trainer):
            self.print_status(trainer.status, trainer.epoch, trainer.current_train_step, trainer.current_valid_step)

    def print_status(self, status, epoch, train_step, valid_step):
        self.LOGGER.info(
            "\nCurrent state: {} |  Epoch: {} | Training step: {} | Validation step: {} \n".format(
                self.color(status, color_key=status), epoch, train_step, valid_step))


class PrintModelTrainersStatus(BaseConsoleLogger):
    SUPPORTED_EVENTS = [Event.ON_BATCH_END, Event.ON_EPOCH_END]

    def __call__(self, event: BaseEvent, trainer: Trainer):
        if event not in self.SUPPORTED_EVENTS:
            raise UnsupportedEventException(event, self.SUPPORTED_EVENTS)

        status = "\nModel: {}, Train Loss: {}, Validation Loss: {},  Train Metric: {}, Valid Metric: {} \n"

        if self.should_handle_epoch_data(event, trainer):
            self.LOGGER.info("".join(list(map(
                lambda model_trainer: status.format(model_trainer.name,
                                                    model_trainer.train_loss.item(),
                                                    model_trainer.valid_loss.item(),
                                                    model_trainer.train_metrics.item(),
                                                    model_trainer.valid_metrics.item()),
                trainer.model_trainers))))
        elif self.should_handle_train_step_data(event, trainer):
            self.LOGGER.info("".join(list(map(
                lambda model_trainer: status.format(model_trainer.name,
                                                    model_trainer.step_train_loss.item(),
                                                    model_trainer.step_valid_loss.item(),
                                                    model_trainer.step_train_metrics.item(),
                                                    model_trainer.step_valid_metrics.item()),
                trainer.model_trainers))))
        elif self.should_handle_valid_step_data(event, trainer):
            self.LOGGER.info("".join(list(map(
                lambda model_trainer: status.format(model_trainer.name,
                                                    model_trainer.step_train_loss.item(),
                                                    model_trainer.step_valid_loss.item(),
                                                    model_trainer.step_train_metrics.item(),
                                                    model_trainer.step_valid_metrics.item()),
                trainer.model_trainers))))
