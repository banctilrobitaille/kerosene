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

from kerosene.events import BaseEvent, Event
from kerosene.events.handlers.base_handler import EventHandler
from kerosene.training.trainers import Trainer
from kerosene.training import Status


class BaseConsoleLogger(EventHandler, ABC):
    LOGGER = logging.getLogger("ConsoleLogger")

    def __init__(self, every):
        super(BaseConsoleLogger, self).__init__(every)


class ConsoleColors(object):
    PURPLE = '\033[95m'
    BLUE = '\033[94m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


class PrintTrainingStatus(BaseConsoleLogger):
    SUPPORTED_EVENTS = [Event.ON_BATCH_END, Event.ON_EPOCH_END, Event.ON_TRAIN_BATCH_END, Event.ON_VALIDATION_BATCH_END]

    def __call__(self, event: BaseEvent, trainer: Trainer):
        assert event in self.SUPPORTED_EVENTS, "Unsupported event provided. Only {} are permitted.".format(
            self.SUPPORTED_EVENTS)

        if self.should_handle_epoch_data(event, trainer):
            self.print_status(trainer.status, trainer.epoch, trainer.current_train_step, trainer.current_valid_step,
                              trainer.current_test_step)
        elif self.should_handle_train_step_data(event, trainer):
            self.print_status(trainer.status, trainer.epoch, trainer.current_train_step, trainer.current_valid_step,
                              trainer.current_test_step)
        elif self.should_handle_validation_step_data(event, trainer):
            self.print_status(trainer.status, trainer.epoch, trainer.current_train_step, trainer.current_valid_step,
                              trainer.current_test_step)
        elif self.should_handle_test_step_data(event, trainer):
            self.print_status(trainer.status, trainer.epoch, trainer.current_train_step, trainer.current_valid_step,
                              trainer.current_test_step)

    def print_status(self, status, epoch, train_step, valid_step, test_step):
        if status == Status.TRAINING:
            status = ConsoleColors.GREEN + str(status) + ConsoleColors.ENDC
        elif status == Status.VALIDATING:
            status = ConsoleColors.BLUE + str(status) + ConsoleColors.ENDC
        elif status == Status.TESTING:
            status = ConsoleColors.YELLOW + str(status) + ConsoleColors.ENDC
        self.LOGGER.info(
            "\nCurrent state: {} |  Epoch: {} | Training step: {} | Validation step: {} | Test step: {} \n".format(
                status, epoch, train_step, valid_step, test_step))


class PrintModelTrainersStatus(BaseConsoleLogger):
    SUPPORTED_EVENTS = [Event.ON_BATCH_END, Event.ON_EPOCH_END]

    def __call__(self, event: BaseEvent, trainer: Trainer):
        assert event in self.SUPPORTED_EVENTS, "Unsupported event provided. Only {} are permitted.".format(
            self.SUPPORTED_EVENTS)
        status = "\nModel: {}, Train Loss: {}, Validation Loss: {}, Test Loss: {}, Train Metric: {}, Valid Metric: {}, Test Metric: {} \n"

        if self.should_handle_epoch_data(event, trainer):
            self.LOGGER.info("".join(list(map(
                lambda model_trainer: status.format(model_trainer.name,
                                                    model_trainer.train_loss.item(),
                                                    model_trainer.valid_loss.item(),
                                                    model_trainer.test_loss.item(),
                                                    model_trainer.train_metric.item(),
                                                    model_trainer.valid_metric.item(),
                                                    model_trainer.test_metric.item()),
                trainer.model_trainers))))
        elif self.should_handle_train_step_data(event, trainer):
            self.LOGGER.info("".join(list(map(
                lambda model_trainer: status.format(model_trainer.name,
                                                    model_trainer.step_train_loss.item(),
                                                    model_trainer.step_valid_loss.item(),
                                                    model_trainer.step_test_loss.item(),
                                                    model_trainer.step_train_metric.item(),
                                                    model_trainer.step_valid_metric.item(),
                                                    model_trainer.step_test_metric.item()),
                trainer.model_trainers))))
        elif self.should_handle_validation_step_data(event, trainer):
            self.LOGGER.info("".join(list(map(
                lambda model_trainer: status.format(model_trainer.name,
                                                    model_trainer.step_train_loss.item(),
                                                    model_trainer.step_valid_loss.item(),
                                                    model_trainer.step_test_loss.item(),
                                                    model_trainer.step_train_metric.item(),
                                                    model_trainer.step_valid_metric.item(),
                                                    model_trainer.step_test_metric.item()),
                trainer.model_trainers))))
        elif self.should_handle_test_step_data(event, trainer):
            self.LOGGER.info("".join(list(map(
                lambda model_trainer: status.format(model_trainer.name,
                                                    model_trainer.step_train_loss.item(),
                                                    model_trainer.step_valid_loss.item(),
                                                    model_trainer.step_test_loss.item(),
                                                    model_trainer.step_train_metric.item(),
                                                    model_trainer.step_valid_metric.item(),
                                                    model_trainer.step_test_metric.item()),
                trainer.model_trainers))))
