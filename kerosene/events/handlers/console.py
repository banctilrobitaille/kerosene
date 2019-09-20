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

from kerosene.events import BaseEvent, Event
from kerosene.events.handlers.base_handler import EventHandler
from kerosene.training.trainers import Trainer


class BaseConsoleLogger(EventHandler, ABC):
    LOGGER = logging.getLogger("ConsoleLogger")

    def __init__(self, every):
        super(BaseConsoleLogger, self).__init__(every)


class PrintTrainingStatus(BaseConsoleLogger):
    SUPPORTED_EVENTS = [Event.ON_BATCH_END, Event.ON_EPOCH_END, Event.ON_TRAIN_BATCH_END, Event.ON_VALID_BATCH_END]

    def __call__(self, event: BaseEvent, trainer: Trainer):
        assert event in self.SUPPORTED_EVENTS, "Unsupported event provided. Only {} are permitted.".format(
            self.SUPPORTED_EVENTS)

        if self.should_handle_epoch_data(event, trainer):
            self.print_status(trainer.epoch, trainer.current_train_step, trainer.current_valid_step)
        elif self.should_handle_step_data(event, trainer):
            self.print_status(trainer.epoch, trainer.current_train_step, trainer.current_valid_step)

    def print_status(self, epoch, train_step, valid_step):
        self.LOGGER.info("Training state: Epoch: {} | Training step: {} | Validation step: {} \n".format(
            epoch, train_step, valid_step))


class PrintModelTrainersStatus(BaseConsoleLogger):
    SUPPORTED_EVENTS = [Event.ON_BATCH_END, Event.ON_EPOCH_END]

    def __call__(self, event: BaseEvent, trainer: Trainer):
        assert event in self.SUPPORTED_EVENTS, "Unsupported event provided. Only {} are permitted.".format(
            self.SUPPORTED_EVENTS)
        status = "Model: {}, Train Loss: {}, Validation Loss: {},  Train Metric: {}, Valid Metric: {}"

        if self.should_handle_epoch_data(event, trainer):
            self.LOGGER.info("".join(list(map(
                lambda model_trainer: status.format(model_trainer.name, model_trainer.train_loss.item(),
                                                    model_trainer.valid_loss.item(),
                                                    model_trainer.train_metric.item(),
                                                    model_trainer.valid_metric.item()),
                trainer.model_trainers))))
        elif self.should_handle_step_data(event, trainer):
            self.LOGGER.info("".join(list(map(
                lambda model_trainer: status.format(model_trainer.name, model_trainer.step_train_loss.item(),
                                                    model_trainer.step_valid_loss.item(),
                                                    model_trainer.step_train_metric.item(),
                                                    model_trainer.step_valid_metric.item()),
                trainer.model_trainers))))
