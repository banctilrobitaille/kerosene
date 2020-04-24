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
from typing import Dict, Union, List

from kerosene.events import Phase, TemporalEvent, BaseEvent
from kerosene.events.handlers.base_handler import EventHandler
from kerosene.training import Status, BaseStatus
from kerosene.training.events import Event
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
        Phase.TRAINING: ConsoleColors.GREEN,
        Phase.VALIDATION: ConsoleColors.BLUE,
        Phase.TEST: ConsoleColors.PURPLE
    }


class BaseConsoleLogger(EventHandler, ABC):
    LOGGER = logging.getLogger("ConsoleLogger")

    def __init__(self, supported_events: List[Union[BaseEvent, TemporalEvent]] = None, every=1):
        super(BaseConsoleLogger, self).__init__(supported_events, every)


class ColoredConsoleLogger(BaseConsoleLogger, ABC):

    def __init__(self, supported_events: List[Union[BaseEvent, TemporalEvent]] = None, every=1,
                 colors: Dict[object, ConsoleColors] = None):
        super().__init__(supported_events, every)
        self._colors = colors if colors is not None else {}

    def color(self, text, color_key):
        return str(self._colors.get(color_key, ConsoleColors.NONE)) + str(text) + str(ConsoleColors.ENDC)


class PrintTrainingStatus(ColoredConsoleLogger):
    SUPPORTED_EVENTS = [Event.ON_BATCH_END, Event.ON_EPOCH_END, Event.ON_TRAIN_BATCH_END, Event.ON_VALID_BATCH_END,
                        Event.ON_TEST_BATCH_END]

    def __init__(self, every=1, colors: Union[Dict[BaseStatus, ConsoleColors], StatusConsoleColorPalette] = None):
        super().__init__(self.SUPPORTED_EVENTS, every, colors)

    def __call__(self, event: TemporalEvent, monitors: dict, trainer: Trainer):
        if self.should_handle(event):
            self.print_status(event.phase, trainer.epoch, trainer.current_train_step, trainer.current_valid_step,
                              trainer.current_test_step)

    def print_status(self, status, epoch, train_step, valid_step, test_step):
        self.LOGGER.info(
            "\nCurrent states: {} |  Epoch: {} | Training step: {} | Validation step: {} | Test step: {}\n".format(
                self.color(status, color_key=status), epoch, train_step, valid_step, test_step))


class PrintMonitors(BaseConsoleLogger):
    SUPPORTED_EVENTS = [Event.ON_BATCH_END, Event.ON_EPOCH_END, Event.ON_TRAIN_BATCH_END, Event.ON_VALID_BATCH_END,
                        Event.ON_TEST_BATCH_END]

    def __init__(self, every=1):
        super().__init__(self.SUPPORTED_EVENTS, every)
        self._monitors = {}

    def __call__(self, event: TemporalEvent, monitors: dict, trainer: Trainer):
        if self.should_handle(event):
            for model, monitor in monitors.items():
                if model in self._monitors:
                    self._monitors[model].update(monitor)
                else:
                    self._monitors[model] = monitor

                self.LOGGER.info("Model {}, {}".format(model, self._monitors[model]))
