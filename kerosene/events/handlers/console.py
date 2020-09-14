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
import math
from abc import ABC
from enum import Enum
from typing import Dict, Union, List, Optional

from kerosene.events import Phase, TemporalEvent, BaseEvent, Monitor
from kerosene.events.handlers.base_handler import EventHandler
from kerosene.training import Status, BaseStatus
from kerosene.training.events import Event
from kerosene.training.trainers import Trainer
import crayons
from beautifultable import BeautifulTable


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


class PrintMonitorsTable(BaseConsoleLogger):
    SUPPORTED_EVENTS = [Event.ON_BATCH_END, Event.ON_EPOCH_END, Event.ON_TRAIN_BATCH_END, Event.ON_VALID_BATCH_END,
                        Event.ON_TEST_BATCH_END]

    def __init__(self, every=1):
        super().__init__(self.SUPPORTED_EVENTS, every)
        self._monitors = {}
        self._monitors_tables = {}

    def __call__(self, event: TemporalEvent, monitors: dict, trainer: Trainer):
        if self.should_handle(event):
            for model, monitor in monitors.items():
                training_values = {**monitors[model][Phase.TRAINING][Monitor.METRICS],
                                   **monitors[model][Phase.TRAINING][Monitor.LOSS]}
                validation_values = {**monitors[model][Phase.VALIDATION][Monitor.METRICS],
                                     **monitors[model][Phase.VALIDATION][Monitor.LOSS]}
                test_values = {**monitors[model][Phase.TEST][Monitor.METRICS],
                               **monitors[model][Phase.TEST][Monitor.LOSS]}

                if model in self._monitors_tables:
                    self._monitors_tables[model].update(training_values, validation_values, test_values)
                else:
                    self._monitors_tables[model] = MonitorsTable(model)
                    self._monitors_tables[model].update(training_values, validation_values, test_values)

                self._monitors_tables[model].show()


class MonitorsTable(object):
    def __init__(self, model_name: str):
        self.model_name = model_name
        self._training_monitors = {}
        self._validation_monitors = {}
        self._test_monitors = {}
        self.table = BeautifulTable(80)

    def append(self, values: dict, old_values: dict):
        self.table.rows.append(list(map(lambda key: self.color(values[key], old_values.get(key, None)), values.keys())))

    def update(self, training_monitors: dict, validation_monitors: dict, test_monitors: Optional[dict] = None):
        self.table.clear()
        self.append(training_monitors, self._training_monitors)
        self.append(validation_monitors, self._validation_monitors)

        if test_monitors is not None:
            self.append(test_monitors, self._test_monitors)
            self.table.rows.header = ["Training", "Validation", "Test"]
        else:
            self.table.rows.header = ["Training", "Validation"]

        self.table.columns.header = training_monitors.keys()

        self._training_monitors = training_monitors
        self._validation_monitors = validation_monitors
        self._test_monitors = test_monitors

    def show(self):
        self.table._compute_width()
        topline = "".join(["+", "-" * (self.table._width + 11), "+"])
        print(topline)
        spc = (len(topline) - 2 - len(self.model_name)) / 2
        print("%s%s%s%s%s" % ("|", " " * math.ceil(spc), self.model_name, " " * math.floor(spc), "|"))
        print(self.table)

    def color(self, value, old_value):
        if old_value is not None:
            if value > old_value:
                return crayons.green("{} \u2197".format(value), bold=True)
            if value < old_value:
                return crayons.red("{} \u2198".format(value), bold=True)

        return crayons.white("{} \u2192".format(value), bold=True)
