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
from abc import ABC
from typing import Union, List

from kerosene.events import TemporalEvent, Monitor, BaseEvent
from kerosene.events.handlers.base_handler import EventHandler
from kerosene.loggers.visdom import PlotFrequency, PlotType
from kerosene.loggers.visdom.visdom import VisdomLogger, VisdomData
from kerosene.training.events import Event
from kerosene.training.trainers import Trainer, ModelTrainer


class BaseVisdomHandler(EventHandler, ABC):
    def __init__(self, supported_events: List[Union[BaseEvent, TemporalEvent]], visdom_logger: VisdomLogger, every=1):
        super().__init__(supported_events, every)
        self._visdom_logger = visdom_logger

    @property
    def visdom_logger(self):
        return self._visdom_logger

    def create_visdom_data(self, *args, **kwargs):
        raise NotImplementedError()

    def flatten(self, list_of_visdom_data):
        return [item for sublist in list_of_visdom_data for item in sublist]


class PlotMonitors(BaseVisdomHandler):
    SUPPORTED_EVENTS = [Event.ON_EPOCH_END, Event.ON_TRAIN_BATCH_END, Event.ON_VALID_BATCH_END, Event.ON_TEST_BATCH_END,
                        Event.ON_BATCH_END]

    def __init__(self, visdom_logger: VisdomLogger, every=1):
        super().__init__(self.SUPPORTED_EVENTS, visdom_logger, every)
        self._plot_losses = PlotLosses(visdom_logger, every)
        self._plot_metrics = PlotMetrics(visdom_logger, every)

    def __call__(self, event: TemporalEvent, monitors: dict, trainer: Trainer):
        if self.should_handle(event):
            self._plot_losses(event, monitors, trainer)
            self._plot_metrics(event, monitors, trainer)

    def create_visdom_data(self, event, model_name, monitors):
        data = self._plot_losses.create_visdom_data(event, model_name, monitors)
        data.extend(self._plot_metrics.create_visdom_data(event, model_name, monitors))


class PlotLosses(BaseVisdomHandler):
    SUPPORTED_EVENTS = [Event.ON_EPOCH_END, Event.ON_TRAIN_BATCH_END, Event.ON_VALID_BATCH_END, Event.ON_TEST_BATCH_END,
                        Event.ON_BATCH_END]

    def __init__(self, visdom_logger: VisdomLogger, every=1):
        super().__init__(self.SUPPORTED_EVENTS, visdom_logger, every)

    def __call__(self, event: TemporalEvent, monitors: dict, trainer: Trainer):
        data = None

        if self.should_handle(event):
            for model_name, monitor in monitors.items():
                data = self.create_visdom_data(event, model_name, monitor[event.phase][Monitor.LOSS])

        if data is not None:
            self.visdom_logger(data)

    def create_visdom_data(self, event, model_name, monitors):
        return [VisdomData(model_name, "Loss", PlotType.LINE_PLOT, event.frequency,
                           [[event.iteration]], [[monitors]],
                           params={'opts': {'xlabel': str(event.frequency),
                                            'ylabel': "Loss",
                                            'title': "{} {} per {}".format(model_name, "Loss", str(event.frequency)),
                                            'name': str(event.phase), 'legend': [str(event.phase)]}})]


class PlotMetrics(BaseVisdomHandler):
    SUPPORTED_EVENTS = [Event.ON_EPOCH_END, Event.ON_TRAIN_BATCH_END, Event.ON_VALID_BATCH_END, Event.ON_TEST_BATCH_END,
                        Event.ON_BATCH_END]

    def __init__(self, visdom_logger: VisdomLogger, every=1):
        super().__init__(self.SUPPORTED_EVENTS, visdom_logger, every)

    def __call__(self, event: TemporalEvent, monitors: dict, trainer: Trainer):
        data = None

        if self.should_handle(event):
            for model_name, monitor in monitors.items():
                data = self.create_visdom_data(event, model_name, monitor[event.phase][Monitor.METRICS])

        if data is not None:
            self.visdom_logger(data)

    def create_visdom_data(self, event, model_name, monitors):
        return [VisdomData(model_name, metric_name, PlotType.LINE_PLOT, event.frequency, [[event.iteration]],
                           [[metric_value]], params={'opts': {'xlabel': str(event.frequency), 'ylabel': metric_name,
                                                              'title': "{} {} per {}".format(model_name, metric_name,
                                                                                             str(event.frequency)),
                                                              'name': str(event.phase), 'legend': [str(event.phase)]}})
                for metric_name, metric_value in monitors.items()]


class PlotCustomVariables(BaseVisdomHandler):
    SUPPORTED_EVENTS = [Event.ON_EPOCH_END, Event.ON_TRAIN_BATCH_END, Event.ON_VALID_BATCH_END, Event.ON_TEST_BATCH_END,
                        Event.ON_BATCH_END]

    def __init__(self, visdom_logger: VisdomLogger, variable_name, plot_type: PlotType, params, every=1):
        super().__init__(self.SUPPORTED_EVENTS, visdom_logger, every)
        self._variable_name = variable_name
        self._plot_type = plot_type
        self._params = params

    def __call__(self, event: TemporalEvent, monitors: dict, trainer: Trainer):
        data = None

        if self.should_handle(event):
            pass

        if self.should_handle_epoch_data(event, trainer.epoch):
            data = self.create_epoch_visdom_data(trainer)
        elif self.should_handle_step_data(event, trainer.current_train_step):
            data = self.create_train_batch_visdom_data(trainer)
        elif self.should_handle_step_data(event, trainer.current_valid_step):
            data = self.create_valid_batch_visdom_data(trainer)

        if data is not None:
            self.visdom_logger(data)

    def create_visdom_data(self):
        pass

    def create_epoch_visdom_data(self, trainer: Trainer):
        return [VisdomData(trainer.name, self._variable_name, self._plot_type, PlotFrequency.EVERY_EPOCH,
                           [trainer.epoch], trainer.custom_variables[self._variable_name], self._params)]

    def create_train_batch_visdom_data(self, trainer: Trainer):
        return [VisdomData(trainer.name, self._variable_name, self._plot_type, PlotFrequency.EVERY_STEP,
                           [trainer.current_train_step], trainer.custom_variables[self._variable_name], self._params)]

    def create_valid_batch_visdom_data(self, trainer: Trainer):
        return [VisdomData(trainer.name, self._variable_name, self._plot_type, PlotFrequency.EVERY_STEP,
                           [trainer.current_valid_step], trainer.custom_variables[self._variable_name], self._params)]


class PlotLR(BaseVisdomHandler):
    SUPPORTED_EVENTS = [Event.ON_EPOCH_END, Event.ON_TRAIN_BATCH_END, Event.ON_VALID_BATCH_END, Event.ON_TEST_BATCH_END,
                        Event.ON_BATCH_END]

    def __init__(self, visdom_logger: VisdomLogger, every=1):
        super().__init__(self.SUPPORTED_EVENTS, visdom_logger, every)

    def __call__(self, event: TemporalEvent, monitors: dict, trainer: Trainer):
        data = None

        if self.should_handle(event):
            data = list(map(
                lambda model_trainer: self.create_visdom_data(event, model_trainer), trainer.model_trainers))

        if data is not None:
            self.visdom_logger(data)

    def create_visdom_data(self, event, model_trainer: ModelTrainer):
        return VisdomData(model_trainer.name, "Learning Rate", PlotType.LINE_PLOT, event.frequency, [event.iteration],
                          model_trainer.optimizer_lr,
                          params={'opts': {'xlabel': str(event.frequency),
                                           'ylabel': "Learning Rate",
                                           'title': "{} {} per {}".format(model_trainer.name, "Learning Rate",
                                                                          str(event.frequency)),
                                           'name': model_trainer.name,
                                           'legend': [model_trainer.name]}})


class PlotGradientFlow(BaseVisdomHandler):
    SUPPORTED_EVENTS = [Event.ON_TRAIN_BATCH_END]

    def __init__(self, visdom_logger: VisdomLogger, every=1):
        super().__init__(self.SUPPORTED_EVENTS, visdom_logger, every)

    def __call__(self, event: TemporalEvent, monitors: dict, trainer: Trainer):
        data = None

        if self.should_handle_step_data(event, trainer.current_train_step):
            data = list(map(lambda model_trainer: self.create_visdom_data(model_trainer), trainer.model_trainers))

        if data is not None:
            self.visdom_logger(data)

    def create_visdom_data(self, model_trainer: ModelTrainer):
        avg_grads, layers = [], []

        for n, p in model_trainer.named_parameters():
            if p.requires_grad and ("bias" not in n):
                layers.append(n)
                if p.grad is not None:
                    avg_grads.append(p.grad.abs().mean().item())
                else:
                    avg_grads.append(0)

        return VisdomData(model_trainer.name, "Gradient Flow", PlotType.BAR_PLOT, PlotFrequency.EVERY_EPOCH, y=layers,
                          x=avg_grads, params={'opts': {'xlabel': "Layers", 'ylabel': "Avg. Gradients",
                                                        'title': "{} {} per {}".format(model_trainer.name,
                                                                                       "Gradient Flow",
                                                                                       "Layer"),
                                                        'marginbottom': 200}})
