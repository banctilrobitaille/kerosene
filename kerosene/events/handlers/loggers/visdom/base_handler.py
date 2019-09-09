from abc import ABC

from kerosene.events import Event
from kerosene.events.handlers.base_handler import EventHandler
from kerosene.events.handlers.loggers.visdom.visdom import VisdomLogger


class BaseVisdomHandler(EventHandler, ABC):
    def __init__(self, visdom_logger: VisdomLogger, every=1):
        self._visdom_logger = visdom_logger
        self._every = every

    @property
    def visdom_logger(self):
        return self._visdom_logger

    @property
    def every(self):
        return self._every

    def should_plot_epoch_data(self, event, epoch):
        return (event in [Event.ON_EPOCH_BEGIN, Event.ON_EPOCH_END]) and (epoch % self._every == 0)

    def should_plot_step_data(self, event, step):
        return (event in [Event.ON_TRAIN_BATCH_BEGIN, Event.ON_TRAIN_BATCH_END, Event.ON_VALID_BATCH_BEGIN,
                          Event.ON_VALID_BATCH_END]) and (step % self._every == 0)

    def flatten(self, list_of_visdom_data):
        return [item for sublist in list_of_visdom_data for item in sublist]
