from abc import ABC

from kerosene.events.handlers.base_handler import EventHandler
from kerosene.events.handlers.loggers.visdom.visdom import VisdomLogger


class BaseVisdomHandler(EventHandler, ABC):
    def __init__(self, visdom_logger: VisdomLogger, every_n=1):
        self._visdom_logger = visdom_logger
        self._every_n = every_n

    @property
    def visdom_logger(self):
        return self._visdom_logger

    @property
    def every_n(self):
        return self._every_n

    def flatten(self, list_of_visdom_data):
        return [item for sublist in list_of_visdom_data for item in sublist]
