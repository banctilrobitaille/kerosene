import logging

from kerosene.events.handlers.base_handler import EventHandler


class ConsoleLogger(EventHandler):
    LOGGER = logging.getLogger("ConsoleLogger")

    def __call__(self, state):
        self.LOGGER.info("{}".format(str(state)))
