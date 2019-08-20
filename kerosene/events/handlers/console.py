import logging

from kerosene.events.handlers.base_handler import EventHandler


class ConsoleLogger(EventHandler):
    LOGGER = logging.getLogger("ConsoleLogger")

    def __call__(self, msg):
        self.LOGGER.info("{}".format(str(msg)))
