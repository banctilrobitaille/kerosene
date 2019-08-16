from kerosene.events.handlers.base_handler import EventHandler


class Experiment(EventHandler):
    def __init__(self, name):
        self._name = name

    def __call__(self, *inputs):
        pass
