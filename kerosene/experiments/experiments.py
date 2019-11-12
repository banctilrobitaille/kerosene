class Experiment(object):

    def __init__(self, name, trainer):
        self._name = name
        self._trainer = trainer

    @property
    def name(self):
        return self._name

    @property
    def trainer(self):
        return self._trainer

    def resume(self):
        pass

    def run(self):
        pass

    def save(self):
        pass

    def build(self):
        pass
