from abc import ABC, abstractmethod


class Gauge(ABC):

    @abstractmethod
    def update(self, **kwargs):
        raise NotImplementedError()

    @abstractmethod
    def compute(self):
        raise NotImplementedError()

    @abstractmethod
    def reset(self):
        raise NotImplementedError()


class AverageGauge(Gauge):

    def __init__(self):
        self._value = 0
        self._sum = 0
        self._count = 0

    def update(self, value, n=1):
        self._value = value
        self._sum += value * n
        self._count += n

    def compute(self):
        return self._sum / self._count if self._count != 0 else 0

    def reset(self):
        self._value = 0
        self._sum = 0
        self._count = 0
