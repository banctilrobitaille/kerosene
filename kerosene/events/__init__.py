from datetime import datetime
from enum import Enum


class Phase(Enum):
    TRAINING = "Training"
    VALIDATION = "Validation"
    TEST = "Test"
    ALL = [TRAINING, VALIDATION, TEST]

    def __str__(self):
        return self.value

    def __eq__(self, other):
        if isinstance(other, str):
            return self.value == other
        elif isinstance(other, Phase):
            return self.value == other.value

    def __hash__(self):
        return hash(self.value)


class Frequency(Enum):
    STEP = "Step"
    EPOCH = "Epoch"
    PHASE = "Phase"

    def __str__(self):
        return self.value

    def __eq__(self, other):
        if isinstance(other, str):
            return self.value == other
        elif isinstance(other, Frequency):
            return self.value == other.value

    def __hash__(self):
        return hash(self.value)


class Moment(object):
    def __init__(self, iteration, frequency: Frequency, phase: Phase, time: datetime = None):
        self._datetime = datetime.now() if time is None else time
        self._frequency = frequency
        self._iteration = iteration
        self._phase = phase

    @property
    def datetime(self):
        return self._datetime

    @datetime.setter
    def datetime(self, value):
        self._datetime = value

    @property
    def frequency(self):
        return self._frequency

    @frequency.setter
    def frequency(self, value):
        self._frequency = value

    @property
    def iteration(self):
        return self._iteration

    @iteration.setter
    def iteration(self, value):
        self._iteration = value

    @property
    def phase(self):
        return self._phase

    @phase.setter
    def phase(self, value):
        self._phase = value


class BaseEvent(Enum):
    def __init__(self, value):
        self._value_ = value

    def __str__(self):
        return self.value

    def __eq__(self, other):
        if isinstance(other, str):
            return self.value == other
        elif isinstance(other, BaseEvent):
            return self.value == other.value

    def __hash__(self):
        return hash(self.value)


class TemporalEvent(object):
    def __init__(self, event: BaseEvent, moment: Moment):
        self._event = event
        self._moment = moment

    @property
    def event(self):
        return self._event

    @property
    def frequency(self):
        return self._moment.frequency

    @property
    def phase(self):
        return self._moment.phase

    @property
    def datetime(self):
        return self._moment.datetime

    @property
    def iteration(self):
        return self._moment.iteration

    def __eq__(self, other):
        if isinstance(other, BaseEvent):
            return self.event == other
        elif isinstance(other, TemporalEvent):
            return self._event == other.event and self.frequency == other.frequency and \
                   self.phase == other.phase and self.datetime == other.datetime and self.iteration == other.iteration

    def __str__(self):
        return str(self.event)


class BaseVariable(Enum):
    def __str__(self):
        return self.value

    def __eq__(self, other):
        if isinstance(other, str):
            return self.value == other
        elif isinstance(other, BaseVariable):
            return self.value == other.value

    def __hash__(self):
        return hash(self.value)


class Monitor(BaseVariable):
    LOSS = "loss"
    METRICS = "metrics"

    TRAIN_LOSS = "train_loss"
    VALID_LOSS = "valid_loss"
    TEST_LOSS = "test_loss"
    TRAIN_METRICS = "train_metrics"
    VALID_METRICS = "valid_metrics"
    TEST_METRICS = "test_metrics"

    def is_loss(self):
        return "loss" in self.value

    def is_metrics(self):
        return "metrics" in self.value


class MonitorMode(Enum):
    MIN = -1
    MAX = 1

    def __str__(self):
        return self.value
