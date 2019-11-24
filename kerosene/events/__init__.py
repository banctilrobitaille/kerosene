from enum import Enum


class BaseEvent(Enum):
    def __str__(self):
        return self.value

    def __eq__(self, other):
        if isinstance(other, str):
            return self.value == other
        elif isinstance(other, BaseEvent):
            return self.value == other.value

    def __hash__(self):
        return hash(self.value)


class Event(BaseEvent):
    ON_TRAINING_BEGIN = "training_begin"
    ON_TRAINING_END = "training_end"
    ON_VALID_BEGIN = "valid_begin"
    ON_VALID_END = "valid_end"
    ON_TEST_BEGIN = "test_begin"
    ON_TEST_END = "test_end"
    ON_EPOCH_BEGIN = "epoch_begin"
    ON_EPOCH_END = "epoch_end"
    ON_TRAIN_EPOCH_BEGIN = "train_epoch_begin"
    ON_TRAIN_EPOCH_END = "train_epoch_end"
    ON_VALID_EPOCH_BEGIN = "valid_epoch_begin"
    ON_VALID_EPOCH_END = "valid_epoch_end"
    ON_TEST_EPOCH_BEGIN = "test_epoch_begin"
    ON_TEST_EPOCH_END = "test_epoch_end"
    ON_BATCH_BEGIN = "batch_begin"
    ON_TRAIN_BATCH_BEGIN = "train_batch_begin"
    ON_TRAIN_BATCH_END = "train_batch_end"
    ON_VALID_BATCH_BEGIN = "valid_batch_begin"
    ON_VALID_BATCH_END = "valid_batch_end"
    ON_TEST_BATCH_BEGIN = "test_batch_begin"
    ON_TEST_BATCH_END = "test_batch_end"
    ON_BATCH_END = "batch_end"
    ON_FINALIZE = "finalizing"

    @staticmethod
    def epoch_events():
        return [Event.ON_EPOCH_BEGIN, Event.ON_TRAIN_EPOCH_BEGIN, Event.ON_TRAIN_EPOCH_END, Event.ON_VALID_EPOCH_BEGIN,
                Event.ON_VALID_EPOCH_END, Event.ON_EPOCH_END, Event.ON_TEST_EPOCH_BEGIN, Event.ON_TEST_BATCH_BEGIN,
                Event.ON_TEST_BATCH_END]

    @staticmethod
    def train_batch_events():
        return [Event.ON_BATCH_BEGIN, Event.ON_TRAIN_BATCH_BEGIN, Event.ON_TRAIN_BATCH_END, Event.ON_BATCH_END]

    @staticmethod
    def valid_batch_events():
        return [Event.ON_BATCH_BEGIN, Event.ON_VALID_BATCH_BEGIN, Event.ON_VALID_BATCH_END, Event.ON_BATCH_END]

    @staticmethod
    def test_batch_events():
        return [Event.ON_BATCH_BEGIN, Event.ON_TEST_BATCH_BEGIN, Event.ON_TEST_BATCH_END, Event.ON_BATCH_END]


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
