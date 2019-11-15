from enum import Enum


class BaseEvent(Enum):
    def __str__(self):
        return self.value

    def __eq__(self, other):
        if isinstance(other, str):
            return self.value == other
        elif isinstance(other, BaseEvent):
            return self.value == other.value


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

    EPOCH_EVENTS = [ON_EPOCH_BEGIN, ON_TRAIN_EPOCH_BEGIN, ON_TRAIN_EPOCH_END, ON_VALID_EPOCH_BEGIN, ON_VALID_EPOCH_END,
                    ON_EPOCH_END, ON_TEST_EPOCH_BEGIN, ON_TEST_BATCH_BEGIN, ON_TEST_BATCH_END]
    TRAIN_BATCH_EVENTS = [ON_BATCH_BEGIN, ON_TRAIN_BATCH_BEGIN, ON_TRAIN_BATCH_END, ON_BATCH_END]
    VALID_BATCH_EVENTS = [ON_BATCH_BEGIN, ON_VALID_BATCH_BEGIN, ON_VALID_BATCH_END, ON_BATCH_END]
    TEST_BATCH_EVENTS = [ON_BATCH_BEGIN, ON_TEST_BATCH_BEGIN, ON_TEST_BATCH_END, ON_BATCH_END]


class BaseVariable(Enum):
    def __str__(self):
        return self.value

    def __eq__(self, other):
        if isinstance(other, str):
            return self.value == other
        elif isinstance(other, BaseVariable):
            return self.value == other.value


class Monitor(BaseVariable):
    TRAINING_LOSS = "train_loss"
    TRAINING_METRIC = "train_metric"
    VALID_LOSS = "valid_loss"
    VALID_METRIC = "valid_metric"
    TEST_LOSS = "test_loss"
    TEST_METRIC = "test_metric"

    def is_loss(self):
        return "loss" in self.value

    def is_metric(self):
        return "metric" in self.value


class MonitorMode(Enum):
    MIN = -1
    MAX = 1
    AUTO = "auto"

    def __str__(self):
        return self.value
