from enum import Enum


class BaseEvent(Enum):
    def __str__(self):
        return self.value


class BaseVariable(Enum):
    def __str__(self):
        return self.value

    def __eq__(self, other):
        if isinstance(other, str):
            return self.value == other
        elif isinstance(other, BaseVariable):
            return self.value == other.value


class Event(BaseEvent):
    ON_TRAINING_BEGIN = "training_begin"
    ON_TRAINING_END = "training_end"
    ON_EPOCH_BEGIN = "epoch_begin"
    ON_TRAIN_EPOCH_BEGIN = "train_epoch_begin"
    ON_TRAIN_EPOCH_END = "train_epoch_end"
    ON_VALID_EPOCH_BEGIN = "valid_epoch_begin"
    ON_VALID_EPOCH_END = "valid_epoch_end"
    ON_100_TRAIN_STEPS = "100_train_steps"
    ON_EPOCH_END = "epoch_end"
    ON_TRAIN_BATCH_BEGIN = "train_batch_begin"
    ON_TRAIN_BATCH_END = "train_batch_end"
    ON_VALID_BATCH_BEGIN = "valid_batch_begin"
    ON_VALID_BATCH_END = "valid_batch_end"
    ON_BATCH_END = "batch_end"


class Monitor(BaseVariable):
    TRAINING_LOSS = "train_loss"
    TRAINING_METRIC = "train_metric"
    VALIDATION_LOSS = "valid_loss"
    VALIDATION_METRIC = "valid_metric"

    def is_loss(self):
        return "Loss" in self.value

    def is_metric(self):
        return "Metric" in self.value


class MonitorMode(Enum):
    MIN = -1
    MAX = 1
    AUTO = "auto"

    def __str__(self):
        return self.value
