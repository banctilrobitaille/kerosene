from enum import Enum


class Event(Enum):
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


class Monitor(Enum):
    TRAINING_LOSS = "TrainingLoss"
    TRAINING_METRIC = "TrainingMetric"
    VALIDATION_LOSS = "ValidationLoss"
    VALIDATION_METRIC = "ValidationMetric"


class MonitorMode(Enum):
    MIN = "min"
    MAX = "max"
