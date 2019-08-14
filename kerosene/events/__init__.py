from enum import Enum

from blinker import signal


class Event(Enum):
    ON_TRAINING_BEGIN = "training_begin"
    ON_TRAINING_END = "training_end"
    ON_EPOCH_BEGIN = "epoch_begin"
    ON_TRAIN_EPOCH_BEGIN = "train_epoch_begin"
    ON_TRAIN_EPOCH_END = signal("train_epoch_end")
    ON_VALID_EPOCH_BEGIN = signal("valid_epoch_begin")
    ON_VALID_EPOCH_END = signal("valid_epoch_end")
    ON_EPOCH_END = signal("epoch_end")
    ON_TRAIN_BATCH_BEGIN = signal("train_batch_begin")
    ON_TRAIN_BATCH_END = signal("train_batch_end")
    ON_VALID_BATCH_BEGIN = signal("valid_batch_begin")
    ON_VALID_BATCH_END = signal("valid_batch_end")
