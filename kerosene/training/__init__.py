from enum import Enum


class Status(Enum):
    INITIALIZED = "Initialization"
    TRAIN = "Training"
    VALID = "Validating"
    FINALIZE = "Finalizing"

    def __str__(self):
        return self.value