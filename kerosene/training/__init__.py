from enum import Enum


class Status(Enum):
    INITIALIZED = "Initialization"
    TRAIN = "Training"
    VALID = "Validating"
    TEST = "Testing"
    FINALIZE = "Finalizing"
