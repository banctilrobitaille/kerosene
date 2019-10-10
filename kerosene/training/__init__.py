from enum import Enum


class Status(Enum):
    INITIALIZATION = "Initialization"
    READY = "Ready"
    TRAINING = "Training"
    VALIDATING = "Validating"
    TESTING = "Testing"
    FINALIZING = "Finalizing"
    FINALIZED = "Finalized"
