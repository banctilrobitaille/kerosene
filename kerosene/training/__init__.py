from enum import Enum


class Status(Enum):
    INITIALIZATION = "Initialization"
    READY = "Ready"
    TRAINING = "Training"
    VALIDATING = "Validating"
    TESTING = "Testing"
    FINALIZING = "Finalizing"
    FINALIZED = "Finalized"

    def __str__(self):
        return self.value
