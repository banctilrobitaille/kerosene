from enum import Enum


class BaseStatus(Enum):

    def __str__(self):
        return self.value

    def __eq__(self, other):
        if isinstance(other, str):
            return self.value == other
        elif isinstance(other, BaseStatus):
            return self.value == other.value

    def __hash__(self):
        return hash(self.value)


class Status(BaseStatus):
    INITIALIZING = "Initializing"
    INITIALIZED = "Initialized"
    READY = "Ready"
    TRAINING = "Training"
    VALIDATING = "Validating"
    TESTING = "Testing"
    FINALIZED = "Finalized"
