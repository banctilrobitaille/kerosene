from enum import Enum


class BaseStatus(Enum):

    def __str__(self):
        return self.value

    def __eq__(self, other):
        if isinstance(other, str):
            return self.value == other
        elif isinstance(other, BaseStatus):
            return self.value == other.value


class Status(BaseStatus):
    INITIALIZED = "Initialization"
    TRAIN = "Training"
    VALID = "Validating"
    TEST = "Testing"
    FINALIZE = "Finalizing"

    def __hash__(self):
        return hash(self.value)
