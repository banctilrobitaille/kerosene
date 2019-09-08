from kerosene.events import Event
from kerosene.events.preprocessors.base_preprocessor import EventPreprocessor
from kerosene.training.trainers import Trainer


class PrintTrainingStatus(EventPreprocessor):
    def __call__(self, event: Event, state: Trainer):
        return "Training state: Epoch: {} | Training step: {} | Validation step: {} \n".format(
            state.epoch, state.current_train_step, state.current_valid_step)
