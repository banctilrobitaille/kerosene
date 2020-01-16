from kerosene.events import TemporalEvent, Moment, Frequency, Phase, BaseEvent
from kerosene.training import Status


class Event(BaseEvent):
    ON_TRAINING_BEGIN = "training_begin"
    ON_TRAINING_END = "training_end"
    ON_VALID_BEGIN = "valid_begin"
    ON_VALID_END = "valid_end"
    ON_TEST_BEGIN = "test_begin"
    ON_TEST_END = "test_end"
    ON_EPOCH_BEGIN = "epoch_begin"
    ON_EPOCH_END = "epoch_end"
    ON_TRAIN_EPOCH_BEGIN = "train_epoch_begin"
    ON_TRAIN_EPOCH_END = "train_epoch_end"
    ON_VALID_EPOCH_BEGIN = "valid_epoch_begin"
    ON_VALID_EPOCH_END = "valid_epoch_end"
    ON_TEST_EPOCH_BEGIN = "test_epoch_begin"
    ON_TEST_EPOCH_END = "test_epoch_end"
    ON_BATCH_BEGIN = "batch_begin"
    ON_TRAIN_BATCH_BEGIN = "train_batch_begin"
    ON_TRAIN_BATCH_END = "train_batch_end"
    ON_VALID_BATCH_BEGIN = "valid_batch_begin"
    ON_VALID_BATCH_END = "valid_batch_end"
    ON_TEST_BATCH_BEGIN = "test_batch_begin"
    ON_TEST_BATCH_END = "test_batch_end"
    ON_BATCH_END = "batch_end"
    ON_FINALIZE = "finalizing"

    def __call__(self, moment: Moment):
        return TemporalEvent(self, moment)


class BatchEventPublisherMixin(object):
    @property
    def iteration_and_phase(self):
        if self.status is Status.TRAINING:
            iteration = self.current_train_step
            phase = Phase.TRAINING
        elif self.status is Status.VALIDATING:
            iteration = self.current_valid_step
            phase = Phase.VALIDATION
        else:
            iteration = self.current_test_step
            phase = Phase.TEST

        return iteration, phase

    def on_batch_begin(self):
        pass

    def on_batch_end(self):
        pass

    def on_train_batch_begin(self):
        pass

    def on_train_batch_end(self):
        pass

    def on_valid_batch_begin(self):
        pass

    def on_valid_batch_end(self):
        pass

    def on_test_batch_begin(self):
        pass

    def on_test_batch_end(self):
        pass

    def _on_batch_begin(self):
        self.on_batch_begin()

        iteration, phase = self.iteration_and_phase

        self.fire(Event.ON_BATCH_BEGIN(Moment(iteration, Frequency.STEP, phase)))

    def _on_batch_end(self):
        self.on_batch_end()

        iteration, phase = self.iteration_and_phase

        self.fire(Event.ON_BATCH_END(Moment(iteration, Frequency.STEP, phase)), self.step_monitors(phase))

    def _on_train_batch_begin(self):
        self.on_train_batch_begin()
        self.fire(Event.ON_TRAIN_BATCH_BEGIN(Moment(self.current_train_step, Frequency.STEP, Phase.TRAINING)))

    def _on_train_batch_end(self):
        self.on_train_batch_end()
        self.fire(Event.ON_TRAIN_BATCH_END(Moment(self.current_train_step, Frequency.STEP, Phase.TRAINING)),
                  self.step_monitors(Phase.TRAINING))

    def _on_valid_batch_begin(self):
        self.on_valid_batch_begin()
        self.fire(Event.ON_VALID_BATCH_BEGIN(Moment(self.current_valid_step, Frequency.STEP, Phase.VALIDATION)))

    def _on_valid_batch_end(self):
        self.on_valid_batch_begin()
        self.fire(Event.ON_VALID_BATCH_END(Moment(self.current_valid_step, Frequency.STEP, Phase.VALIDATION)),
                  self.step_monitors(Phase.VALIDATION))

    def _on_test_batch_begin(self):
        self.on_test_batch_begin()
        self.fire(Event.ON_TEST_BATCH_BEGIN(Moment(self.current_test_step, Frequency.STEP, Phase.TEST)))

    def _on_test_batch_end(self):
        self.on_test_batch_end()
        self.fire(Event.ON_TEST_BATCH_END(Moment(self.current_test_step, Frequency.STEP, Phase.TEST)),
                  self.step_monitors(Phase.TEST))


class EpochEventPublisherMixin(object):
    @property
    def epoch_and_phase(self):
        if self.status is Status.TRAINING:
            phase = Phase.TRAINING
        elif self.status is Status.VALIDATING:
            phase = Phase.VALIDATION
        else:
            phase = Phase.TEST

        return self.epoch, phase

    def on_epoch_begin(self):
        pass

    def on_epoch_end(self):
        pass

    def on_train_epoch_begin(self):
        pass

    def on_train_epoch_end(self):
        pass

    def on_valid_epoch_begin(self):
        pass

    def on_valid_epoch_end(self):
        pass

    def on_test_epoch_begin(self):
        pass

    def on_test_epoch_end(self):
        pass

    def _on_epoch_begin(self):
        self._reset_model_trainers()
        self.on_epoch_begin()

        epoch, phase = self.epoch_and_phase

        self.fire(Event.ON_EPOCH_BEGIN(Moment(epoch, Frequency.EPOCH, phase)))

    def _on_epoch_end(self):
        self.on_epoch_end()

        epoch, phase = self.epoch_and_phase

        self.fire(Event.ON_EPOCH_END(Moment(epoch, Frequency.EPOCH, phase)), self.epoch_monitors(phase))

    def _on_train_epoch_begin(self):
        self._status = Status.TRAINING
        self.on_train_epoch_begin()
        self.fire(Event.ON_TRAIN_EPOCH_BEGIN(Moment(self.epoch, Frequency.EPOCH, Phase.TRAINING)))

    def _on_train_epoch_end(self):
        self.on_train_epoch_end()
        self.fire(Event.ON_TRAIN_EPOCH_END(Moment(self.epoch, Frequency.EPOCH, Phase.TRAINING)),
                  self.epoch_monitors(Phase.TRAINING))

    def _on_valid_epoch_begin(self):
        self.on_valid_epoch_begin()
        self.fire(Event.ON_VALID_EPOCH_BEGIN(Moment(self.epoch, Frequency.EPOCH, Phase.VALIDATION)))

    def _on_valid_epoch_end(self):
        self._current_valid_batch = 0
        self.on_valid_epoch_end()
        self.fire(Event.ON_VALID_EPOCH_END(Moment(self.epoch, Frequency.EPOCH, Phase.VALIDATION)),
                  self.epoch_monitors(Phase.VALIDATION))

    def _on_test_epoch_begin(self):
        self.on_test_epoch_begin()
        self.fire(Event.ON_TEST_EPOCH_BEGIN(Moment(self.epoch, Frequency.EPOCH, Phase.TEST)))

    def _on_test_epoch_end(self):
        self._current_test_batch = 0
        self.on_test_epoch_end()
        self.fire(Event.ON_TEST_EPOCH_END(Moment(self.epoch, Frequency.STEP, Phase.TEST)),
                  self.epoch_monitors(Phase.TEST))


class TrainingPhaseEventPublisherMixin(object):
    def on_training_begin(self):
        pass

    def on_training_end(self):
        pass

    def on_valid_begin(self):
        pass

    def on_valid_end(self):
        pass

    def on_test_begin(self):
        pass

    def on_test_end(self):
        pass

    def finalize(self):
        pass

    def _on_training_begin(self):
        self.on_training_begin()
        self._status = Status.TRAINING
        self.fire(Event.ON_TRAINING_BEGIN(Moment(0, Frequency.PHASE, Phase.TRAINING)))

    def _on_training_end(self):
        self.on_training_end()
        self.scheduler_step()
        self.fire(Event.ON_TRAINING_END(Moment(0, Frequency.PHASE, Phase.TRAINING)))

    def _on_valid_begin(self):
        self.on_valid_begin()
        self._status = Status.VALIDATING
        self.fire(Event.ON_VALID_BEGIN(Moment(0, Frequency.PHASE, Phase.VALIDATION)))

    def _on_valid_end(self):
        self.on_valid_end()
        self.fire(Event.ON_VALID_END(Moment(0, Frequency.PHASE, Phase.VALIDATION)))

    def _on_test_begin(self):
        self.on_test_begin()
        self._status = Status.TESTING
        self.fire(Event.ON_TEST_BEGIN(Moment(0, Frequency.PHASE, Phase.TEST)))

    def _on_test_end(self):
        self.on_test_end()
        self.fire(Event.ON_TEST_END(Moment(0, Frequency.PHASE, Phase.TEST)))

    def _finalize(self):
        for model_trainer in self.model_trainers:
            model_trainer.finalize()
        self.finalize()
        self.status = Status.FINALIZED
