from blinker import signal

ON_TRAINING_BEGIN = signal("OnTrainingBegin")
ON_TRAINING_END = signal("OnTrainingEnd")
ON_EPOCH_BEGIN = signal("OnEpochBegin")
ON_TRAIN_EPOCH_BEGIN = signal("OnTrainEpochBegin")
ON_TRAIN_EPOCH_END = signal("OnTrainEpochEnd")
ON_VALID_EPOCH_BEGIN = signal("OnValidEpochBegin")
ON_VALID_EPOCH_END = signal("OnValidEpochEnd")
ON_EPOCH_END = signal("OnEpochEnd")
ON_TRAIN_BATCH_BEGIN = signal("OnTrainBatchBegin")
ON_TRAIN_BATCH_END = signal("OnTrainBatchEnd")
ON_VALID_BATCH_BEGIN = signal("OnValidBatchBegin")
ON_VALID_BATCH_END = signal("OnValidBatchEnd")
