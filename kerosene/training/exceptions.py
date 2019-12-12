class InvalidDataSamplerTypeException(Exception):
    def __init__(self, dataloader_phase, dataloader_type):
        super().__init__(
            "{} dataloader's sampler must be instance of DistributedSampler when using more than one GPU, but found "
            "type {}.".format(dataloader_phase, dataloader_type))
