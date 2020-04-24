from enum import Enum

from examples.wcgan.models.discriminators import BasicBlock, ResNet
from examples.wcgan.models.generators import Unet
from kerosene.models.models import ModelFactory


class ModelType(Enum):
    GENERATOR = "generator"
    DISCRIMINATOR = "discriminator"

    def __str__(self):
        return self.value

    def __eq__(self, other):
        if isinstance(other, str):
            return self.value == other
        elif isinstance(other, ModelType):
            return self.value == other.value


class CustomModelFactory(ModelFactory):
    def create(self, model_type, params):
        if model_type == ModelType.GENERATOR:
            return Unet(**params)
        elif model_type == ModelType.DISCRIMINATOR:
            return ResNet(BasicBlock, [2, 2, 2, 2], **params)
        else:
            raise ValueError("The provided model type {} is invalid.".format(str(model_type)))
