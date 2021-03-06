import torch

DELTA = 0.0001
ZERO = torch.tensor([0.0])
ONE = torch.tensor([1.0])
TARGET_CLASS_0 = torch.tensor([0])
TARGET_CLASS_1 = torch.tensor([1])
MODEL_PREDICTION_CLASS_0 = torch.tensor([[1.0, 0.0]])
MODEL_PREDICTION_CLASS_1 = torch.tensor([[0.0, 1.0]])
MINIMUM_BINARY_CROSS_ENTROPY_LOSS = torch.tensor(0.3133)
MAXIMUM_BINARY_CROSS_ENTROPY_LOSS = torch.tensor(1.3133)
AVERAGED_BINARY_CROSS_ENTROPY_LOSS = (MINIMUM_BINARY_CROSS_ENTROPY_LOSS + MAXIMUM_BINARY_CROSS_ENTROPY_LOSS) / 2
