import torch


def to_onehot(indices: torch.Tensor, num_classes: int) -> torch.Tensor:
    """
    Convert a tensor of indices of any shape `(N, ...)` to a tensor of one-hot indicators of shape
    `(N, num_classes, ...)`.

    Args:
        indices (:obj:`torch.Tensor`): Tensor of indices.
        num_classes (int): The number of classes of the problem.

    Returns:
        :obj:`torch.Tensor`: The one-hot representation of the input tensor.
    """
    onehot = torch.zeros(indices.shape[0], num_classes, *indices.shape[1:], device=indices.device)
    return onehot.scatter_(1, indices.unsqueeze(1), 1)


def flatten(tensor: torch.Tensor) -> torch.Tensor:
    """
    Flattens a given tensor such that the channel axis is first.
    The shapes are transformed as follows: (N, C, D, H, W) -> (C, N * D * H * W)

    Args:
        tensor (:obj:`torch.Tensor`): Tensor to flatten.

    Returns:
        :obj:`torch.Tensor`: The flattened, 1-D tensor.
    """
    C = tensor.size(1)
    # new axis order
    axis_order = (1, 0) + tuple(range(2, tensor.dim()))
    # Transpose: (N, C, D, H, W) -> (C, N, D, H, W)
    transposed = tensor.permute(axis_order).contiguous()
    # Flatten: (C, N, D, H, W) -> (C, N * D * H * W)
    return transposed.view(C, -1)
