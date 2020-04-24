# -*- coding: utf-8 -*-
# Copyright 2019 Kerosene Authors. All Rights Reserved.
#
# Licensed under the MIT License;
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://opensource.org/licenses/MIT
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
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
