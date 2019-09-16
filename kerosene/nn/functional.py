import torch
import torch.nn.functional as F


def js_div(inputs):
    return torch.sum(F.kl_div(torch.mean(inputs, dim=1).log(), inputs, reduction='none'), dim=2).mean()
