import torch
import torch.nn.functional as F


def js_div(inputs):
    js_divs = []
    for dist in range(inputs.size(1)):
        js_divs.append(
            F.kl_div(torch.mean(inputs, dim=1, keepdim=True).log(), inputs[:, dist].unsqueeze(0), reduction='sum'))

    return torch.tensor(js_divs).mean()
