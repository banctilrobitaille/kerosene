import torch


class GradientClippingMixin(object):

    def __init__(self):
        self._clipping_func = {
            "norm": torch.nn.utils.clip_grad_norm_,
            "value": torch.nn.utils.clip_grad_value_
        }

    def clip_gradients(self, parameters, clipping_func, params):
        if self._clipping_func.get(clipping_func) is not None:
            self._clipping_func[clipping_func](parameters, **params)
