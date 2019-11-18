import torch


class GradientClippingMixin(object):
    CLIP_FUNC = {
        "norm": torch.nn.utils.clip_grad_norm_,
        "value": torch.nn.utils.clip_grad_value_
    }

    def clip_gradients(self):
        self.CLIP_FUNC[self._gradient_clipping_func](self._model.parameters(), **self._gradient_clipping_params)
