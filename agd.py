"""
Automatic Gradient Descent was introduced in this paper - https://arxiv.org/pdf/2304.05187.pdf.
I followed the implementation style of PyTorch. You can find the original implementation by the creators
of the paper here - https://github.com/jxbz/agd.
"""
import math
from typing import Iterable, Callable

import torch
from torch.nn.init import orthogonal_
from torch.optim import Optimizer


class AGD(Optimizer):

    def __init__(self, params: Iterable[torch.Tensor], gain=1.0):
        self.gain = gain
        # The model will have only one optional hyperparameter, which is an acceleration
        # parameter. It increases the update size of Automatic Gradient Descent.
        defaults = dict(gain=gain)
        super().__init__(params, defaults=defaults)
        # Since the Optimizer.__init__() method by default adds the weights `params` 
        # to an attribute `param_groups`, we take the 1st (and only) group and its
        # parameters.
        self.params = self.param_groups[0]["params"]

    @torch.no_grad()
    def step(self):
        # If we are on step 1. Else, we won't initialize the weights.
        if self.state["initialized"]:
            self._init_weights()
            # This parameter indicatest that we've initialized the weights.
            self.state["initialized"] = True 

        loss = agd(
            params=self.params, 
            get_largest_singular_value=self._scale,
            gain=self.gain
        )

        return loss

    @torch.no_grad()
    def _init_weights(self):
        # Looping through each weight matrix and initializing it to be uniform
        # semi-orthogonal and re-scaled by a factor of sqrt(matrix_rows/matrix_cols).
        # We orthogonalize the weight matrices for stability. When the gradient
        # is backpropagated, if we use an orthogonal matrix, the gradient preserves
        # its magnitude. 
        for p in self.params:
            # AGD doesn't support biases yet.
            if p.dim() == 1:
                raise Exception("AGD doesn't support biases.")
            
            self._orthogonalize(p, dim=p.dim())
            self._scale(p)

    @torch.no_grad()
    def _orthogonalize(self, weights: torch.Tensor, dim: int):
        # Orthogonalize a 2D weights matrix.
        if dim == 2:
            orthogonal_(weights)
        # Orthogonalize a 4D weights matrix. Here we only orthogonalize each 2D
        # tensor of the last 2 dimensions (those might be used for a Conv2d layer).
        if dim == 4:
            for x in range(weights.shape[2]):
                for y in range(weights.shape[3]):
                    orthogonal_(weights[:, :, x, y])

    @torch.no_grad()
    def _scale(self, weights: torch.Tensor):
        # Here we are calculating the approximation of the largest singular value
        # of `weights`. This is mentioned in Persicription 1 of the paper. It's mainly
        # used because it simplifies the derivation of AGD.
        singular_values_approx = math.sqrt(weights.shape[0] / weights.shape[1])
        
        # When `weights` is a weights matrix of a Con2d.
        if weights.dim() == 4:
            # This division here is done because the singular values are affected
            # by the other two dimensions, when `weights` is 4D.
            singular_values_approx /= math.sqrt(weights.shape[2] * weights.shape[3])

        return singular_values_approx


def agd(params: Iterable[torch.Tensor], get_largest_singular_value: Callable, gain=1.0):
    grad_summary = 0
    num_layers = len(list(params))

    # Getting the gradient summary
    for p in params:
        # Here, the gradient is calculated across the first two dimensions.
        # If we use `norm` with or without `dim=(0, 1)` for a 2D weights matrix `p`
        # there is no difference. It only makes a difference for 4D weights tensors.
        # In that case, we calculate the Euclidean norm accross the 1st and 2nd
        # dimensions (`dim=(0, 1)`) which generates a 2D tensor. The values of 
        # this tensor is then summed (`sum()`). This way we acknowledge that 
        # the dimensionality of a 4D tensor affects the weights differently.
        grad_summary += p.grad.norm(dim=(0, 1)).sum() * get_largest_singular_value(p)

    # Normalize by the gradient using the number of layers.
    grad_summary /= num_layers
    learning_rate = math.log((1 + math.sqrt(1 + 4 * grad_summary)) / 2)

    for p in params:
        # Updating each weights matrix. In comparison to the loop above, we set
        # the `keepdim` parameter to `True`. This means that when we calculate the
        # norm for a 2D/4D `p`, the dimensionality will remain. I.e. if `p.shape` is
        # (3, 2, 4, 2), after `p.grad.norm(dim=(0, 1))` `p.shape` will be (4, 2). But if
        # we set `keepdim=True`, `p.shape` will be (1, 1, 4, 2). This is done so that
        # we can divide the gradient by its norm.
        p -= (gain * learning_rate / num_layers) * (p.grad / p.grad.norm(dim=(0, 1), keepdim=True)) * get_largest_singular_value(p)
