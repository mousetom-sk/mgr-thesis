# Neural Networks (2-AIN-132/15), FMFI UK BA
# (c) Tomas Kuzma, Juraj Holas, Peter Gergel, Endre Hamerlik, Stefan Pócoš, Iveta Bečková 2017-2024


import torch
from torch import Tensor


def add_bias(input: Tensor) -> Tensor:
    """
    Add bias term to vector, or to every (column) vector in a matrix.
    """

    if input.ndim == 1:
        return torch.concatenate((input, [1]))
    
    pad = torch.ones((1, input.shape[1]))
    
    return torch.concatenate((input, pad), 0)

def expand(input: Tensor) -> Tensor:
    return torch.atleast_2d(input.T).T
    
def squeeze(input: Tensor) -> Tensor:
    if input.shape[-1] == 1:
        return torch.squeeze(input, -1)
    
    return input
