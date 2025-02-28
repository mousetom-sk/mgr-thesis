import torch
from torch import Tensor
from torch.nn import Module


class ScaledSoftmax(Module):

    scale: float

    def __init__(self, scale: float):
        super().__init__()

        self.scale = scale

    def forward(self, input: Tensor) -> Tensor:
        net = self.scale * input
        
        return torch.softmax(net, -1)
