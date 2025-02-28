from __future__ import annotations
from numpy.typing import NDArray

import torch
from torch import Tensor


class ScaledTanh(torch.nn.Module):

    scale: Tensor
    shift: Tensor

    def __init__(self, low: Tensor | NDArray, high: Tensor | NDArray) -> None:
        super().__init__()
        
        self.scale = (torch.tensor(high) - torch.tensor(low)) / 2
        self.shift = self.scale + torch.tensor(low)

    def forward(self, input: Tensor) -> Tensor:
        return self.shift.to(input) + self.scale.to(input) * torch.tanh(input)


class ScaledSigmoid(torch.nn.Module):

    scale: Tensor
    shift: Tensor

    def __init__(self, low: Tensor | NDArray, high: Tensor | NDArray) -> None:
        super().__init__()
        
        self.scale = torch.tensor(high) - torch.tensor(low)
        self.shift = torch.tensor(low)

    def forward(self, input: Tensor) -> Tensor:
        return self.shift.to(input) + self.scale.to(input) * torch.sigmoid(input)

# class ScaledTanh(torch.nn.Tanh):

#     scale: float

#     def __init__(self, scale: float) -> None:
#         super().__init__()
        
#         self.scale = scale

#     def forward(self, input: Tensor) -> Tensor:
#         return self.scale * super().forward(input)
