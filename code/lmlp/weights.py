from typing import Tuple

from abc import ABC, abstractmethod

import torch
from torch import Tensor

from .activations import And3, Xor3Fixed2, Or3Fixed


####################
#  Initialization  #
####################


class WeightInitializer(ABC):
    """
    An initializer for weight tensors.
    """

    @abstractmethod
    def initialize(self, weight: Tensor) -> None:
        """
        Initializes `weight` in place.
        """

        pass


class NormalInitializer(WeightInitializer):
    
    sigma: float

    def __init__(self, sigma: float = 1.0) -> None:
        super().__init__()

        self.sigma = sigma

    def initialize(self, weight: Tensor) -> None:
        weight.copy_(torch.normal(std=self.sigma, size=weight.shape))


class UniformInitializer(WeightInitializer):

    scale: float

    def __init__(self, scale: float = 0.5) -> None:
        super().__init__()

        self.scale = abs(scale)

    def initialize(self, weight: Tensor) -> None:
        weight.copy_(2 * self.scale * torch.rand_like(weight) - self.scale)


class GlorotNormalInitializer(WeightInitializer):

    def initialize(self, weight: Tensor) -> None:
        initializer = NormalInitializer(torch.sqrt(2 / sum(weight.shape)))
        initializer.initialize(weight)


class GlorotUniformInitializer(WeightInitializer):

    def initialize(self, weight: Tensor) -> None:
        initializer = UniformInitializer(torch.sqrt(6 / sum(weight.shape)))
        initializer.initialize(weight)


class HeNormalInitializer(WeightInitializer):

    def initialize(self, weight: Tensor) -> None:
        initializer = NormalInitializer(torch.sqrt(2 / weight.shape[1]))
        initializer.initialize(weight)


class HeUniformInitializer(WeightInitializer):

    def initialize(self, weight: Tensor) -> None:
        initializer = UniformInitializer(torch.sqrt(6 / weight.shape[1]))
        initializer.initialize(weight)


class ConstantInitializer(WeightInitializer):

    value: float

    def __init__(self, value: float) -> None:
        super().__init__()

        self.value = value

    def initialize(self, weight: Tensor) -> None:
        weight.copy_(self.value * torch.ones_like(weight))


class EyeInitializer(WeightInitializer):

    def initialize(self, weight: Tensor) -> None:
        weight.copy_(torch.eye(*weight.shape))


####################
#  Regularization  #
####################


class WeightRegularizer(ABC):

    @abstractmethod
    def compute_loss(self, weight: Tensor, output: Tensor) -> Tensor:
        pass


class NoRegularizer(WeightRegularizer):

    def compute_loss(self, weight: Tensor, output: Tensor) -> Tensor:
        return torch.tensor(0.0)


class CombinedRegularizer(WeightRegularizer):

    regularizers: Tuple[WeightRegularizer]

    def __init__(self, *regularizers: WeightRegularizer) -> None:
        super().__init__()

        self.regularizers = regularizers

    def compute_loss(self, weight: Tensor, output: Tensor) -> Tensor:
        return sum([r.compute_loss(weight, output) for r in self.regularizers])


class L1Regularizer(WeightRegularizer):

    strength: float

    def __init__(self, strength: float) -> None:
        super().__init__()

        self.strength = strength

    def compute_loss(self, weight: Tensor, output: Tensor) -> Tensor:
        abs_scaled_weight = torch.abs(torch.tanh(weight))

        return self.strength * torch.sum(abs_scaled_weight)


# TODO: also upper bound
class L1AdaptiveRegularizer(WeightRegularizer):

    relative_strength: float
    minimal_strength: float

    def __init__(self, relative_strength: float, minimal_strength: float) -> None:
        super().__init__()

        self.relative_strength = relative_strength
        self.minimal_strength = minimal_strength

    def compute_loss(self, weight: Tensor, output: Tensor) -> Tensor:
        abs_scaled_weight = torch.abs(torch.tanh(weight))
        strength = torch.maximum(self.relative_strength * torch.abs(weight.grad),
                                 self.minimal_strength * torch.ones_like(weight.grad))

        return torch.sum(strength * abs_scaled_weight)


class FL1Regularizer(WeightRegularizer):

    strength: float

    def __init__(self, strength: float) -> None:
        super().__init__()

        self.strength = strength

    def compute_loss(self, weight: Tensor, output: Tensor) -> Tensor:
        return self.strength * torch.sum(torch.abs(weight[1, :]))


class L1NRegularizer(WeightRegularizer):

    strength: float

    def __init__(self, strength: float) -> None:
        super().__init__()

        self.strength = strength

    def compute_loss(self, weight: Tensor, output: Tensor) -> Tensor:
        return self.strength * torch.mean(torch.sum(torch.abs(weight), 1))
    

class L1DynamicRegularizer(WeightRegularizer):

    strength: float

    def __init__(self, strength: float) -> None:
        super().__init__()

        self.strength = strength

    def compute_loss(self, weight: Tensor, output: Tensor) -> Tensor:
        abs_scaled_weight = torch.abs(torch.tanh(weight))
        weight_sum = torch.sum(abs_scaled_weight, 1, True)
        loss = (weight_sum - abs_scaled_weight).detach() * abs_scaled_weight

        return self.strength * torch.sum(loss)


class L1OutRegularizer(WeightRegularizer):

    strength: float

    def __init__(self, strength: float) -> None:
        super().__init__()

        self.strength = strength

    def compute_loss(self, weight: Tensor, output: Tensor) -> Tensor:
        return self.strength * torch.sum(output)


class UncertaintyRegularizer(WeightRegularizer):

    strength: float

    def __init__(self, strength: float) -> None:
        super().__init__()

        self.strength = strength

    def compute_loss(self, weight: Tensor, output: Tensor) -> Tensor:
        # return self.strength * torch.sum(weight ** 2 * (weight - 1) ** 2 * (weight + 1) ** 2)
        
        # abs_weight = torch.abs(weight)

        # # return self.strength * torch.sum(abs_weight + torch.sin(3 * torch.pi * abs_weight) / 10)
        
        # loss_1 = (3 / 2) * abs_weight
        # loss_2 = (1 / 3) * abs_weight ** 3 - (12 / 5) * abs_weight ** 2 + (28 / 15) * abs_weight - (1 / 5)
        # take_loss_2 = abs_weight > 0.5

        # torch.where(take_loss_2, loss_2, loss_1)

        # return self.strength * torch.sum(torch.where(take_loss_2, loss_2, loss_1))
    
        # return self.strength * torch.sum((torch.sin(torch.pi * (weight - 0.5)) + 1) / 2)

        abs_scaled_weight = torch.abs(torch.tanh(weight))

        return self.strength * torch.sum(abs_scaled_weight * (1 - abs_scaled_weight))
    
        return self.strength * torch.sum(abs_scaled_weight * (1.5 - abs_scaled_weight) / 2)

        return self.strength * torch.sum(abs_scaled_weight * (1.8 - abs_scaled_weight) / 3)


class UncertaintyRegularizer2(WeightRegularizer):

    strength: float

    def __init__(self, strength: float) -> None:
        super().__init__()

        self.strength = strength

    def compute_loss(self, weight: Tensor, output: Tensor) -> Tensor:
        abs_scaled_weight = torch.abs(torch.tanh(weight))
    
        return self.strength * torch.sum(abs_scaled_weight * (1.6 - abs_scaled_weight))
    

class UncertaintyRegularizer3(WeightRegularizer):

    strength: float

    def __init__(self, strength: float) -> None:
        super().__init__()

        self.strength = strength

    def compute_loss(self, weight: Tensor, output: Tensor) -> Tensor:
        abs_scaled_weight = torch.abs(torch.tanh(weight))
        loss = abs_scaled_weight * (1.6 - abs_scaled_weight)
        loss = torch.where(abs_scaled_weight > 0.8, 0.64 * torch.ones_like(loss), loss)
    
        return self.strength * torch.sum(loss)


class XorRegularizer(WeightRegularizer):

    _xor = Xor3Fixed2()
    _or = Or3Fixed()

    strength: float

    def __init__(self, strength: float) -> None:
        super().__init__()

        self.strength = strength
        self._or_history = []
        self._xor_history = []

    def compute_loss(self, weight: Tensor, output: Tensor) -> Tensor:
        self._or_history.append(self._or.forward(None, output))
        self._xor_history.append(torch.sum(self._xor.forward(None, output)))

        return - self.strength * self._xor_history[-1]


class OrRegularizer(WeightRegularizer):

    _xor = Xor3Fixed2()
    _or = Or3Fixed()

    strength: float

    def __init__(self, strength: float) -> None:
        super().__init__()

        self.strength = strength
        self._or_history = []
        self._xor_history = []

    def compute_loss(self, weight: Tensor, output: Tensor) -> Tensor:
        self._or_history.append(self._or.forward(None, output))
        self._xor_history.append(torch.sum(self._xor.forward(None, output)))

        return torch.tensor(0) #- self.strength * self._or_history[-1]


class SimilarityRegularizer(WeightRegularizer):

    strength: float
    
    # _and: And3 = And3()
    # _weight: Tensor = torch.tensor([[10, 10]])

    def __init__(self, strength: float) -> None:
        super().__init__()

        self.strength = strength

    def compute_loss(self, weight: Tensor, output: Tensor) -> Tensor:
        if len(output.shape) < 2:
            output = torch.atleast_2d(output).T
        
        detached = output.detach()
        # self._weight = self._weight.to(output)

        loss = torch.sum(torch.tril(output @ detached.T, -1))

        # n = len(output)
        # output_pairs = torch.stack([torch.repeat_interleave(output[1:], n),
        #                             torch.tile(detached, (n - 1,))])
        # previous = torch.tril(torch.ones((n - 1, n), dtype=torch.bool)).flatten()
        # loss = torch.sum(self._and.forward(self._weight, output_pairs[previous]))

        # loss = sum(self._and.forward(self._weight, torch.concatenate([detached[j:j+1], output[i:i+1]]))
        #            for i in range(len(output)) for j in range(i))

        return self.strength * loss


# class SimilarityRegularizer(WeightRegularizer):

#     strength: float

#     def __init__(self, strength: float) -> None:
#         super().__init__()

#         self.strength = strength

#     def compute_loss(self, weight: Tensor, output: Tensor) -> Tensor:
#         detached = output.detach()
#         self._weight = self._weight.to(output)

#         n = len(output)
#         output_pairs = torch.stack([torch.repeat_interleave(output[1:], n),
#                                     torch.tile(detached, (n - 1,))])
#         previous = torch.tril(torch.ones((n - 1, n), dtype=torch.bool)).flatten()
#         loss = torch.sum(self._and.forward(self._weight, output_pairs[previous]))

#         # loss = sum(self._and.forward(self._weight, torch.concatenate([detached[j:j+1], output[i:i+1]]))
#         #            for i in range(len(output)) for j in range(i))

#         return self.strength * loss


####################
#  Postprocessing  #
####################


class WeightPostprocessor(ABC):

    @abstractmethod
    def postprocess(self, weight: Tensor) -> None:
        pass


class NoPostprocessor(WeightPostprocessor):

    def postprocess(self, weight: Tensor) -> None:
        pass


class ClampingPostprocessor(WeightPostprocessor):

    lower_bound: float
    upper_bound: float

    def __init__(self, lower_bound: float, upper_bound: float) -> None:
        super().__init__()

        self.lower_bound = lower_bound
        self.upper_bound = upper_bound

    def postprocess(self, weight: Tensor) -> None:
        weight.copy_(torch.clamp(weight, self.lower_bound, self.upper_bound))


class ZeroingPostprocessor(WeightPostprocessor):

    factor: float
    frequency: float

    _t: int

    def __init__(self, factor: float, frequency: int) -> None:
        super().__init__()

        self.factor = factor
        self.frequency = frequency
        self._t = 0

    def postprocess(self, weight: Tensor) -> None:
        self._t += 1

        if (self._t % self.frequency) == 0:
            abs_scaled_weight = torch.abs(torch.tanh(weight))
            bound = self.factor * torch.max(abs_scaled_weight, dim=1, keepdim=True).values
            below_bound = abs_scaled_weight < bound
            weight[below_bound] = 0
            
            # clear = torch.max(torch.where(below_threshold, torch.rand_like(weight), torch.zeros_like(weight)), 1)

            # row = 0
            # for col, val in zip(clear.indices, clear.values):
            #     if val > 0:
            #         weight[row, col] = 0
                
            #     row += 1


class DampingPostprocessor(WeightPostprocessor):

    max_damping: float
    frequency: float

    _t: int

    def __init__(self, max_damping: float, frequency: int) -> None:
        super().__init__()

        self.max_damping = max_damping
        self.frequency = frequency
        self._t = 0

    def postprocess(self, weight: Tensor) -> None:
        self._t += 1

        if (self._t % self.frequency) == 0:
            abs_scaled_weight = torch.abs(torch.tanh(weight))
            abs_max = torch.max(abs_scaled_weight, dim=1, keepdim=True).values
            damping = 1 + (self.max_damping - 1) * (abs_max - abs_scaled_weight) / abs_max
            weight /= damping
            
            # clear = torch.max(torch.where(below_threshold, torch.rand_like(weight), torch.zeros_like(weight)), 1)

            # row = 0
            # for col, val in zip(clear.indices, clear.values):
            #     if val > 0:
            #         weight[row, col] = 0
                
            #     row += 1
