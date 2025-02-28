from abc import ABC, abstractmethod

import torch
from torch import Tensor


####################
#  Initialization  #
####################


class WeightInitializer(ABC):
    """
    Initializes weights.
    """

    @abstractmethod
    def initialize(self, weight: Tensor) -> None:
        """
        Initializes `weight` in place.
        """

        pass


class UniformInitializer(WeightInitializer):

    scale: float

    def __init__(self, scale: float) -> None:
        super().__init__()

        self.scale = abs(scale)

    def initialize(self, weight: Tensor) -> None:
        weight.copy_(2 * self.scale * torch.rand_like(weight) - self.scale)


class ConstantInitializer(WeightInitializer):

    value: float

    def __init__(self, value: float) -> None:
        super().__init__()

        self.value = value

    def initialize(self, weight: Tensor) -> None:
        weight.copy_(self.value * torch.ones_like(weight))


# TODO: desired weights initializer (?)


####################
#  Regularization  #
####################


class WeightRegularizer(ABC):
    """
    Computes a regularization loss for weights.
    """

    @abstractmethod
    def compute_loss(self, weight: Tensor, output: Tensor) -> Tensor:
        pass


class NoRegularizer(WeightRegularizer):

    def compute_loss(self, weight: Tensor, output: Tensor) -> Tensor:
        return torch.tensor(0.0)


class L1Regularizer(WeightRegularizer):

    strength: float

    def __init__(self, strength: float) -> None:
        super().__init__()

        self.strength = strength

    def compute_loss(self, weight: Tensor, output: Tensor) -> Tensor:
        abs_scaled_weight = torch.abs(torch.tanh(weight))

        return self.strength * torch.sum(abs_scaled_weight)


class UncertaintyRegularizer(WeightRegularizer):

    strength: float
    peak: float
    constant_after_peak: bool

    def __init__(self, strength: float, peak: float, constant_after_peak: bool) -> None:
        super().__init__()

        self.strength = strength
        self.peak = peak
        self.constant_after_peak = constant_after_peak

    def compute_loss(self, weight: Tensor, output: Tensor) -> Tensor:
        abs_scaled_weight = torch.abs(torch.tanh(weight))
        loss = abs_scaled_weight * (2 * self.peak - abs_scaled_weight)

        if self.constant_after_peak:
            peak_values = (self.peak ** 2) * torch.ones_like(loss)
            loss = torch.where(abs_scaled_weight > self.peak, peak_values, loss)

        return self.strength * torch.sum(loss)


# TODO: when regularizing output calculate batch mean


####################
#  Postprocessing  #
####################


class WeightPostprocessor(ABC):
    """
    Postprocesses weights after optimization updates.
    """

    @abstractmethod
    def postprocess(self, weight: Tensor) -> None:
        pass


class NoPostprocessor(WeightPostprocessor):

    def postprocess(self, weight: Tensor) -> None:
        return


class ClampingPostprocessor(WeightPostprocessor):

    lower_bound: float
    upper_bound: float

    def __init__(self, lower_bound: float, upper_bound: float) -> None:
        super().__init__()

        self.lower_bound = lower_bound
        self.upper_bound = upper_bound

    def postprocess(self, weight: Tensor) -> None:
        weight.copy_(torch.clamp(weight, self.lower_bound, self.upper_bound))
