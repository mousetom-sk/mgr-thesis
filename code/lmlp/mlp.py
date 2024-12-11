from __future__ import annotations
from typing import Union, List

from abc import ABC, abstractmethod
from functools import reduce

import torch
from torch import Tensor
from torch.nn import Module, ModuleList, Parameter

from .activations import Activation
from .weights import WeightInitializer, WeightRegularizer, NoRegularizer, WeightPostprocessor, NoPostprocessor


Tensors = Union[Tensor, List["Tensors"]]


class LMLPModule(ABC, Module):

    @abstractmethod
    def reset_parameters(self, recurse: bool = True) -> None:
        pass

    @abstractmethod
    def compute_regularization_loss(self, recurse: bool = True) -> Tensor:
        pass

    @abstractmethod
    def postprocess_parameters(self, recurse: bool = True) -> None:
        pass


class LMLPLayer(LMLPModule):

    __constants__ = ["out_features", "activation", "weight_initializer",
                     "weight_regularizer", "weight_postprocessor", "weight"]

    out_features: int
    activation: Activation
    weight_initializer: WeightInitializer
    weight_regularizer: WeightRegularizer
    weight_postprocessor: WeightPostprocessor
    weight: Tensor

    _last_output: Tensor

    def __init__(
        self, in_features: int, out_features: int, activation: Activation,
        weight_initializer: WeightInitializer,
        weight_regularizer: WeightRegularizer = NoRegularizer(),
        weight_postprocessor: WeightPostprocessor = NoPostprocessor(),
        trainable: bool = True
    ):
        super().__init__()

        self.out_features = out_features
        self.activation = activation
        self.weight_initializer = weight_initializer
        self.weight_regularizer = weight_regularizer
        self.weight_postprocessor = weight_postprocessor
        self.weight = Parameter(torch.empty((out_features, in_features)),
                                requires_grad=trainable)
        
        self.reset_parameters()

    def reset_parameters(self, recurse: bool = True) -> None:
        with torch.no_grad():
            self.weight_initializer.initialize(self.weight)
    
    def compute_regularization_loss(self, recurse: bool = True) -> Tensor:
        return self.weight_regularizer.compute_loss(self.weight, self._last_output)
    
    def postprocess_parameters(self, recurse: bool = True) -> None:
        with torch.no_grad():
            self.weight_postprocessor.postprocess(self.weight)

    def forward(self, input: Tensor) -> Tensor:
        self._last_output = self.activation.forward(self.weight, input)

        return self._last_output
    
    def __str__(self) -> str:
        return "\n".join(["LMLPLayer",
                          f"out_features: {self.out_features}",
                          f"activation: {self.activation}",
                          f"weight_initializer: {self.weight_initializer}",
                          f"weight: {self.weight}"])


class LMLPParallel(LMLPModule):

    __constants__ = ["out_features", "activation", "weight_initializer",
                     "weight_regularizer", "weight_postprocessor", "weight",
                     "lmlp_modules"]

    out_features: int
    activation: Activation
    weight_initializer: WeightInitializer
    weight_regularizer: WeightRegularizer
    weight_postprocessor: WeightPostprocessor
    weight: Tensor
    lmlp_modules: ModuleList

    _last_output: Tensor

    def __init__(
        self, out_features: int, activation: Activation,
        weight_initializer: WeightInitializer,
        weight_regularizer: WeightRegularizer = NoRegularizer(),
        weight_postprocessor: WeightPostprocessor = NoPostprocessor(),
        trainable: bool = True,
        *lmlp_modules: LMLPSequential | LMLPParallel
    ):
        super().__init__()

        in_features = sum(m.out_features for m in lmlp_modules)

        self.out_features = out_features
        self.activation = activation
        self.weight_initializer = weight_initializer
        self.weight_regularizer = weight_regularizer
        self.weight_postprocessor = weight_postprocessor
        self.weight = Parameter(torch.empty((out_features, in_features)),
                                requires_grad=trainable)
        self.lmlp_modules = ModuleList(lmlp_modules)
        
        self.reset_parameters(False)

    def reset_parameters(self, recurse: bool = True) -> None:
        with torch.no_grad():
            self.weight_initializer.initialize(self.weight)

        if recurse:
            for m in self.lmlp_modules:
                m.reset_parameters()

    def compute_regularization_loss(self, recurse: bool = True) -> Tensor:
        loss = self.weight_regularizer.compute_loss(self.weight, self._last_output)
        
        if recurse:
            loss += sum(m.compute_regularization_loss().to(loss) for m in self.lmlp_modules)

        return loss
    
    def postprocess_parameters(self, recurse: bool = True) -> None:
        with torch.no_grad():
            self.weight_postprocessor.postprocess(self.weight)
        
        if recurse:
            for m in self.lmlp_modules:
                m.postprocess_parameters()

    def forward(self, input: Tensors) -> Tensor:
        if not isinstance(input, list):
            input = [input for _ in range(len(self.lmlp_modules))]

        modules_out = torch.concatenate([m.forward(i)
                                         for i, m in zip(input, self.lmlp_modules)])
        self._last_output = self.activation.forward(self.weight, modules_out)

        return self._last_output

    def __str__(self) -> str:
        desc = [f"LMLPParallel with {len(self.lmlp_modules)} module(s)",
                f"out_features: {self.out_features}",
                f"activation: {self.activation}",
                f"weight_initializer: {self.weight_initializer}",
                f"weight: {self.weight}",
                ""]

        for i, m in enumerate(self.lmlp_modules, 1):
            desc.append(f"Module {i}")
            desc.append(str(m))
            desc.append("")

        return "\n".join(desc)


class LMLPSequential(LMLPModule):

    __constants__ = ["out_features", "lmlp_modules"]

    out_features: int
    lmlp_modules: ModuleList

    def __init__(self, *lmlp_modules: LMLPLayer | LMLPParallel):
        super().__init__()

        self.out_features = lmlp_modules[-1].out_features
        self.lmlp_modules = ModuleList(lmlp_modules)

    def reset_parameters(self, recurse: bool = True) -> None:
        if recurse:
            for m in self.lmlp_modules:
                m.reset_parameters()

    def compute_regularization_loss(self, recurse: bool = True) -> Tensor:
        loss = torch.tensor(0.0)
        
        if recurse:
            loss = sum(m.compute_regularization_loss() for m in self.lmlp_modules)

        return loss
    
    def postprocess_parameters(self, recurse: bool = True) -> None:
       if recurse:
            for m in self.lmlp_modules:
                m.postprocess_parameters()
    
    def forward(self, input: Tensors) -> Tensor:
        return reduce(lambda i, m: m.forward(i), self.lmlp_modules, input)

    def __str__(self) -> str:
        desc = [f"LMLPSequential with {len(self.lmlp_modules)} module(s)",
                f"out_features: {self.out_features}",
                ""]

        for i, m in enumerate(self.lmlp_modules, 1):
            desc.append(f"Module {i}")
            desc.append(str(m))
            desc.append("")

        return "\n".join(desc)
