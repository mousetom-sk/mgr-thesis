from __future__ import annotations
from typing import List

import numpy as np
import numpy.typing as nptype

from .activations import Activation
from .weights import WeightsInitializer, WeightsRegularizer, NoRegularizer
from .util import add_bias


class LMLPLayer:

    output_dim: int = None
    activation: Activation = None
    use_bias: bool = None
    trainable: bool = None
    
    W: nptype.NDArray = None
    weights_initializer: WeightsInitializer = None
    weights_regularizer: WeightsRegularizer = None

    def __init__(
        self, output_dim: int, activation: Activation,
        weights_initializer: WeightsInitializer,
        weights_regularizer: WeightsRegularizer = NoRegularizer(),
        trainable: bool = True, use_bias: bool = False
    ):
        self.output_dim = output_dim
        self.activation = activation
        self.use_bias = use_bias
        self.trainable = trainable
        self.weights_initializer = weights_initializer
        self.weights_regularizer = weights_regularizer

    def init_weights(self, input_dim: int):
        self.W = self.weights_initializer.initialize(input_dim + self.use_bias, self.output_dim)

    def forward(self, input: nptype.NDArray) -> nptype.NDArray:
        if self.use_bias:
            input = add_bias(input)
        
        net = self.activation.net(self.W, input)

        return self.activation.evaluate(net)
    
    def __str__(self) -> str:
        desc = ["LMLP layer",
                f"output_dim = {self.output_dim}",
                f"activation = {self.activation}",
                f"trainable = {self.trainable}",
                f"use_bias = {self.use_bias}",
                "",
                str(self.W)]
        
        return "\n".join(desc)


class LMLPCompositeLayer:

    def __init__(self, output_dim: int, units: List[LMLP], activation: Activation, weights_initializer: WeightsInitializer):
        self.units = units
        self.output_dim = output_dim
        self.activation = activation
        self.weights_initializer = weights_initializer

    def init_weights(self, input_dim: int):
        self.W = self.weights_initializer.initialize(len(self.units), self.output_dim)

    def forward(self, input: List[nptype.NDArray] | nptype.NDArray) -> nptype.NDArray:
        if not isinstance(input, list):
            input = [input for _ in range(len(self.units))]

        units_intermediate = list(unit.forward(input[i]) for i, unit in enumerate(self.units))
        units_output = []

        for inter in units_intermediate:
            units_output.append(inter[-1])

        units_output = np.hstack(units_output)
        net = self.activation.net(self.W, units_output)
        out = self.activation.evaluate(net)

        # if np.isnan(out).any():
        #     print(units_output)
        #     print("here")

        return units_intermediate + [units_output, out]

    def __str__(self) -> str:
        desc = [f"Composite layer with {len(self.units)} units",
                f"activation = {self.activation}",
                ""]

        for i, unit in enumerate(self.units, 1):
            desc.append(f"Unit {i}")
            desc.append(str(unit))
            desc.append("")

        return "\n".join(desc)


class LMLP:

    input_dim: int = None
    output_dim: int = None
    layers: List[LMLPLayer | LMLPCompositeLayer] = None

    def __init__(self, input_dim: int, layers: List[LMLPLayer | LMLPCompositeLayer]):
        self.input_dim = input_dim
        self.output_dim = input_dim
        self.layers = []

        for layer in layers:
            self._add_layer(layer)

    def _add_layer(self, layer: LMLPLayer | LMLPCompositeLayer):
        if isinstance(layer, LMLPCompositeLayer) and len(self.layers) > 0:
            raise ValueError("Composite layer can only be the first layer of an LMLP")

        layer.init_weights(self.output_dim)

        self.layers.append(layer)
        self.output_dim = layer.output_dim

    def forward(self, input: List[nptype.NDArray] | nptype.NDArray) -> List[nptype.NDArray]:
        intermediate = []
        next = input

        for layer in self.layers:
            if isinstance(layer, LMLPCompositeLayer):
                next = layer.forward(next)
                last = next.pop()
                intermediate.append(next)
                next = last
            else:
                intermediate.append(next)
                next = layer.forward(next)
                
        intermediate.append(next)

        return intermediate

    def predict(self, input: List[nptype.NDArray] | nptype.NDArray) -> nptype.NDArray:
        return self.forward(input)[-1]
    
    def __str__(self) -> str:
        desc = [f"LMLP with {len(self.layers)} layers", ""]

        for i, layer in enumerate(self.layers, 1):
            desc.append(f"Layer {i}")
            desc.append(str(layer))
            desc.append("")

        return "\n".join(desc)
    