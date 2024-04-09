from __future__ import annotations
from typing import List, Callable
from abc import ABC, abstractmethod

import numpy as np
import numpy.typing as nptype


class Activation(ABC):

    @abstractmethod
    def __call__(self, W: nptype.NDArray, input: nptype.NDArray) -> nptype.NDArray:
        pass

    @abstractmethod
    def dW(self, W: nptype.NDArray, input: nptype.NDArray) -> nptype.NDArray:
        pass

    @abstractmethod
    def dinput(self, W: nptype.NDArray, input: nptype.NDArray) -> nptype.NDArray:
        pass


class And(Activation):

    def _sig(self, x: nptype.NDArray) -> nptype.NDArray:
        return 1 / (1 + np.exp(-x))
    
    def _dsig(self, x: nptype.NDArray) -> nptype.NDArray:
        sig = self._sig(x)
        return sig * (1 - sig)

    def _not(self, x: nptype.NDArray) -> nptype.NDArray:
        return 1 - x

    def __call__(self, W: nptype.NDArray, input: nptype.NDArray) -> nptype.NDArray:
        # net = self._not(W * self._not(input)) + 1e-6

        # return np.prod(net, axis=1)
    
        net = self._not(self._sig(W) * self._not(input))

        return np.prod(net, axis=1)
    
    def dW(self, W: nptype.NDArray, input: nptype.NDArray) -> nptype.NDArray:
        # net = self._not(W * self._not(input)) + 1e-6
        # out = np.prod(net, axis=1)

        # return - (((1 / net) * self._not(input)).T * out).T
    
        net = self._not(self._sig(W) * self._not(input)) + 1e-6
        out = np.prod(net, axis=1)

        return - self._dsig(W) * (((1 / net) * self._not(input)).T * out).T

        # net = self._not(self._sig(W) * self._not(input))
        # dinner = - self._dsig(W) * self._not(input)

        # cols = []
        # for i in range(W.shape[1]):
        #     mask = [True] * i + [False] + [True] * (W.shape[1] - i - 1)
        #     cols.append(np.prod(net, axis=1, where=mask))
        
        # return np.vstack(cols).T * dinner
    
    def dinput(self, W: nptype.NDArray, input: nptype.NDArray) -> nptype.NDArray:
        # net = self._not(W * self._not(input)) + 1e-6
        # out = np.prod(net, axis=1)

        # return (W / net).T * out
        
        net = self._not(self._sig(W) * self._not(input)) + 1e-6
        out = np.prod(net, axis=1)

        return (self._sig(W) / net).T * out
    
        # net = self._not(self._sig(W) * self._not(input))
        # dinner = np.sum(self._sig(W), axis=0) * input

        # cols = []
        # for i in range(W.shape[1]):
        #     mask = [True] * i + [False] + [True] * (W.shape[1] - i - 1)
        #     cols.append(np.prod(net, axis=1, where=mask))
        
        # return np.sum(np.vstack(cols).T, axis=0) * dinner

    # def _sig2(self, x: nptype.NDArray) -> nptype.NDArray:
    #     return 1 / (1 + np.exp(-100*x))
    
    # def _dsig2(self, x: nptype.NDArray) -> nptype.NDArray:
    #     sig = self._sig2(x)
    #     return 100 * sig * (1 - sig)

    # def __call__(self, W: nptype.NDArray, input: nptype.NDArray) -> nptype.NDArray:
    #     net = W @ input
    #     max_net = W @ np.ones(input.shape)

    #     return net / max_net

    # def dW(self, W: nptype.NDArray, input: nptype.NDArray) -> nptype.NDArray:
    #     net = W @ input
    #     max_net = W @ np.ones(input.shape)

    #     return ((np.outer(max_net, input).T - net) / (max_net ** 2)).T

    # def dinput(self, W: nptype.NDArray, input: nptype.NDArray) -> nptype.NDArray:
    #     max_net = W @ np.ones(input.shape)

    #     return W.T / max_net


class Or(Activation):

    _and: And = And()

    def _not(self, x: nptype.NDArray) -> nptype.NDArray:
        return 1 - x

    def __call__(self, W: nptype.NDArray, input: nptype.NDArray) -> nptype.NDArray:
        return self._not(self._and(W, self._not(input)))

    def dW(self, W: nptype.NDArray, input: nptype.NDArray) -> nptype.NDArray:
        return -self._and.dW(W, self._not(input))

    def dinput(self, W: nptype.NDArray, input: nptype.NDArray) -> nptype.NDArray:
        return self._and.dinput(W, self._not(input))


class Xor(Activation):

    _and: And = And()

    def _not(self, x: nptype.NDArray) -> nptype.NDArray:
        return 1 - x

    def __call__(self, W: nptype.NDArray, input: nptype.NDArray) -> nptype.NDArray:
        return self._not(self._and(W, self._not(input)))

    def dW(self, W: nptype.NDArray, input: nptype.NDArray) -> nptype.NDArray:
        return -self._and.dW(W, self._not(input))

    def dinput(self, W: nptype.NDArray, input: nptype.NDArray) -> nptype.NDArray:
        return self._and.dinput(W, self._not(input))


class Softmax(Activation):

    dprocess: Callable[[nptype.NDArray], nptype.NDArray] = lambda x: np.sum(x, axis=-1)

    def __call__(self, W: nptype.NDArray, input: nptype.NDArray) -> nptype.NDArray:
        net = W @ input
        exp_net = np.exp(net)

        return exp_net / np.sum(exp_net)

    def dW(self, W: nptype.NDArray, input: nptype.NDArray) -> nptype.NDArray:
        output = self(W, input)
        dout_dnet = (np.identity(len(output)) - output).T * output

        # jki -> jth net, kth input, ith output
        return np.apply_along_axis(lambda c: np.outer(c, input), 0, dout_dnet)
    
        # dout_dnet_dW = np.apply_along_axis(lambda c: np.outer(c, input), 0, dout_dnet)

        # return self.dprocess(dout_dnet_dW)

    def dinput(self, W: nptype.NDArray, input: nptype.NDArray) -> nptype.NDArray:
        output = self(W, input)
        dout_dnet = (np.identity(len(output)) - output).T * output

        # jki -> jth input, ith output
        return np.apply_along_axis(lambda c: W.T @ c, 0, dout_dnet)

        # dout_dnet_din = np.apply_along_axis(lambda c: W.T * c, 0, dout_dnet)

        # return self.dprocess(dout_dnet_din)
    

class Normalization(Activation):

    dprocess: Callable[[nptype.NDArray], nptype.NDArray] = lambda x: np.sum(x, axis=-1)

    def __call__(self, W: nptype.NDArray, input: nptype.NDArray) -> nptype.NDArray:
        net = W @ input

        return net / np.sum(net)

    def dW(self, W: nptype.NDArray, input: nptype.NDArray) -> nptype.NDArray:
        net = W @ input
        net_sum = np.sum(net)
        dout_dnet = (np.diag(np.full(len(net), net_sum)) - net) / net_sum ** 2

        # jki -> jth net, kth input, ith output
        return np.apply_along_axis(lambda c: np.outer(c, input), 0, dout_dnet)

    def dinput(self, W: nptype.NDArray, input: nptype.NDArray) -> nptype.NDArray:
        net = W @ input
        net_sum = np.sum(net)
        dout_dnet = (np.diag(np.full(len(net), net_sum)) - net) / net_sum ** 2

        # jki -> jth input, ith output
        return np.apply_along_axis(lambda c: W.T @ c, 0, dout_dnet)


class Sigmoid(Activation):

    def __call__(self, W: nptype.NDArray, input: nptype.NDArray) -> nptype.NDArray:
        net = W @ input

        return 1 / (1 + np.exp(-net))

    def dW(self, W: nptype.NDArray, input: nptype.NDArray) -> nptype.NDArray:
        output = self(W, input)
        dout_dnet = output * (1 - output)

        return np.outer(dout_dnet, input)

    def dinput(self, W: nptype.NDArray, input: nptype.NDArray) -> nptype.NDArray:
        output = self(W, input)
        dout_dnet = output * (1 - output)

        return W.T @ dout_dnet


class LMLPLayer:

    def __init__(
        self, output_dim: int, activation: Activation,
        trainable: bool, use_bias: bool = False
    ):
        self.output_dim = output_dim
        self.activation = activation
        self.trainable = trainable
        self.use_bias = use_bias
        
        self.W = None
        self._input_dim = None

        self.gradients_history = []

    @property
    def input_dim(self) -> int | None:
        return self._input_dim

    @input_dim.setter
    def input_dim(self, input_dim: int):
        self._input_dim = input_dim
        self._init_weights()

    def _init_weights(self):
        if self.trainable:
            self.W = np.random.uniform(-0.25, 0.25, (self.output_dim, self._input_dim + self.use_bias))
        else:
            self.W = 10 * np.ones((self.output_dim, self._input_dim + self.use_bias))

    def _add_bias(self, input: nptype.NDArray) -> nptype.NDArray:
        return np.concatenate((input, [1]))

    def evaluate(self, input: nptype.NDArray) -> nptype.NDArray:
        if self.use_bias:
            input = self._add_bias(input)
        
        self.gradients_history.append((self.activation.dW(self.W, input), self.activation.dinput(self.W, input)))

        return self.activation(self.W, input)
    
    def backpropagate(self, alpha: float, dobj_dout: nptype.NDArray) -> nptype.NDArray:
        dout_dW, dout_din = self.gradients_history.pop()

        if self.trainable:
            # sigW = self.activation._sig(self.W)
            # p = np.prod(sigW, axis=0)
            # dsigW = self.activation._dsig(self.W)
            gradW = dout_dW * dobj_dout #- 0.001 * np.sign(alpha) * np.exp(-np.abs(self.W)) * np.sign(self.W) #- 0.01 * np.sign(alpha) * (- np.sign(self.W)) #- 0.1 * np.sign(alpha) * (dsigW / (sigW + 1e-6)) * p
            
            # 0.0001 * np.sign(alpha) * np.exp(-np.abs(self.W)) * self.W / (np.abs(self.W) + 1e-6)

            self.W += alpha * gradW
            # self.W = np.clip(self.W, -10, 10)

        dobj_din = (dout_din.T * dobj_dout).T

        # if isinstance(self.activation, Or):
        #     ind = np.unravel_index(np.argmax(dobj_din), dobj_din.shape)
        #     val = dobj_din[ind]
        #     dobj_din = np.clip(dobj_din, None, 0)
        #     dobj_din[ind] = val

        if self.use_bias:
            dobj_din = dobj_din[:-1]

        return dobj_din
    
    def __str__(self) -> str:
        desc = ["LMLP layer",
                f"input_dim = {self.input_dim}",
                f"output_dim = {self.output_dim}",
                f"activation = {self.activation}",
                f"trainable = {self.trainable}",
                f"use_bias = {self.use_bias}",
                "",
                str(self.W)]
        
        return "\n".join(desc)


class LMLPCompositeLayer:

    def __init__(self, units: List[LMLP], activation: Activation):
        self.units = units
        self.output_dim = len(units)
        self.activation = activation

        self.W = np.identity(self.output_dim)

        self.gradients_history = []

    def _units_output(self, input: List[nptype.NDArray] | nptype.NDArray) -> nptype.NDArray:
        if not isinstance(input, list):
            input = [input for _ in range(len(self.units))]

        return np.concatenate(tuple(unit.evaluate(input[i])
                                    for i, unit in enumerate(self.units)))
        
    def evaluate(self, input: List[nptype.NDArray] | nptype.NDArray) -> nptype.NDArray:
        units_output = self._units_output(input)

        self.gradients_history.append(self.activation.dinput(self.W, units_output))

        return self.activation(self.W, units_output)
    
    def backpropagate(self, alpha: float, dobj_dout: nptype.NDArray):
        dout_din = self.gradients_history.pop()

        if isinstance(self.activation, Softmax) or isinstance(self.activation, Normalization):
            dout_din = self.activation.dprocess(dout_din)
        
        dobj_din = (dout_din.T * dobj_dout).T

        for i, unit in enumerate(self.units):
            unit.backpropagate(alpha, np.array([dobj_din[i]]))

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

    def __init__(self, input_dim: int):
        self.input_dim = input_dim
        self.layers = []

    def add_layer(self, layer: LMLPCompositeLayer | LMLPLayer):
        if isinstance(layer, LMLPCompositeLayer) and len(self.layers) > 0:
            raise ValueError("Composite layer can only be the first layer of an LMLP")
        
        layer.input_dim = (self.layers[-1].output_dim if len(self.layers) > 0
                           else self.input_dim)
        
        self.layers.append(layer)
    
    def evaluate(self, input: List[nptype.NDArray] | nptype.NDArray) -> List[nptype.NDArray] | nptype.NDArray:
        output = input

        for layer in self.layers:
            output = layer.evaluate(output)

        return output
    
    def backpropagate(self, alpha: float, dobj_dout: nptype.NDArray):
        for layer in reversed(self.layers):
            dobj_dout = layer.backpropagate(alpha, dobj_dout)

    def __str__(self) -> str:
        desc = [f"LMLP with {len(self.layers)} layers", ""]

        for i, layer in enumerate(self.layers, 1):
            desc.append(f"Layer {i}")
            desc.append(str(layer))
            desc.append("")

        return "\n".join(desc)
