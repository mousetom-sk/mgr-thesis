from __future__ import annotations
from typing import Dict, List
from abc import ABC, abstractmethod

from functools import reduce

import numpy as np
import numpy.typing as nptype
import numpy._typing as _nptype

from .mlp import LMLP, LMLPLayer, LMLPCompositeLayer
from .util import add_bias, expand


class LMLPOptimizer(ABC):

    lmlp: LMLP = None
    inner_optimizers: Dict[LMLP, LMLPOptimizer] = None

    def prepare(self, lmlp: LMLP):
        self.lmlp = lmlp
        self.inner_optimizers = dict()

        for layer in lmlp.layers:
            if isinstance(layer, LMLPCompositeLayer):
                for unit in layer.units:
                    self.inner_optimizers[unit] = self.copy()
                    self.inner_optimizers[unit].prepare(unit)

    @abstractmethod
    def optimize(
        self, grad_obj_out: nptype.NDArray, intermediate: List[nptype.NDArray]
    ):
        pass
    
    def copy(self) -> LMLPOptimizer:
        return self.__class__()


class BackpropagationOptimizer(LMLPOptimizer, ABC):

    def optimize(
        self, grad_obj_out: nptype.NDArray, intermediate: List[nptype.NDArray]
    ):
        backprop_step = lambda grad, layer_input: self._update_layer(*layer_input, grad)

        reduce(backprop_step, reversed(list(zip(self.lmlp.layers, intermediate))), grad_obj_out)

    def _update_layer(
        self, layer: LMLPLayer | LMLPCompositeLayer, input: nptype.NDArray, grad_obj_out: nptype.NDArray
    ) -> nptype.NDArray:
        if isinstance(layer, LMLPCompositeLayer):
            units_interm = input[:-1]
            input = input[-1]
        elif layer.use_bias:
            input = add_bias(input)
        
        net = layer.activation.net(layer.W, input)
        grad_out_net = layer.activation.grad(net)
        
        if len(grad_out_net.shape) == 2:
            grad_obj_out_ext = np.apply_along_axis(
                lambda col: np.tile(col, (len(col), 1)), 0, grad_obj_out
            )

            # assumes that the overall objective is an affine transformation of objectives over individual outputs
            grad_obj_net = np.sum(grad_obj_out_ext * grad_out_net, axis=1)
        else:
            grad_obj_net = grad_obj_out * grad_out_net
        
        mask = np.where(np.abs(grad_obj_out) > 0, 1, 0)
        mask = np.tile(mask, (len(input), 1))
        grad_obj_in = (layer.activation.grad_net_input(layer.W, input) * mask) @ grad_obj_net            
        # grad_obj_in = layer.activation.grad_net_input(layer.W, input) @ grad_obj_net

        if isinstance(layer, LMLPCompositeLayer):
            for i, unit in enumerate(layer.units):
                self.inner_optimizers[unit].optimize(np.array([grad_obj_in[i]]), units_interm[i])
        elif layer.trainable:
            grad_obj_W = expand(grad_obj_net) * layer.activation.grad_net_weights(layer.W, input)
            grad_obj_W += layer.weights_regularizer.grad(layer.W)

            layer.W = self._update_weights(layer.W, grad_obj_W)
            layer.W = layer.weights_regularizer.post_grad_update(layer.W)

            if layer.use_bias:
                return grad_obj_in[:-1]
        
        return grad_obj_in

    @abstractmethod
    def _update_weights(
        self, W: nptype.NDArray, grad_obj_W: nptype.NDArray
    ) -> nptype.NDArray:
        pass


class GradientDescent(BackpropagationOptimizer):

    alpha: float = None

    def __init__(self, alpha: float):
        super().__init__()

        self.alpha = alpha
    
    def _update_weights(
        self, W: nptype.NDArray, grad_obj_W: nptype.NDArray
    ) -> nptype.NDArray:
        return W + self.alpha * grad_obj_W
    
    def copy(self) -> GradientDescent:
       return self.__class__(self.alpha)


class Adam(BackpropagationOptimizer):

    alpha: float = None
    beta1: float = None
    beta2: float = None
    epsilon: float = None

    _layer_idx: int = None
    _m: List[nptype.NDArray] = None
    _v: List[nptype.NDArray] = None
    _t: int = None

    def __init__(
        self, alpha: float, beta1: float = 0.9, beta2: float = 0.999,
        epsilon: float = 1e-8
    ):
        super().__init__()

        self.alpha = alpha
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon

    def prepare(self, lmlp: LMLP):
        super().prepare(lmlp)

        self._layer_idx = 0
        self._m = []
        self._v = []
        self._t = 0

        for layer in reversed(lmlp.layers):
            self._add_layer(layer.W.shape)

    def _add_layer(self, shape: _nptype._ShapeLike):
        self._m.append(np.zeros(shape))
        self._v.append(np.zeros(shape))

    def optimize(
        self, grad_obj_out: nptype.NDArray, intermediate: List[nptype.NDArray]
    ):
        self._layer_idx = -1
        self._t += 1

        super().optimize(grad_obj_out, intermediate)
    
    def _update_layer(
        self, layer: LMLPLayer | LMLPCompositeLayer, input: nptype.NDArray, grad_obj_out: nptype.NDArray
    ) -> nptype.NDArray:
        self._layer_idx += 1

        return super()._update_layer(layer, input, grad_obj_out)
    
    def _update_weights(
        self, W: nptype.NDArray, grad_obj_W: nptype.NDArray
    ) -> nptype.NDArray:
        self._m[self._layer_idx] *= self.beta1
        self._m[self._layer_idx] += (1 - self.beta1) * grad_obj_W
        self._v[self._layer_idx] *= self.beta2
        self._v[self._layer_idx] += (1 - self.beta2) * (grad_obj_W ** 2)

        m_hat = self._m[self._layer_idx] / (1 - self.beta1 ** self._t)
        v_hat = self._v[self._layer_idx] / (1 - self.beta2 ** self._t)

        return W + self.alpha * m_hat / (np.sqrt(v_hat) + self.epsilon)

    def copy(self) -> Adam:
       return self.__class__(self.alpha, self.beta1, self.beta2, self.epsilon)
