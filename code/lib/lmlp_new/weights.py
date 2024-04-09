from abc import ABC, abstractmethod

import numpy as np
import numpy.typing as nptype


class WeightsInitializer(ABC):

    @abstractmethod
    def initialize(self, input_dim: int, output_dim: int) -> nptype.NDArray:
        pass


class NormalInitializer(WeightsInitializer):

    sigma: float = None

    def __init__(self, sigma: float = 1.0):
        super().__init__()

        self.sigma = sigma

    def initialize(self, input_dim: int, output_dim: int) -> nptype.NDArray:
        return np.random.normal(scale=self.sigma, size=(output_dim, input_dim))


class UniformInitializer(WeightsInitializer):

    scale: float =None

    def __init__(self, scale: float = 0.5):
        super().__init__()

        self.scale = abs(scale)

    def initialize(self, input_dim: int, output_dim: int) -> nptype.NDArray:
        return np.random.uniform(-self.scale, self.scale, (output_dim, input_dim))


class GlorotNormalInitializer(WeightsInitializer):

    def initialize(self, input_dim: int, output_dim: int) -> nptype.NDArray:
        initializer = NormalInitializer(np.sqrt(2 / (input_dim + output_dim)))

        return initializer.initialize(input_dim, output_dim)


class GlorotUniformInitializer(WeightsInitializer):

    def initialize(self, input_dim: int, output_dim: int) -> nptype.NDArray:
        initializer = UniformInitializer(np.sqrt(6 / (input_dim + output_dim)))

        return initializer.initialize(input_dim, output_dim)


class HeNormalInitializer(WeightsInitializer):

    def initialize(self, input_dim: int, output_dim: int) -> nptype.NDArray:
        initializer = NormalInitializer(np.sqrt(2 / input_dim))

        return initializer.initialize(input_dim, output_dim)


class HeUniformInitializer(WeightsInitializer):

    def initialize(self, input_dim: int, output_dim: int) -> nptype.NDArray:
        initializer = UniformInitializer(np.sqrt(6 / input_dim))

        return initializer.initialize(input_dim, output_dim)


class WeightsRegularizer(ABC):

    @abstractmethod
    def grad(self, W: nptype.NDArray) -> nptype.NDArray:
        pass
    
    @abstractmethod
    def post_grad_update(self, W: nptype.NDArray) -> nptype.NDArray:
        pass


class NoRegularizer(WeightsRegularizer):

    def grad(self, W: nptype.NDArray) -> nptype.NDArray:
        return np.zeros_like(W)
    
    def post_grad_update(self, W: nptype.NDArray) -> nptype.NDArray:
        return W


class L1Regularizer(WeightsRegularizer):

    strength: float = None

    def __init__(self, strength: float) -> None:
        super().__init__()

        self.strength = strength

    def grad(self, W: nptype.NDArray) -> nptype.NDArray:
        return -self.strength * np.sign(W)
    
    def post_grad_update(self, W: nptype.NDArray) -> nptype.NDArray:
        return W


class L2Regularizer(WeightsRegularizer):

    strength: float = None

    def __init__(self, strength: float) -> None:
        super().__init__()

        self.strength = strength

    def grad(self, W: nptype.NDArray) -> nptype.NDArray:
        return self.strength * W
    
    def post_grad_update(self, W: nptype.NDArray) -> nptype.NDArray:
        return W


class WeightDecayRegularizer(WeightsRegularizer):

    decay: float = None

    def __init__(self, decay: float) -> None:
        super().__init__()

        self.decay = decay

    def grad(self, W: nptype.NDArray) -> nptype.NDArray:
        return np.zeros(W.shape)
    
    def post_grad_update(self, W: nptype.NDArray) -> nptype.NDArray:
        return self.decay * W


class CustomRegularizer(WeightsRegularizer):

    # strength: float = None

    # def __init__(self, strength: float) -> None:
    #     super().__init__()

    #     self.strength = strength

    def _sig(self, x):
        return 1 / (1 + np.exp(-x))

    def _dsig(self, x):
        y = self._sig(x)

        return y * (1 - y)

    def grad(self, W: nptype.NDArray) -> nptype.NDArray:
        #return -0.001 * np.exp(-np.abs(W)) * np.sign(W)
    
        # return -0.001 * self._dsig(W)

        last = self._sig(W)[-1] #[:, -1]
        g = -0.001 * self._dsig(W) * last
        g[-1] = 0

        return g
    
    def post_grad_update(self, W: nptype.NDArray) -> nptype.NDArray:
        return W