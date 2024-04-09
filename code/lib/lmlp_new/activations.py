from abc import ABC, abstractmethod

import numpy as np
import numpy.typing as nptype


class Activation(ABC):

    @abstractmethod
    def net(self, W: nptype.NDArray, x: nptype.NDArray) -> nptype.NDArray:
        pass
    
    @abstractmethod
    def evaluate(self, net: nptype.NDArray) -> nptype.NDArray:
        pass

    @abstractmethod
    def grad(self, net: nptype.NDArray) -> nptype.NDArray:
        pass

    @abstractmethod
    def grad_net_weights(self, W: nptype.NDArray, x: nptype.NDArray) -> nptype.NDArray:
        pass

    @abstractmethod
    def grad_net_input(self, W: nptype.NDArray, x: nptype.NDArray) -> nptype.NDArray:
        pass


class Sigmoid(Activation):

    def net(self, W: nptype.NDArray, x: nptype.NDArray) -> nptype.NDArray:
        return W @ x

    def evaluate(self, net: nptype.NDArray) -> nptype.NDArray:
        return 1 / (1 + np.exp(-net))
    
    def grad(self, net: nptype.NDArray) -> nptype.NDArray:
        output = self.evaluate(net)

        return output * (1 - output)
    
    def grad_net_weights(self, W: nptype.NDArray, x: nptype.NDArray) -> nptype.NDArray:
        return np.tile(x, (W.shape[0], 1))
    
    def grad_net_input(self, W: nptype.NDArray, x: nptype.NDArray) -> nptype.NDArray:
        return W.T


class Tanh(Activation):

    def net(self, W: nptype.NDArray, x: nptype.NDArray) -> nptype.NDArray:
        return W @ x

    def evaluate(self, net: nptype.NDArray) -> nptype.NDArray:
        return 2 / (1 + np.exp(-2*net)) - 1
    
    def grad(self, net: nptype.NDArray) -> nptype.NDArray:
        output = self.evaluate(net)

        return 1 - output ** 2
    
    def grad_net_weights(self, W: nptype.NDArray, x: nptype.NDArray) -> nptype.NDArray:
        return np.tile(x, (W.shape[0], 1))
    
    def grad_net_input(self, W: nptype.NDArray, x: nptype.NDArray) -> nptype.NDArray:
        return W.T


class ReLU(Activation):

    def net(self, W: nptype.NDArray, x: nptype.NDArray) -> nptype.NDArray:
        return W @ x

    def evaluate(self, net: nptype.NDArray) -> nptype.NDArray:
        return np.clip(net, 0, None)
    
    def grad(self, net: nptype.NDArray) -> nptype.NDArray:
        output = self.evaluate(net)

        return np.where(output >= 0, 1, 0)
    
    def grad_net_weights(self, W: nptype.NDArray, x: nptype.NDArray) -> nptype.NDArray:
        return np.tile(x, (W.shape[0], 1))
    
    def grad_net_input(self, W: nptype.NDArray, x: nptype.NDArray) -> nptype.NDArray:
        return W.T


class Softmax(Activation):

    def net(self, W: nptype.NDArray, x: nptype.NDArray) -> nptype.NDArray:
        return W @ x

    def evaluate(self, net: nptype.NDArray) -> nptype.NDArray:
        c = np.max(net, axis=0)
        exp_net = np.exp(net - c)

        return exp_net / np.sum(exp_net, axis=0)

    def grad(self, net: nptype.NDArray) -> nptype.NDArray:
        output = self.evaluate(net)
        id = np.identity(output.shape[0])

        # element at [i, j, k] -> ith x, jth output, kth sample
        return np.apply_along_axis(lambda col: (id - col).T * col, 0, output)
    
    def grad_net_weights(self, W: nptype.NDArray, x: nptype.NDArray) -> nptype.NDArray:
        return np.tile(x, (W.shape[0], 1))
    
    def grad_net_input(self, W: nptype.NDArray, x: nptype.NDArray) -> nptype.NDArray:
        return W.T


class And(Activation):

    def _sig(self, x: nptype.NDArray) -> nptype.NDArray:
        return 1 / (1 + np.exp(-x))
    
    def _dsig(self, x: nptype.NDArray) -> nptype.NDArray:
        sig = self._sig(x)

        return sig * (1 - sig)

    def _not(self, x: nptype.NDArray) -> nptype.NDArray:
        return 1 - x
    
    def net(self, W: nptype.NDArray, x: nptype.NDArray) -> nptype.NDArray:
        return np.prod(self._not(self._sig(W) * self._not(x)), axis=1)

    def evaluate(self, net: nptype.NDArray) -> nptype.NDArray:
        return net

    def grad(self, net: nptype.NDArray) -> nptype.NDArray:
        return np.ones_like(net)
    
    def grad_net_weights(self, W: nptype.NDArray, x: nptype.NDArray) -> nptype.NDArray:
        scaled = self._not(self._sig(W) * self._not(x))
        net = np.prod(scaled, axis=1)
        net_ext = (np.ones((scaled.shape[1], scaled.shape[0])) * net).T

        grad_scaled = net_ext / scaled #np.where(scaled < 1e-15, 1, scaled)
    
        return -(grad_scaled * self._dsig(W)) * self._not(x)
    
    def grad_net_input(self, W: nptype.NDArray, x: nptype.NDArray) -> nptype.NDArray:
        scaled = self._not(self._sig(W) * self._not(x))
        net = np.prod(scaled, axis=1)
        net_ext = (np.ones((scaled.shape[1], scaled.shape[0])) * net).T

        grad_scaled = net_ext / scaled #np.where(scaled < 1e-15, 1, scaled)
    
        return (grad_scaled * self._sig(W)).T


class Or(Activation):

    _and: And = And()

    def _not(self, x: nptype.NDArray) -> nptype.NDArray:
        return 1 - x

    def net(self, W: nptype.NDArray, x: nptype.NDArray) -> nptype.NDArray:
        return self._and.net(W, self._not(x))

    def evaluate(self, net: nptype.NDArray) -> nptype.NDArray:
        return self._not(self._and.evaluate(net))

    def grad(self, net: nptype.NDArray) -> nptype.NDArray:
        return -self._and.grad(net)
    
    def grad_net_weights(self, W: nptype.NDArray, x: nptype.NDArray) -> nptype.NDArray:
        return self._and.grad_net_weights(W, self._not(x))
    
    def grad_net_input(self, W: nptype.NDArray, x: nptype.NDArray) -> nptype.NDArray:
        return -self._and.grad_net_input(W, self._not(x))


class Normalization(Activation):

    def net(self, W: nptype.NDArray, x: nptype.NDArray) -> nptype.NDArray:
        return W @ x

    def evaluate(self, net: nptype.NDArray) -> nptype.NDArray:
        return net / np.sum(net)

    def grad(self, net: nptype.NDArray) -> nptype.NDArray:
        net_sum = np.sum(net)

        return (np.diag(np.full(len(net), net_sum)) - net) / net_sum ** 2
    
    def grad_net_weights(self, W: nptype.NDArray, x: nptype.NDArray) -> nptype.NDArray:
        return np.tile(x, (W.shape[0], 1))
    
    def grad_net_input(self, W: nptype.NDArray, x: nptype.NDArray) -> nptype.NDArray:
        return W.T
