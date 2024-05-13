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
        return np.apply_along_axis(lambda col: np.tile(col, (W.shape[0], 1)), 0, x)
    
    def grad_net_input(self, W: nptype.NDArray, x: nptype.NDArray) -> nptype.NDArray:
        return np.tile(W.T, (1, 1, x.shape[1]))


class Tanh(Activation):

    def net(self, W: nptype.NDArray, x: nptype.NDArray) -> nptype.NDArray:
        return W @ x

    def evaluate(self, net: nptype.NDArray) -> nptype.NDArray:
        return 2 / (1 + np.exp(-2*net)) - 1
    
    def grad(self, net: nptype.NDArray) -> nptype.NDArray:
        output = self.evaluate(net)

        return 1 - output ** 2
    
    def grad_net_weights(self, W: nptype.NDArray, x: nptype.NDArray) -> nptype.NDArray:
        return np.apply_along_axis(lambda col: np.tile(col, (W.shape[0], 1)), 0, x)
    
    def grad_net_input(self, W: nptype.NDArray, x: nptype.NDArray) -> nptype.NDArray:
        return np.tile(W.T, (1, 1, x.shape[1]))


class ReLU(Activation):

    def net(self, W: nptype.NDArray, x: nptype.NDArray) -> nptype.NDArray:
        return W @ x

    def evaluate(self, net: nptype.NDArray) -> nptype.NDArray:
        return np.clip(net, 0, None)
    
    def grad(self, net: nptype.NDArray) -> nptype.NDArray:
        output = self.evaluate(net)

        return np.where(output >= 0, 1, 0)
    
    def grad_net_weights(self, W: nptype.NDArray, x: nptype.NDArray) -> nptype.NDArray:
        return np.apply_along_axis(lambda col: np.tile(col, (W.shape[0], 1)), 0, x)
    
    def grad_net_input(self, W: nptype.NDArray, x: nptype.NDArray) -> nptype.NDArray:
        return np.tile(W.T, (1, 1, x.shape[1]))


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

        # element at (i, j, k) -> i-th x, j-th output, k-th sample
        return np.apply_along_axis(lambda col: (id - col).T * col, 0, output)
    
    def grad_net_weights(self, W: nptype.NDArray, x: nptype.NDArray) -> nptype.NDArray:
        return np.apply_along_axis(lambda col: np.tile(col, (W.shape[0], 1)), 0, x)
    
    def grad_net_input(self, W: nptype.NDArray, x: nptype.NDArray) -> nptype.NDArray:
        return np.tile(W.T, (1, 1, x.shape[1]))


class And(Activation):

    def _sig(self, x: nptype.NDArray) -> nptype.NDArray:
        return 1 / (1 + np.exp(-x))
    
    def _dsig(self, x: nptype.NDArray) -> nptype.NDArray:
        sig = self._sig(x)

        return sig * (1 - sig)

    def _not(self, x: nptype.NDArray) -> nptype.NDArray:
        return 1 - x
    
    def _scaled(self, W: nptype.NDArray, x: nptype.NDArray) -> nptype.NDArray:
        sigW = self._sig(W)
        
        return self._not(np.apply_along_axis(lambda col: sigW * col, 0, self._not(x)))
    
    def net(self, W: nptype.NDArray, x: nptype.NDArray) -> nptype.NDArray:
        return np.prod(self._scaled(W, x), axis=1)

    def evaluate(self, net: nptype.NDArray) -> nptype.NDArray:
        return net

    def grad(self, net: nptype.NDArray) -> nptype.NDArray:
        return np.ones_like(net)
    
    def grad_net_weights(self, W: nptype.NDArray, x: nptype.NDArray) -> nptype.NDArray:
        scaled = self._scaled(W, x)
        net = np.prod(scaled, axis=1)
        net_ext = np.apply_along_axis(lambda col: np.tile(col, (x.shape[0], 1)).T, 0, net)

        grad_scaled = net_ext / scaled #np.where(scaled < 1e-15, 1, scaled)
        grad_ = -(grad_scaled * self._dsig(W))
        
        return self._not(np.apply_along_axis(lambda col: grad_ * col, 0, self._not(x)))
    
    def grad_net_input(self, W: nptype.NDArray, x: nptype.NDArray) -> nptype.NDArray:
        scaled = self._scaled(W, x)
        net = np.prod(scaled, axis=1)
        net_ext = np.apply_along_axis(lambda col: np.tile(col, (x.shape[0], 1)).T, 0, net)

        grad_scaled = net_ext / scaled #np.where(scaled < 1e-15, 1, scaled)
    
        return np.swapaxes(grad_scaled * self._sig(W), 0, 1)


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
        return net / np.clip(np.sum(net, axis=0), 1e-10, None)

    def grad(self, net: nptype.NDArray) -> nptype.NDArray:
        id = np.identity(net.shape[0])

        # element at (i, j, k) -> i-th x, j-th output, k-th sample
        return np.apply_along_axis(lambda col: (max(np.sum(col), 1e-10) * id - col) / (max(np.sum(col), 1e-10) ** 2), 0, net)
    
    def grad_net_weights(self, W: nptype.NDArray, x: nptype.NDArray) -> nptype.NDArray:
        return np.apply_along_axis(lambda col: np.tile(col, (W.shape[0], 1)), 0, x)
    
    def grad_net_input(self, W: nptype.NDArray, x: nptype.NDArray) -> nptype.NDArray:
        return np.tile(W.T, (1, 1, x.shape[1]))
