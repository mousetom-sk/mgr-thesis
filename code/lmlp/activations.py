from abc import ABC, abstractmethod

import torch
from torch import Tensor


class Activation(ABC):
    
    @abstractmethod
    def forward(self, weight: Tensor, input: Tensor) -> Tensor:
        pass


class And(Activation):
    
    def forward(self, weight: Tensor, input: Tensor) -> Tensor:
        return torch.prod(1 - torch.sigmoid(weight) * (1 - input), 1)


class AndC(Activation):
    
    def forward(self, weight: Tensor, input: Tensor) -> Tensor:
        return torch.prod(1 - weight * (1 - input), 1)


class And1(Activation):
    
    def forward(self, weight: Tensor, input: Tensor) -> Tensor:
        scaled_weight = torch.tanh(weight)
        zeros = torch.zeros_like(scaled_weight)

        return torch.prod(1
                          - torch.maximum(scaled_weight, zeros) * (1 - input)
                          + torch.minimum(scaled_weight, zeros) * input,
                          1)


class And1C(Activation):
    
    def forward(self, weight: Tensor, input: Tensor) -> Tensor:        
        zeros = torch.zeros_like(weight)

        return torch.prod(1
                          - torch.maximum(weight, zeros) * (1 - input)
                          + torch.minimum(weight, zeros) * input,
                          1)


class And2(Activation):

    epsilon: float = 1e-1
    
    def forward(self, weight: Tensor, input: Tensor) -> Tensor:
        scaled_weight = torch.tanh(weight)
        zeros = torch.zeros_like(scaled_weight)

        weighted_input = (1
                          - torch.maximum(scaled_weight, zeros) * (1 - input)
                          + torch.minimum(scaled_weight, zeros) * input)

        min_threshold = torch.min(weighted_input, 1).values + self.epsilon
        min_mask = (weighted_input.T < min_threshold).T
        mins = min_mask * weighted_input

        return torch.mean(mins, 1)


class And3(Activation):

    epsilon: float = 1e-10
    
    def forward(self, weight: Tensor, input: Tensor) -> Tensor:
        scaled_weight = torch.tanh(weight)
        zeros = torch.zeros_like(scaled_weight)

        weighted_input = (1
                          - torch.maximum(scaled_weight, zeros) * (1 - input)
                          + torch.minimum(scaled_weight, zeros) * input)

        product_and = torch.prod(weighted_input, 1)
        vanishing_grad = product_and < self.epsilon

        if torch.any(vanishing_grad):
            min_threshold = torch.min(weighted_input, 1).values + self.epsilon
            min_mask = (weighted_input.T < min_threshold).T
            godel_and = torch.mean(min_mask * weighted_input, 1)

            return torch.where(vanishing_grad, godel_and, product_and)
        
        return product_and


class Or(Activation):

    _and: And = And()

    def forward(self, weight: Tensor, input: Tensor) -> Tensor:
        return 1 - self._and.forward(weight, 1 - input)


class Or1(Activation):

    _and: And1 = And1()

    def forward(self, weight: Tensor, input: Tensor) -> Tensor:
        return 1 - self._and.forward(weight, 1 - input)


class Or1C(Activation):

    _and: And1C = And1C()

    def forward(self, weight: Tensor, input: Tensor) -> Tensor:
        return 1 - self._and.forward(weight, 1 - input)


class Or2(Activation):

    _and: And2 = And2()

    def forward(self, weight: Tensor, input: Tensor) -> Tensor:
        return 1 - self._and.forward(weight, 1 - input)


class Or3(Activation):

    _and: And3 = And3()

    def forward(self, weight: Tensor, input: Tensor) -> Tensor:
        return 1 - self._and.forward(weight, 1 - input)
    

class Xor3(Activation):

    _and: And3 = And3()
    _or: Or3 = Or3()
    _weight: Tensor = torch.tensor([[10, 10]])

    def forward(self, weight: Tensor, input: Tensor) -> Tensor:
        n = len(input)
        detached = input.detach()
        in_pairs = torch.stack([torch.repeat_interleave(input[1:], n),
                                torch.tile(detached, (n - 1,))])
        previous = torch.tril(torch.ones((n - 1, n), dtype=torch.bool)).flatten()
        nands = 1 - self._and.forward(self._weight.to(in_pairs), in_pairs[previous])

        or_out = self._or.forward(weight, input)

        # xor_in = torch.cat([torch.tile(nands, (1, len(or_out))).T, torch.atleast_2d(or_out)])
        # xor_weight = 10 * torch.ones((1, xor_in.shape[0])).to(xor_in)

        xor_in = torch.cat([nands, or_out])
        xor_weight = 10 * torch.ones((1, xor_in.shape[0])).to(xor_in)

        return self._and.forward(xor_weight, xor_in)


class Xor3Fixed(Activation):

    epsilon: float = 1e-10

    def forward(self, weight: Tensor, input: Tensor) -> Tensor:
        clauses = torch.tile((1 - input), (len(input), 1))
        clauses[range(len(input)), range(len(input))] = input

        clauses_and = torch.prod(clauses, 1)
        vanishing_grad = clauses_and < self.epsilon

        if torch.any(vanishing_grad):
            min_threshold = torch.min(clauses, 1).values + self.epsilon
            min_mask = (clauses.T < min_threshold).T
            godel_and = torch.mean(min_mask * clauses, 1)

            clauses_and = torch.where(vanishing_grad, godel_and, clauses_and)

        negated_or = torch.prod(1 - clauses_and, 0, keepdim=True)

        if negated_or < self.epsilon:
            min_threshold = torch.min(clauses_and) + self.epsilon
            min_mask = clauses_and < min_threshold
            
            negated_or = torch.mean(min_mask * clauses_and)

        return 1 - negated_or


class Normalization(Activation):

    def forward(self, weight: Tensor, input: Tensor) -> Tensor:
        net = weight @ input
        
        return torch.nan_to_num(net / net.sum())


class ScaledSoftmax(Activation):

    scale: float

    def __init__(self, scale: float):
        super().__init__()

        self.scale = scale

    def forward(self, weight: Tensor, input: Tensor) -> Tensor:
        net = self.scale * weight @ input
        
        return torch.softmax(net, -1)
