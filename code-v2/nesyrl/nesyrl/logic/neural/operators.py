from typing import Iterable

import torch
from torch import Tensor

from nesyrl.logic.neural.base import NLModuleParametrized
from nesyrl.logic.propositional import Structure


class NLAndProd(NLModuleParametrized):
    
    def forward(self, structures: Iterable[Structure]) -> Tensor:
        input = self._collect_inputs(structures)
        
        scaled_weight = torch.sigmoid(self.weight)
        weighted_input = 1 - (1 - input) * scaled_weight
        self.last_output = torch.prod(weighted_input, -1, keepdim=True)

        return self.last_output


class NLAndProdClamped(NLModuleParametrized):
    
    def forward(self, structures: Iterable[Structure]) -> Tensor:
        input = self._collect_inputs(structures)
        
        weighted_input = 1 - (1 - input) * self.weight
        self.last_output = torch.prod(weighted_input, -1, keepdim=True)

        return self.last_output


class NLAndBiProd(NLModuleParametrized):
    
    def forward(self, structures: Iterable[Structure]) -> Tensor:
        input = self._collect_inputs(structures)
        
        scaled_weight = torch.tanh(self.weight)
        zeros = torch.zeros_like(scaled_weight)
        weighted_input = (1
                          - (1 - input) * torch.maximum(scaled_weight, zeros)
                          + input * torch.minimum(scaled_weight, zeros))

        self.last_output = torch.prod(weighted_input, -1, keepdim=True)

        return self.last_output


class NLAndBiProdClamped(NLModuleParametrized):
    
    def forward(self, structures: Iterable[Structure]) -> Tensor:
        input = self._collect_inputs(structures)
        
        zeros = torch.zeros_like(self.weight)
        weighted_input = (1
                          - (1 - input) * torch.maximum(self.weight, zeros)
                          + input * torch.minimum(self.weight, zeros))

        self.last_output = torch.prod(weighted_input, -1, keepdim=True)

        return self.last_output


class NLAndBiGodel(NLModuleParametrized):

    epsilon: float

    def __init__(self, epsilon: float = 1e-2, **kwargs):
        super().__init__(**kwargs)

        self.epsilon = epsilon
    
    def forward(self, structures: Iterable[Structure]) -> Tensor:
        input = self._collect_inputs(structures)
        
        scaled_weight = torch.tanh(self.weight)
        zeros = torch.zeros_like(scaled_weight)
        weighted_input = (1
                          - (1 - input) * torch.maximum(scaled_weight, zeros)
                          + input * torch.minimum(scaled_weight, zeros))

        min_threshold = torch.min(weighted_input, -1).values + self.epsilon
        min_mask = (weighted_input.T < min_threshold).T
        mins = min_mask * weighted_input

        self.last_output = torch.mean(mins, -1, keepdim=True)

        return self.last_output


class NLAndBiMixed(NLModuleParametrized):

    epsilon: float

    def __init__(self, epsilon: float = 1e-7, **kwargs):
        super().__init__(**kwargs)

        self.epsilon = epsilon
    
    def forward(self, structures: Iterable[Structure]) -> Tensor:
        input = self._collect_inputs(structures).to(self.weight)
        
        scaled_weight = torch.tanh(self.weight)
        zeros = torch.zeros_like(scaled_weight)
        weighted_input = (1
                          - (1 - input) * torch.maximum(scaled_weight, zeros)
                          + input * torch.minimum(scaled_weight, zeros))

        product_and = torch.prod(weighted_input, -1, keepdim=True)
        vanishing_grad = product_and < self.epsilon

        if torch.any(vanishing_grad):
            min_threshold = torch.min(weighted_input, -1).values + self.epsilon
            min_mask = (weighted_input.T < min_threshold).T
            godel_and = torch.mean(min_mask * weighted_input, -1, keepdim=True)

            self.last_output = torch.where(vanishing_grad, godel_and, product_and)
        else:
            self.last_output = product_and

        return self.last_output


class NLXorBiMixed(NLModuleParametrized):

    epsilon: float

    def __init__(self, epsilon: float = 1e-7, **kwargs):
        super().__init__(**kwargs)

        self.epsilon = epsilon
    
    def forward(self, structures: Iterable[Structure]) -> Tensor:
        input = self._collect_inputs(structures)
        
        tiling = [1] * input.shape + [1]
        tiling[-2] = input.shape[-1]
        clauses_full_grad = torch.tile((1 - torch.unsqueeze(input, -2)), tiling)
        clauses_full_grad[:, range(input.shape[-1]), range(input.shape[-1])] = input

        grad_mask = torch.tril(torch.ones((input.shape[-1], input.shape[-1]))).T.type(torch.bool)
        clauses = torch.where(grad_mask, clauses_full_grad, clauses_full_grad.detach())

        clauses_and = torch.prod(clauses, -1)
        vanishing_grad = clauses_and < self.epsilon

        if torch.any(vanishing_grad):
            min_threshold = torch.min(clauses, -1).values + self.epsilon
            min_mask = (clauses.T < min_threshold).T
            godel_and = torch.mean(min_mask * clauses, -1)

            clauses_and = torch.where(vanishing_grad, godel_and, clauses_and)

        negated_or = torch.prod(1 - clauses_and, -1, keepdim=True)
        vanishing_grad = negated_or < self.epsilon

        if torch.any(vanishing_grad):
            min_threshold = torch.min(1 - clauses_and, -1).values + self.epsilon
            min_mask = ((1 - clauses_and).T < min_threshold).T
            godel_and = torch.mean(min_mask * (1 - clauses_and), -1, keepdim=True)

            negated_or = torch.where(vanishing_grad, godel_and, negated_or)

        return 1 - negated_or
