from typing import Iterable, Tuple

import torch
from torch import Tensor

from nesyrl.logic.neural.base import NLModuleParametrized
from nesyrl.logic.propositional import Structure


# class MixedAnd(torch.autograd.Function):

#     @staticmethod
#     def forward(ctx, input, epsilon: float, keepdim: bool):
#         product_and = torch.prod(input, -1, keepdim=keepdim)
#         # vanishing_grad = product_and < epsilon

#         ctx.epsilon = epsilon
#         ctx.save_for_backward(input)

#         return product_and

#     @staticmethod
#     def backward(ctx, grad_output):
#         epsilon = ctx.epsilon
#         input, = ctx.saved_tensors

#         tiling = [1] * len(input.shape) + [1]
#         tiling[-2] = input.shape[-1]
#         product_backward = torch.tile(torch.unsqueeze(input, -2), tiling)
#         product_backward[:, range(input.shape[-1]), range(input.shape[-1])] = torch.ones_like(input)
#         product_backward = torch.prod(product_backward, -1)
#         grad_input = torch.clip(product_backward, epsilon / (input + epsilon))

#         return grad_output * grad_input, None, None


class LukaAnd(torch.autograd.Function):

    @staticmethod
    def forward(ctx, input: Tensor, keepdim: bool) -> Tensor:
        n = input.shape[-1]
        luka_and = torch.sum(input, -1, keepdim=keepdim) - n + 1
        luka_and = torch.clip(luka_and, min=0.0)

        ctx.save_for_backward(input)

        return luka_and

    @staticmethod
    def backward(ctx, grad_output: Tensor) -> Tuple[Tensor, None]:
        input, = ctx.saved_tensors

        grad_input = torch.ones_like(input)

        return grad_output * grad_input, None


# class PreservingClamp(torch.autograd.Function):

#     @staticmethod
#     def forward(ctx, input: Tensor, min: Tensor | None = None, max: Tensor | None = None) -> Tensor:
#         return torch.clamp(input, min, max)

#     @staticmethod
#     def backward(ctx, grad_output: Tensor) -> Tuple[Tensor, None, None]:
#         return grad_output, None, None


class NLAndProd(NLModuleParametrized):
    
    def forward(self, structures: Iterable[Structure]) -> Tensor:
        input = self._collect_inputs(structures)
        
        scaled_weight = torch.sigmoid(self.weight)
        weighted_input = 1 - (1 - input) * scaled_weight
        self.last_output = torch.prod(weighted_input, -1, keepdim=True)

        return self.last_output


class NLAndLuka(NLModuleParametrized):
    
    def forward(self, structures: Iterable[Structure]) -> Tensor:
        input = self._collect_inputs(structures)
        
        scaled_weight = torch.sigmoid(self.weight)
        weighted_input = 1 - (1 - input) * scaled_weight
        
        self.last_output = LukaAnd.apply(weighted_input, True)

        return self.last_output


# class NLAndProdClamped(NLModuleParametrized):
    
#     def forward(self, structures: Iterable[Structure]) -> Tensor:
#         input = self._collect_inputs(structures)
        
#         weighted_input = 1 - (1 - input) * self.weight
#         self.last_output = torch.prod(weighted_input, -1, keepdim=True)

#         return self.last_output


# class NLAndMixed(NLModuleParametrized):

#     epsilon: float

#     def __init__(self, epsilon: float = 1e-5, **kwargs):
#         super().__init__(**kwargs)

#         self.epsilon = epsilon
    
#     def forward(self, structures: Iterable[Structure]) -> Tensor:
#         input = self._collect_inputs(structures)
        
#         scaled_weight = torch.sigmoid(self.weight)
#         weighted_input = 1 - (1 - input) * scaled_weight

#         product_and = torch.prod(weighted_input, -1, keepdim=True)
#         vanishing_grad = product_and < self.epsilon

#         if torch.any(vanishing_grad):
#             min_threshold = torch.min(weighted_input, -1, keepdim=True).values + self.epsilon
#             min_mask = weighted_input < min_threshold
#             godel_and = min_mask * weighted_input / torch.sum(min_mask, -1, keepdim=True)
#             godel_and = torch.sum(godel_and, -1, keepdim=True)

#             self.last_output = torch.where(vanishing_grad, godel_and, product_and)
#         else:
#             self.last_output = product_and

#         return self.last_output


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


# class NLAndBiProd2(NLModuleParametrized):

#     epsilon: float
#     _last_input: Tensor

#     def __init__(self, epsilon: float = 1e-5, **kwargs):
#         super().__init__(**kwargs)

#         self.epsilon = epsilon
#         self.weight.register_hook(self._push_grad)

#     def _push_grad(self, grad: Tensor) -> Tensor:
#         is_weight_zero = torch.abs(torch.tanh(self.weight)) < self.epsilon
#         is_grad_zero = grad < self.epsilon
        
#         pushed = torch.clone(grad)
#         mask = is_weight_zero & is_grad_zero
#         pushed[mask] = (2 * self._last_input.mean(0)[mask] - 1) * self.epsilon

#         return pushed
    
#     def forward(self, structures: Iterable[Structure]) -> Tensor:
#         input = self._collect_inputs(structures)
#         self._last_input = input
        
#         scaled_weight = torch.tanh(self.weight)
#         zeros = torch.zeros_like(scaled_weight)
#         weighted_input = (1
#                           - ((1 - input) ** 2) * (torch.maximum(scaled_weight, zeros) ** 2)
#                           - (input ** 2) * (torch.minimum(scaled_weight, zeros) ** 2))

#         self.last_output = torch.prod(weighted_input, -1, keepdim=True)

#         return self.last_output


# class NLAndBiProdClamped(NLModuleParametrized):
    
#     def forward(self, structures: Iterable[Structure]) -> Tensor:
#         input = self._collect_inputs(structures)
        
#         zeros = torch.zeros_like(self.weight)
#         weighted_input = (1
#                           - (1 - input) * torch.maximum(self.weight, zeros)
#                           + input * torch.minimum(self.weight, zeros))

#         self.last_output = torch.prod(weighted_input, -1, keepdim=True)

#         return self.last_output


# class NLAndBiGodel(NLModuleParametrized):

#     epsilon: float

#     def __init__(self, epsilon: float = 1e-2, **kwargs):
#         super().__init__(**kwargs)

#         self.epsilon = epsilon
    
#     def forward(self, structures: Iterable[Structure]) -> Tensor:
#         input = self._collect_inputs(structures)
        
#         scaled_weight = torch.tanh(self.weight)
#         zeros = torch.zeros_like(scaled_weight)
#         weighted_input = (1
#                           - (1 - input) * torch.maximum(scaled_weight, zeros)
#                           + input * torch.minimum(scaled_weight, zeros))

#         min_threshold = torch.min(weighted_input, -1, keepdim=True).values + self.epsilon
#         min_mask = weighted_input < min_threshold
#         mins = min_mask * weighted_input

#         self.last_output = torch.mean(mins, -1, keepdim=True)

#         return self.last_output


class NLAndBiLuka(NLModuleParametrized):
    
    def forward(self, structures: Iterable[Structure]) -> Tensor:
        input = self._collect_inputs(structures)
        
        scaled_weight = torch.tanh(self.weight)
        zeros = torch.zeros_like(scaled_weight)
        weighted_input = (1
                          - (1 - input) * torch.maximum(scaled_weight, zeros)
                          + input * torch.minimum(scaled_weight, zeros))
        
        self.last_output = LukaAnd.apply(weighted_input, True)

        return self.last_output


# class NLAndBiLuka2(NLModuleParametrized):
    
#     epsilon: float
#     _last_input: Tensor

#     def __init__(self, epsilon: float = 1e-5, **kwargs):
#         super().__init__(**kwargs)

#         self.epsilon = epsilon
#         self.weight.register_hook(self._push_grad)

#     def _push_grad(self, grad: Tensor) -> Tensor:
#         is_weight_zero = torch.abs(torch.tanh(self.weight)) < self.epsilon
#         is_grad_zero = grad < self.epsilon
        
#         pushed = torch.clone(grad)
#         mask = is_weight_zero & is_grad_zero
#         pushed[mask] = (2 * self._last_input.mean(0)[mask] - 1) * self.epsilon

#         return pushed
    
#     def forward(self, structures: Iterable[Structure]) -> Tensor:
#         input = self._collect_inputs(structures)
#         self._last_input = input
        
#         scaled_weight = torch.tanh(self.weight)
#         zeros = torch.zeros_like(scaled_weight)
#         weighted_input = (1
#                           - ((1 - input) ** 2) * (torch.maximum(scaled_weight, zeros) ** 2)
#                           - (input ** 2) * (torch.minimum(scaled_weight, zeros) ** 2))
        
#         self.last_output = LukaAnd.apply(weighted_input, True)

#         return self.last_output


# class NLAndBiLuka2(NLModuleParametrized):
    
#     def forward(self, structures: Iterable[Structure]) -> Tensor:
#         input = self._collect_inputs(structures)
        
#         scaled_weight = torch.tanh(self.weight)
#         input_neg = input < 0.5
#         weight_neg = scaled_weight < 0

#         w1 = 1 - (1 - 2 * input * scaled_weight) * (1 - 2 * input) * scaled_weight
#         w2 = 1 - (1 + 2 * input * scaled_weight) * (1 - 2 * input) * scaled_weight
#         w3 = 1 + (1 - (1 - (2 * input - 1)) * scaled_weight) * (2 * input - 1) * scaled_weight
#         w4 = 1 + (1 + (1 - (2 * input - 1)) * scaled_weight) * (2 * input - 1) * scaled_weight

#         weighted_input = torch.zeros_like(input)
#         weighted_input[input_neg & ~weight_neg] = w1[input_neg & ~weight_neg]
#         weighted_input[input_neg & weight_neg] = w2[input_neg & weight_neg]
#         weighted_input[~input_neg & ~weight_neg] = w3[~input_neg & ~weight_neg]
#         weighted_input[~input_neg & weight_neg] = w4[~input_neg & weight_neg]
#         weighted_input = PreservingClamp.apply(weighted_input, None, 1.0)

#         self.last_output = LukaAnd.apply(weighted_input, True)

#         return self.last_output


# class NLAndBiMixed(NLModuleParametrized):

#     epsilon: float

#     def __init__(self, epsilon: float = 1e-5, **kwargs):
#         super().__init__(**kwargs)

#         self.epsilon = epsilon
    
#     def forward(self, structures: Iterable[Structure]) -> Tensor:
#         input = self._collect_inputs(structures)
        
#         scaled_weight = torch.tanh(self.weight)
#         zeros = torch.zeros_like(scaled_weight)
#         weighted_input = (1
#                           - (1 - input) * torch.maximum(scaled_weight, zeros)
#                           + input * torch.minimum(scaled_weight, zeros))
        
#         self.last_output = MixedAnd.apply(weighted_input, self.epsilon, True)

#         # product_and = torch.prod(weighted_input, -1, keepdim=True)
#         # vanishing_grad = product_and < self.epsilon

#         # if torch.any(vanishing_grad):
#         #     min_threshold = torch.min(weighted_input, -1, keepdim=True).values + self.epsilon
#         #     min_mask = weighted_input < min_threshold
#         #     godel_and = min_mask * weighted_input / torch.sum(min_mask, -1, keepdim=True)
#         #     godel_and = torch.sum(godel_and, -1, keepdim=True)

#         #     self.last_output = torch.where(vanishing_grad, godel_and, product_and)
#         # else:
#         #     self.last_output = product_and

#         return self.last_output


class NLOrClamped(NLModuleParametrized):

    def _and(self, input: Tensor, weight: Tensor) -> Tensor:
        weighted_input = 1 - (1 - input) * weight
        
        return torch.prod(weighted_input, -1, keepdim=True)
    
    def forward(self, structures: Iterable[Structure]) -> Tensor:
        input = self._collect_inputs(structures)

        self.last_output = 1 - self._and(1 - input, self.weight)

        return self.last_output


class NLOrLukaClamped(NLModuleParametrized):
    
    def forward(self, structures: Iterable[Structure]) -> Tensor:
        input = self._collect_inputs(structures)

        weighted_input = 1 - input * self.weight
        self.last_output = 1 - LukaAnd.apply(weighted_input, True)

        return self.last_output


# class NLOrMixedClamped(NLModuleParametrized):

#     epsilon: float

#     def __init__(self, epsilon: float = 1e-5, **kwargs):
#         super().__init__(**kwargs)

#         self.epsilon = epsilon
    
#     def _and(self, input: Tensor, weight: Tensor) -> Tensor:
#         weighted_input = 1 - (1 - input) * weight

#         product_and = torch.prod(weighted_input, -1, keepdim=True)
#         vanishing_grad = product_and < self.epsilon

#         if torch.any(vanishing_grad):
#             min_threshold = torch.min(weighted_input, -1, keepdim=True).values + self.epsilon
#             min_mask = weighted_input < min_threshold
#             godel_and = min_mask * weighted_input / torch.sum(min_mask, -1, keepdim=True)
#             godel_and = torch.sum(godel_and, -1, keepdim=True)

#             return torch.where(vanishing_grad, godel_and, product_and)

#         return product_and
    
#     def forward(self, structures: Iterable[Structure]) -> Tensor:
#         input = self._collect_inputs(structures)

#         self.last_output = 1 - self._and(1 - input, self.weight)

#         return self.last_output


class NLXorClamped(NLModuleParametrized):

    def _and(self, input: Tensor, weight: Tensor, keepdim: bool) -> Tensor:
        weighted_input = 1 - (1 - input) * weight
        
        return torch.prod(weighted_input, -1, keepdim=keepdim)
    
    def forward(self, structures: Iterable[Structure]) -> Tensor:
        input = self._collect_inputs(structures)
        
        tiling = [1] * len(input.shape) + [1]
        tiling[-2] = input.shape[-1]
        clauses_full_grad = torch.tile((1 - torch.unsqueeze(input, -2)), tiling)
        clauses_full_grad[:, range(input.shape[-1]), range(input.shape[-1])] = input

        # grad_mask = torch.tril(torch.ones((input.shape[-1], input.shape[-1]))).T.to(input).type(torch.bool)
        # clauses = torch.where(grad_mask, clauses_full_grad, clauses_full_grad.detach())
        clauses = clauses_full_grad
        
        clauses_and = self._and(clauses, torch.ones_like(clauses[0]).to(input), False)
        self.last_output = 1 - self._and(1 - clauses_and, self.weight, True)
    
        return self.last_output
    

# class NLXorMixedClamped(NLModuleParametrized):

#     epsilon: float

#     def __init__(self, epsilon: float = 1e-5, **kwargs):
#         super().__init__(**kwargs)

#         self.epsilon = epsilon
    
#     def _and(self, input: Tensor, weight: Tensor, keepdim: bool) -> Tensor:
#         weighted_input = 1 - (1 - input) * weight

#         product_and = torch.prod(weighted_input, -1, keepdim=keepdim)
#         vanishing_grad = product_and < self.epsilon

#         if torch.any(vanishing_grad):
#             min_threshold = torch.min(weighted_input, -1, keepdim=True).values + self.epsilon
#             min_mask = weighted_input < min_threshold
#             godel_and = min_mask * weighted_input / torch.sum(min_mask, -1, keepdim=True)
#             godel_and = torch.sum(godel_and, -1, keepdim=keepdim)

#             return torch.where(vanishing_grad, godel_and, product_and)

#         return product_and
    
#     def forward(self, structures: Iterable[Structure]) -> Tensor:
#         input = self._collect_inputs(structures)

#         tiling = [1] * len(input.shape) + [1]
#         tiling[-2] = input.shape[-1]
#         clauses_full_grad = torch.tile((1 - torch.unsqueeze(input, -2)), tiling)
#         clauses_full_grad[:, range(input.shape[-1]), range(input.shape[-1])] = input

#         grad_mask = torch.tril(torch.ones((input.shape[-1], input.shape[-1]))).T.to(input).type(torch.bool)
#         clauses = torch.where(grad_mask, clauses_full_grad, clauses_full_grad.detach())
        
#         clauses_and = self._and(clauses, torch.ones_like(clauses[0]).to(input), False)
#         self.last_output = 1 - self._and(1 - clauses_and, self.weight, True)

#         return self.last_output


class NLXorLukaClamped(NLModuleParametrized):
    
    def forward(self, structures: Iterable[Structure]) -> Tensor:
        input = self._collect_inputs(structures)

        tiling = [1] * len(input.shape) + [1]
        tiling[-2] = input.shape[-1]
        clauses_full_grad = torch.tile((1 - torch.unsqueeze(input, -2)), tiling)
        clauses_full_grad[:, range(input.shape[-1]), range(input.shape[-1])] = input

        # grad_mask = torch.tril(torch.ones((input.shape[-1], input.shape[-1]))).T.to(input).type(torch.bool)
        # clauses = torch.where(grad_mask, clauses_full_grad, clauses_full_grad.detach())
        clauses = clauses_full_grad
        
        weighted = 1 - (1 - clauses) * torch.ones_like(clauses[0]).to(input)
        clauses_and = LukaAnd.apply(weighted, False)
        weighted = 1 - clauses_and * self.weight
        self.last_output = 1 - LukaAnd.apply(weighted, True)

        return self.last_output


# class NLXorBiMixed(NLModuleParametrized):

#     epsilon: float

#     def __init__(self, epsilon: float = 1e-5, **kwargs):
#         super().__init__(**kwargs)

#         self.epsilon = epsilon
    
#     def forward(self, structures: Iterable[Structure]) -> Tensor:
#         input = self._collect_inputs(structures)
        
#         tiling = [1] * input.shape + [1]
#         tiling[-2] = input.shape[-1]
#         clauses_full_grad = torch.tile((1 - torch.unsqueeze(input, -2)), tiling)
#         clauses_full_grad[:, range(input.shape[-1]), range(input.shape[-1])] = input

#         grad_mask = torch.tril(torch.ones((input.shape[-1], input.shape[-1]))).T.type(torch.bool)
#         clauses = torch.where(grad_mask, clauses_full_grad, clauses_full_grad.detach())

#         clauses_and = torch.prod(clauses, -1)
#         vanishing_grad = clauses_and < self.epsilon

#         if torch.any(vanishing_grad):
#             min_threshold = torch.min(clauses, -1).values + self.epsilon
#             min_mask = (clauses.T < min_threshold).T
#             godel_and = torch.mean(min_mask * clauses, -1)

#             clauses_and = torch.where(vanishing_grad, godel_and, clauses_and)

#         negated_or = torch.prod(1 - clauses_and, -1, keepdim=True)
#         vanishing_grad = negated_or < self.epsilon

#         if torch.any(vanishing_grad):
#             min_threshold = torch.min(1 - clauses_and, -1).values + self.epsilon
#             min_mask = ((1 - clauses_and).T < min_threshold).T
#             godel_and = torch.mean(min_mask * (1 - clauses_and), -1, keepdim=True)

#             negated_or = torch.where(vanishing_grad, godel_and, negated_or)

#         return 1 - negated_or
