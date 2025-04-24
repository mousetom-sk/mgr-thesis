from typing import Tuple, Any

# import matplotlib.pyplot as plt

from abc import ABC, abstractmethod

import torch
from torch import Tensor
from torch.nn import Module, Parameter, Sequential, Linear
from torch.optim import Optimizer

from tianshou.data import Batch

from nesyrl.envs.symbolic import SymbolicEnvironment, StateAtom
from nesyrl.envs.symbolic.blocks_world import Move, On, Top
from nesyrl.logic.propositional import *
from nesyrl.logic.neural import *
from nesyrl.util.activations import *


class NLActor(ABC, Module):

    @abstractmethod
    def compute_regularization_loss(self) -> Tensor:
        pass
    
    @abstractmethod
    def postprocess_parameters(self) -> None:
        pass

    @abstractmethod
    def params_str(self) -> str:
        pass


class NLCritic(ABC, Module):

    @abstractmethod
    def compute_regularization_loss(self) -> Tensor:
        pass
    
    @abstractmethod
    def postprocess_parameters(self) -> None:
        pass

    @abstractmethod
    def params_str(self) -> str:
        pass


class Actor(NLActor):

    nln: NLNetwork
    net: Module

    _domain_atoms: List[StateAtom]
    _semantics: FuzzySemantics

    def __init__(
        self, env: SymbolicEnvironment,
        and_impl: NLModuleParametrized, and_initializer: WeightInitializer,
        and_postprocessor: WeightPostprocessor = NoPostprocessor(),
        and_regularizer: WeightRegularizer = NoRegularizer(),
        negated_inputs: bool = False, assume_false: bool = False, drop_false: bool = False,
        device: str | int | torch.device = "cpu"
    ) -> None:
        super().__init__()

        nl_atoms = []
        
        for atom in env.state_atoms:
            if atom not in env.domain_atoms:
                if isinstance(atom, Contradiction):
                    if drop_false:
                        continue
                    
                    false_idx = len(nl_atoms) - 1

                nl_atoms.append(NLPredicateAtom(atom.__class__, *atom.args))
        
        if negated_inputs:
            nl_atoms += [NLNegatedPredicateAtom(atom.__class__, *atom.args)
                         for atom in env.state_atoms
                         if atom not in env.domain_atoms
                         and (not drop_false or not isinstance(atom, Contradiction))]

        nl_actions = []
        for _ in env.action_atoms:
            nl_act = and_impl(operands=nl_atoms,
                              weight_initializer=and_initializer,
                              weight_regularizer=and_regularizer,
                              weight_postprocessor=and_postprocessor)
            nl_actions.append(nl_act)
        
        self.nln = NLNetwork(NLStack(nl_actions))
        self.nln.reset_parameters()

        if assume_false:
            if isinstance(and_regularizer, UncertaintyRegularizer):
                val = and_regularizer.peak
            else:
                val = 0.5

            weight = torch.tensor([val if i == false_idx else float("nan")
                                   for i in range(len(nl_atoms))])
            
            for nl_act in nl_actions:
                nl_act.set_parameters(weight)
        
        self.net = Sequential(self.nln, ScaledSoftmax(10))
        self.net = self.net.to(device)

        self.add_module("net", self.net)

        self._domain_atoms = env.domain_atoms
        self._semantics = env.semantics


        # plt.ion()
        # plt.rcParams['figure.figsize'] = [18, 8]
        # fig, self._ax = plt.subplots()
        # self._bars = None

        # self._ax.set_xlabel("State Atoms")
        # self._ax.set_xticks(range(len(nl_atoms)), [str(a) for a in env.state_atoms
        #                                            if a not in env.domain_atoms
        #                                            and (not drop_false or not isinstance(a, Contradiction))])
        # self._ax.set_ylabel("Scaled Weight")
        # self._ax.set_yticks(torch.arange(-10, 11) / 10)
        # self._ax.set_ylim(-1, 1)
        # self._ax.grid()
        # self._i = 0

    def forward(self, obs: Iterable[Valuation] | Batch, state: Any = None, info: Dict[str, Any] = {}) -> Tuple[Tensor, Any]:
        obs = list(map(self._valuation_to_structure, obs))
        probs = self.net.forward(obs)

        # if self._i % 10 == 0:
        #     scaled_weight = torch.tanh(self.nln.head.nl_modules[0].weight).cpu().detach()

        #     if self._bars:
        #         self._bars.remove()
            
        #     self._bars = self._ax.bar(range(len(scaled_weight)), scaled_weight, color="tab:blue")
            
        #     plt.pause(0.05)
            
        # self._i += 1

        return probs, state
    
    def _valuation_to_structure(self, obs: Valuation | Batch) -> Structure:
        return Structure.from_valuation(dict(obs), self._domain_atoms, self._semantics)
    
    def compute_regularization_loss(self) -> Tensor:
        return self.nln.compute_regularization_loss()
    
    def postprocess_parameters(self) -> None:
        return self.nln.postprocess_parameters()

    def params_str(self) -> str:
        return self.nln.params_str()


class ActorValidity(NLActor):

    nln: NLNetwork
    net: Module

    _domain_atoms: List[StateAtom]
    _semantics: FuzzySemantics

    def __init__(
        self, env: SymbolicEnvironment,
        and_impl: NLModuleParametrized, and_initializer: WeightInitializer,
        high_value: float, device: str | int | torch.device = "cpu"
    ) -> None:
        super().__init__()

        nl_atoms = []
        
        for atom in env.state_atoms:
            if atom not in env.domain_atoms:
                if isinstance(atom, Contradiction):
                    continue

                nl_atoms.append(NLPredicateAtom(atom.__class__, *atom.args))

        nl_actions = []
        for _ in env.action_atoms:
            nl_act = and_impl(operands=nl_atoms,
                              weight_initializer=and_initializer)
            nl_actions.append(nl_act)
        
        self.nln = NLNetwork(NLStack(nl_actions))
        self.nln.reset_parameters()

        for i, atom in enumerate(env.action_atoms):
            if isinstance(atom, Move):
                if atom.args[1] == env._table:
                    top_x = Top(atom.args[0])
                    on_xy = On(*atom.args)

                    vals = {str(top_x): high_value, str(on_xy): -high_value}
                else:
                    top_x = Top(atom.args[0])
                    top_y = Top(atom.args[1])

                    vals = {str(top_x): high_value, str(top_y): high_value}
                
                weight = torch.tensor([vals.get(str(a), float("nan")) for a in nl_atoms])
                nl_actions[i].set_parameters(weight)
        
        self.net = Sequential(self.nln, ScaledSoftmax(10))
        self.net = self.net.to(device)

        self.add_module("net", self.net)

        self._domain_atoms = env.domain_atoms
        self._semantics = env.semantics


        # plt.ion()
        # plt.rcParams['figure.figsize'] = [18, 8]
        # fig, self._ax = plt.subplots()
        # self._bars = None

        # self._ax.set_xlabel("State Atoms")
        # self._ax.set_xticks(range(len(nl_atoms)), [str(a) for a in env.state_atoms
        #                                            if a not in env.domain_atoms
        #                                            and not isinstance(a, Contradiction)])
        # self._ax.set_ylabel("Scaled Weight")
        # self._ax.set_yticks(torch.arange(-10, 11) / 10)
        # self._ax.set_ylim(-1, 1)
        # self._ax.grid()
        # self._i = 0

    def forward(self, obs: Iterable[Valuation] | Batch, state: Any = None, info: Dict[str, Any] = {}) -> Tuple[Tensor, Any]:
        obs = list(map(self._valuation_to_structure, obs))
        probs = self.net.forward(obs)

        # if self._i % 10 == 0:
        #     scaled_weight = torch.tanh(self.nln.head.nl_modules[0].weight).cpu().detach()

        #     if self._bars:
        #         self._bars.remove()
            
        #     self._bars = self._ax.bar(range(len(scaled_weight)), scaled_weight, color="tab:blue")
            
        #     plt.pause(0.05)
            
        # self._i += 1

        return probs, state
    
    def _valuation_to_structure(self, obs: Valuation | Batch) -> Structure:
        return Structure.from_valuation(dict(obs), self._domain_atoms, self._semantics)
    
    def compute_regularization_loss(self) -> Tensor:
        return self.nln.compute_regularization_loss()
    
    def postprocess_parameters(self) -> None:
        return self.nln.postprocess_parameters()

    def params_str(self) -> str:
        return self.nln.params_str()


class ActorMulti(NLActor):

    nln: NLNetwork
    net: Module

    _domain_atoms: List[StateAtom]
    _semantics: FuzzySemantics

    def __init__(
        self, env: SymbolicEnvironment,
        and_impl: NLModuleParametrized, and_initializer: WeightInitializer,
        num_ands: int, or_impl: NLModuleParametrized,
        inject_validity: bool = False, high_value: float = 2.0,
        device: str | int | torch.device = "cpu"
    ) -> None:
        super().__init__()

        nl_atoms = []
        
        for atom in env.state_atoms:
            if atom not in env.domain_atoms:
                if isinstance(atom, Contradiction):
                    continue

                nl_atoms.append(NLPredicateAtom(atom.__class__, *atom.args))

        nl_actions = []
        for _ in env.action_atoms:
            nl_act = or_impl(operands=[and_impl(operands=nl_atoms,
                                                weight_initializer=and_initializer)
                                       for _ in range(num_ands)],
                             weight_initializer=ConstantInitializer(1),
                             trainable=False)
            nl_actions.append(nl_act)
        
        self.nln = NLNetwork(NLStack(nl_actions))
        self.nln.reset_parameters()
        
        if inject_validity:
            for i, atom in enumerate(env.action_atoms):
                if isinstance(atom, Move):
                    if atom.args[1] == env._table:
                        top_x = Top(atom.args[0])
                        on_xy = On(*atom.args)

                        vals = {str(top_x): high_value, str(on_xy): -high_value}
                    else:
                        top_x = Top(atom.args[0])
                        top_y = Top(atom.args[1])

                        vals = {str(top_x): high_value, str(top_y): high_value}
                    
                    weight = torch.tensor([vals.get(str(a), float("nan")) for a in nl_atoms])
                    for m in nl_actions[i].nl_modules:
                        m.set_parameters(weight)
        
        self.net = Sequential(self.nln, ScaledSoftmax(10))
        self.net = self.net.to(device)

        self.add_module("net", self.net)

        self._domain_atoms = env.domain_atoms
        self._semantics = env.semantics

        # plt.ion()
        # plt.rcParams['figure.figsize'] = [18, 8]
        # fig, self._ax = plt.subplots()
        # self._bars = []

        # self._ax.set_xlabel("State Atoms")
        # self._ax.set_xticks(range(len(nl_atoms)), [str(a) for a in env.state_atoms
        #                                            if a not in env.domain_atoms
        #                                            and (not drop_false or not isinstance(a, Contradiction))])
        # self._ax.set_ylabel("Scaled Weight")
        # self._ax.set_yticks(torch.arange(-10, 11) / 10)
        # self._ax.set_ylim(-1, 1)
        # self._ax.grid()
        # self._i = 0

    def forward(self, obs: Iterable[Valuation] | Batch, state: Any = None, info: Dict[str, Any] = {}) -> Tuple[Tensor, Any]:
        obs = list(map(self._valuation_to_structure, obs))
        probs = self.net.forward(obs)

        # if self._i % 10 == 0:
        #     weight = torch.stack([m.weight for m in self.nln.head.nl_modules[0].nl_modules])
        #     scaled_weight = torch.tanh(weight).cpu().detach()

        #     if len(self._bars) > 0:
        #         for b in self._bars: b.remove()
            
        #     width = 1 / len(scaled_weight)
        #     for i, w in enumerate(scaled_weight):
        #         if len(self._bars) == i: self._bars.append(None)

        #         self._bars[i] = self._ax.bar([j + i * width for j in range(len(w))], w, width=width, color=f"C{i}")
            
        #     plt.pause(0.05)
            
        # self._i += 1

        return probs, state
    
    def _valuation_to_structure(self, obs: Valuation | Batch) -> Structure:
        return Structure.from_valuation(dict(obs), self._domain_atoms, self._semantics)
    
    def compute_regularization_loss(self) -> Tensor:
        return self.nln.compute_regularization_loss()
    
    def postprocess_parameters(self) -> None:
        return self.nln.postprocess_parameters()

    def params_str(self) -> str:
        return self.nln.params_str()


class ActorMLP(Module):

    mlp: Module
    v: Tensor

    def __init__(self, mlp: Module) -> None:
        super().__init__()
        
        self.mlp = mlp

    def forward(self, obs: Iterable[Valuation] | Batch, **kwargs) -> Tensor:
        obs_vec = []
        
        for valuation in obs:
            sorted_state = sorted(dict(valuation).items(), key=lambda kv: kv[0])
            state_vec = torch.tensor(list(map(lambda kv: int(kv[1]), sorted_state)))
            obs_vec.append(state_vec)

        return self.mlp(torch.stack(obs_vec))


# class Critic(NLCritic):

#     nln: NLNetwork
#     combiner: Module
#     net: Module
    
#     _domain_atoms: List[StateAtom]
#     _semantics: FuzzySemantics

#     def __init__(
#         self, env: SymbolicEnvironment,
#         num_rules: int, and_initializer: WeightInitializer, assume_false: bool,
#         device: str | int | torch.device
#     ) -> None:
#         super().__init__()

#         nl_atoms = [NLPredicateAtom(atom.__class__, *atom.args)
#                     for atom in env.state_atoms
#                     if atom not in env.domain_atoms]    

#         nl_ands = [NLAndBiProd(operands=nl_atoms, weight_initializer=and_initializer)
#                    for _ in range(num_rules)]
        
#         self.nln = NLNetwork(NLStack(nl_ands))
#         self.nln.reset_parameters()

#         if assume_false:
#             false_idx = nl_atoms.index(Contradiction())
#             weight = torch.tensor([2.0 if i == false_idx else float("nan")
#                                    for i in range(len(nl_atoms))])
            
#             for nl_act in nl_ands:
#                 nl_act.set_parameters(weight)
        
#         self.combiner = Linear(num_rules, 1, False)
#         torch.nn.init.uniform_(self.combiner.weight, -0.5, 0.5)

#         self.net = Sequential(self.nln, self.combiner)
#         self.net = self.net.to(device)

#         self.add_module("net", self.net)

#         self._domain_atoms = env.domain_atoms
#         self._semantics = env.semantics

#     def forward(self, obs: Iterable[Valuation] | Batch, **kwargs) -> Tensor:
#         obs = list(map(self._valuation_to_structure, obs))
#         vals = self.net.forward(obs)

#         return vals
    
#     def _valuation_to_structure(self, obs: Valuation | Batch) -> Structure:
#         return Structure.from_valuation(dict(obs), self._domain_atoms, self._semantics)
    
#     def compute_regularization_loss(self) -> Tensor:
#         return self.nln.compute_regularization_loss()
    
#     def postprocess_parameters(self) -> None:
#         return self.nln.postprocess_parameters()

#     def params_str(self) -> str:
#         return self.nln.params_str() + "\n\n" + "Combiner" + "\n" + str(self.combiner.weight)


class CriticTab(NLCritic):

    mask: Tensor
    v: Tensor

    def __init__(self, env: SymbolicEnvironment, device: str | int | torch.device) -> None:
        super().__init__()

        sorted_states = [sorted(s.items(), key=lambda kv: kv[0]) for s in env.all_states]
        self.mask = torch.tensor([list(map(lambda kv: int(kv[1]), s)) for s in sorted_states], device=device)
        self.mask = 2 * self.mask - 1
        self.v = Parameter(torch.zeros((self.mask.shape[0],), device=device), requires_grad=True)

    def forward(self, obs: Iterable[Valuation] | Batch, **kwargs) -> Tensor:
        vals = []
        
        for valuation in obs:
            sorted_state = sorted(dict(valuation).items(), key=lambda kv: kv[0])
            state_vec = torch.tensor(list(map(lambda kv: int(kv[1]), sorted_state))).to(self.mask)
            zeros = torch.zeros_like(self.mask)
            masked_state = (1
                            - torch.maximum(self.mask, zeros) * (1 - state_vec)
                            + torch.minimum(self.mask, zeros) * state_vec)
        
            val_index = torch.prod(masked_state, -1)
            val = torch.sum(val_index * self.v, -1, keepdim=True)
            
            vals.append(val)

        return torch.cat(vals)
    
    def compute_regularization_loss(self) -> Tensor:
        return torch.tensor(0.0).to(self.v)
    
    def postprocess_parameters(self) -> None:
        return

    def params_str(self) -> str:
        return str(self.v)


class ActorCriticOptimizer:

    actor_optimizer: Optimizer
    critic_optimizer: Optimizer

    def __init__(self, actor_optimizer: Optimizer, critic_optimizer: Optimizer):
        self.actor_optimizer = actor_optimizer
        self.critic_optimizer = critic_optimizer

    def zero_grad(self, set_to_none = True) -> None:
        self.actor_optimizer.zero_grad(set_to_none)
        self.critic_optimizer.zero_grad(set_to_none)

    def step(self) -> None:
        self.actor_optimizer.step()
        self.critic_optimizer.step()
