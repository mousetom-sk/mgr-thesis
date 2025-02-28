from typing import Tuple, Callable, Any

from abc import ABC, abstractmethod

import torch
from torch import Tensor
from torch.nn import Module, Parameter, Sequential, Linear
from torch.optim import Optimizer

from tianshou.data import Batch

from nesyrl.envs.symbolic import SymbolicEnvironment
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


# TODO: num_ands, desired init (inject knowledge)
class Actor(NLActor):

    nln: NLNetwork
    net: Module
    valuation_to_structure: Callable[[Valuation | Batch], Structure]

    def __init__(
        self, env: SymbolicEnvironment,
        and_initializer: WeightInitializer, assume_false: bool,
        device: str | int | torch.device
    ) -> None:
        super().__init__()

        nl_atoms = [NLPredicateAtom(atom.__class__, *atom.args)
                    for atom in env.state_atoms
                    if atom not in env.domain_atoms]    

        nl_actions = []
        for _ in env.action_atoms:
            nl_act = NLAndBiMixed(operands=nl_atoms, weight_initializer=and_initializer)
            nl_actions.append(nl_act)
        
        self.nln = NLNetwork(NLStack(nl_actions))
        self.nln.reset_parameters()

        if assume_false:
            false_idx = nl_atoms.index(Contradiction())
            weight = torch.tensor([2.0 if i == false_idx else float("nan")
                                   for i in range(len(nl_atoms))])
            
            for nl_act in nl_actions:
                nl_act.set_parameters(weight)
        
        self.net = Sequential(self.nln, ScaledSoftmax(10))
        self.net = self.net.to(device)

        self.add_module("net", self.net)

        self.valuation_to_structure = (
            lambda obs: Structure.from_valuation(
                dict(obs), env.domain_atoms, env.semantics
            )
        )

    def forward(self, obs: Iterable[Valuation] | Batch, state: Any = None, info: Dict[str, Any] = {}) -> Tuple[Tensor, Any]:
        obs = list(map(self.valuation_to_structure, obs))
        probs = self.net.forward(obs)

        return probs, state
    
    def compute_regularization_loss(self) -> Tensor:
        return self.nln.compute_regularization_loss()
    
    def postprocess_parameters(self) -> None:
        return self.nln.postprocess_parameters()

    def params_str(self) -> str:
        return self.nln.params_str()


class Critic(NLCritic):

    nln: NLNetwork
    combiner: Module
    net: Module
    valuation_to_structure: Callable[[Valuation | Batch], Structure]

    def __init__(
        self, env: SymbolicEnvironment,
        num_rules: int, and_initializer: WeightInitializer, assume_false: bool,
        device: str | int | torch.device
    ) -> None:
        super().__init__()

        nl_atoms = [NLPredicateAtom(atom.__class__, *atom.args)
                    for atom in env.state_atoms
                    if atom not in env.domain_atoms]    

        nl_ands = [NLAndBiMixed(operands=nl_atoms, weight_initializer=and_initializer)
                   for _ in range(num_rules)]
        
        self.nln = NLNetwork(NLStack(nl_ands))
        self.nln.reset_parameters()

        if assume_false:
            false_idx = nl_atoms.index(Contradiction())
            weight = torch.tensor([2.0 if i == false_idx else float("nan")
                                   for i in range(len(nl_atoms))])
            
            for nl_act in nl_ands:
                nl_act.set_parameters(weight)
        
        self.combiner = Linear(num_rules, 1, False)
        torch.nn.init.uniform_(self.combiner.weight, -0.5, 0.5)

        self.net = Sequential(self.nln, self.combiner)
        self.net = self.net.to(device)

        self.add_module("net", self.net)

        self.valuation_to_structure = (
            lambda obs: Structure.from_valuation(
                dict(obs), env.domain_atoms, env.semantics
            )
        )

    def forward(self, obs: Iterable[Valuation] | Batch, **kwargs) -> Tensor:
        obs = list(map(self.valuation_to_structure, obs))
        vals = self.net.forward(obs)

        return vals
    
    def compute_regularization_loss(self) -> Tensor:
        return self.nln.compute_regularization_loss()
    
    def postprocess_parameters(self) -> None:
        return self.nln.postprocess_parameters()

    def params_str(self) -> str:
        return self.nln.params_str() + "\n\n" + "Combiner" + "\n" + str(self.combiner.weight)


class CriticTab(NLCritic):

    mask: Tensor
    v: Tensor
    valuation_to_structure: Callable[[Valuation | Batch], Structure]

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


class ActorCriticOptimizer(Optimizer):

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
