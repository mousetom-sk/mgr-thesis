import torch
from torch import Tensor
from torch.nn import Module, ModuleList

from tianshou.data import Batch

from nesyrl.envs.symbolic import SymbolicEnvironment, StateAtom
from nesyrl.logic.propositional import *
from nesyrl.logic.neural import *


class TransitionModel(Module):

    nlns: ModuleList

    _domain_atoms: List[StateAtom]
    _semantics: FuzzySemantics

    def __init__(
        self, env: SymbolicEnvironment,
        and_impl: NLModuleParametrized, and_initializer: WeightInitializer,
        device: str | int | torch.device = "cpu"
    ) -> None:
        super().__init__()

        nl_atoms = []
        
        for atom in env.state_atoms:
            if atom not in env.domain_atoms and not isinstance(atom, Contradiction):
                nl_atoms.append(NLPredicateAtom(atom.__class__, *atom.args))

        self.nlns = ModuleList()

        for _ in env.action_atoms:
            nl_next = []

            for atom in env.state_atoms:
                if atom not in env.domain_atoms and not isinstance(atom, Contradiction):
                    nl_natom = and_impl(operands=nl_atoms,
                                        weight_initializer=and_initializer)
                    nl_next.append(nl_natom)

            nln = NLNetwork(NLStack(nl_next)).to(device)
            nln.reset_parameters()

            self.nlns.append(nln)

        self._domain_atoms = env.domain_atoms
        self._semantics = env.semantics

    def forward(self, obs: Valuation, act: int) -> Tensor:
        obs = self._valuation_to_structure(obs)
        truth_vals = self.nlns[act].forward([obs])[0]

        return truth_vals
    
    def _valuation_to_structure(self, obs: Valuation | Batch) -> Structure:
        return Structure.from_valuation(dict(obs), self._domain_atoms, self._semantics)

    def params_str(self) -> str:
        params = []

        for i, nln in enumerate(self.nlns):
            params.append(f"Action {i}\n")
            params.append(nln.params_str())

        return "\n".join(params)
