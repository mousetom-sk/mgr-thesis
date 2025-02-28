from typing import List

from abc import ABC

import gymnasium as gym

from nesyrl.logic.propositional import PredicateAtom, Valuation, FuzzySemantics


class StateAtom(PredicateAtom):
    
    pass


class ActionAtom(PredicateAtom):

    pass


class SymbolicEnvironment(ABC, gym.Env[Valuation, int]):

    __constants__ = ["state_atoms", "domain_atoms", "action_atoms",
                     "semantics", "all_states"]
    
    state_atoms: List[StateAtom]
    domain_atoms: List[StateAtom]
    action_atoms: List[ActionAtom]

    semantics: FuzzySemantics
    all_states: List[Valuation]
