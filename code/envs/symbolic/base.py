from typing import Tuple, List, Dict

from abc import ABC

import gymnasium as gym


class Proposition:
    
    __constants__ = ["name", "args"]

    name: str
    args: Tuple[str]

    def __init__(self, name: str, *args: str) -> None:
        self.name = name
        self.args = args
    
    def __str__(self) -> str:
        args = (f"({', '.join(self.args)})" if len(self.args) > 0
                else "")

        return f"{self.name}{args}"
    
    def __hash__(self) -> int:
        return hash(str(self))
    
    def __eq__(self, other: object) -> bool:
        if not isinstance(other, self.__class__):
            return False

        return str(self) == str(other)


class StateAtom(Proposition):
    
    pass


class Contradiction(StateAtom):

    def __init__(self) -> None:
        super().__init__("FALSE")


Valuation = Dict[StateAtom, float]


class ActionAtom(Proposition):

    pass


class SymbolicEnvironment(ABC, gym.Env[Valuation, int]):

    __constants__ = ["state_atoms", "action_atoms"]
    
    state_atoms: List[StateAtom]
    action_atoms: List[ActionAtom]
