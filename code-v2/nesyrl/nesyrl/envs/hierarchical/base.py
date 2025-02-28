from typing import List, Any

from abc import ABC, abstractmethod

import gymnasium as gym

from envs.physical import PhysicalObservation, PhysicalAction, PhysicalEnvironment
from envs.symbolic import StateAtom, Valuation, ActionAtom


class PhysicalEnvironmentWrapper(ABC, gym.Wrapper[PhysicalObservation, PhysicalAction, PhysicalObservation, PhysicalAction]):
    
    __constants__ = ["env"]

    env: PhysicalEnvironment

    def __init__(self, env: PhysicalEnvironment) -> None:
        super().__init__(env)


class SymbolicOptionsEnvironment(PhysicalEnvironmentWrapper):

    __constants__ = ["state_atoms", "option_atoms"]
    
    state_atoms: List[StateAtom]
    option_atoms: List[ActionAtom]

    @abstractmethod
    def get_valuation(self) -> Valuation:
        pass

    @abstractmethod
    def activate_option(self, option: int) -> None:
        pass

    def create_snapshot(self) -> Any:
        return self.env.create_snapshot()
    
    def restore_snapshot(self, snapshot: Any) -> None:
        return self.env.restore_snapshot(snapshot)
