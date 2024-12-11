from __future__ import annotations
from typing import List
from abc import ABC, abstractmethod

from envs.symbolic import SymbolicEnvironment


class Agent(ABC):

    @abstractmethod
    def train(self, environment: SymbolicEnvironment, episodes: int) -> List[float]:
        pass

    @abstractmethod
    def evaluate(self, environment: SymbolicEnvironment, episodes: int) -> List[float]:
        pass
