from __future__ import annotations
from typing import List
from abc import ABC, abstractmethod

from .environment import Environment


class Agent(ABC):

    @abstractmethod
    def train(self, environment: Environment, episodes: int) -> List[float]:
        pass

    @abstractmethod
    def evaluate(self, environment: Environment, episodes: int) -> List[float]:
        pass
