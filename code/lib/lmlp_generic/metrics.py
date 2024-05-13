from abc import ABC, abstractmethod

import numpy as np
import numpy.typing as nptype


class Objective(ABC):

    @abstractmethod
    def evaluate(self, reward: float, action_prob: float) -> float:
        pass

    @abstractmethod
    def grad(self, reward: float, action_prob: float) -> float:
        pass


class EpisodicObjective(Objective):

    def evaluate(self, reward: float, action_prob: float) -> float:
        pass

    @abstractmethod
    def grad(self, reward: float, action_prob: float) -> float:
        pass
