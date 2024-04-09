from __future__ import annotations
from typing import List, Dict, Tuple, Optional

import numpy as np

from lib import Agent, Environment


class LNNAgent(Agent):

    class Critic:
        pass

    _trained: bool = False

    def train(self, environment: Environment, episodes: int) -> List[float]:
        for _ in range(episodes):
            pass

        self._trained = True

        

    def evaluate(self, environment: Environment, episodes: int) -> List[float]:
        if not self._trained:
            raise RuntimeError("Agent is not trained")
        
