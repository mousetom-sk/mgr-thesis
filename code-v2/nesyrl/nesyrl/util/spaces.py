from typing import List, Sequence, Any

from numpy.random import Generator

import gymnasium as gym
from gymnasium.spaces import Space


class Float(Space):

    low: float
    high: float

    def __init__(self, low: float, high: float, seed: int | Generator | None = None) -> None:
        super().__init__(None, float, seed)

        self.low = low
        self.high = high

    def sample(self, mask = None):
        if mask is not None:
            raise gym.error.Error(
                f"Float.sample cannot be provided a mask, actual value: {mask}"
            )
        
        return self.np_random.uniform(self.low, self.high)
    
    def contains(self, x: Any) -> bool:
        if not isinstance(x, float):
            return False
        
        return self.low <= x <= self.high
    
    def to_jsonable(self, sample_n: Sequence[float]) -> List[float]:
        return [sample for sample in sample_n]

    def from_jsonable(self, sample_n: Sequence[float]) -> List[float]:
        return [sample for sample in sample_n]
