from typing import Dict, Any
from numpy.typing import NDArray

from abc import ABC, abstractmethod

import gymnasium as gym


PhysicalObservation = Dict[str, float | NDArray]
PhysicalAction = NDArray


class PhysicalEnvironment(ABC, gym.Env[PhysicalObservation, PhysicalAction]):

    @abstractmethod
    def create_snapshot(self) -> Any:
        pass
    
    @abstractmethod
    def restore_snapshot(self, snapshot: Any) -> PhysicalObservation:
        pass
