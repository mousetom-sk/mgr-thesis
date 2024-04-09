from __future__ import annotations
from typing import List, Tuple, Dict, Any
from abc import ABC, abstractmethod


class Environment(ABC):

    class Feature(ABC):
        pass
    
    class State(ABC):
        
        @property
        def features(self) -> Dict[Environment.Feature, Any]:
            pass
        
    class Action(ABC):
        pass


    @property
    def feature_space(self) -> List[Feature]:
        pass

    @property
    def action_space(self) -> List[Action]:
        pass

    @abstractmethod
    def is_final(self) -> bool:
        pass

    @abstractmethod
    def step(self, action: Action) -> Tuple[State, float]:
        pass

    @abstractmethod
    def reset(self) -> State:
        pass
