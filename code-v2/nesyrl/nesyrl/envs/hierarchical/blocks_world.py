from typing import Tuple, List, Dict, Any

import numpy as np

from .base import SymbolicOptionsEnvironment
from envs.physical import PhysicalObservation, PhysicalAction, NicoBlocksWorldBase
from envs.physical.util import length
from envs.symbolic import StateAtom, Contradiction, Valuation, ActionAtom
from envs.symbolic.util import FuzzySemantics


class On(StateAtom):

    def __init__(self, block1: str, block2: str) -> None:
        super().__init__("on", block1, block2)


class Top(StateAtom):

    def __init__(self, block: str) -> None:
        super().__init__("top", block)


class ReachForGrasp(ActionAtom):

    def __init__(self, block) -> None:
        super().__init__("reach_grasp", block)


class ReachForRelease(ActionAtom):

    def __init__(self, block) -> None:
        super().__init__("reach_release", block)


class Grasp(ActionAtom):

    def __init__(self) -> None:
        super().__init__("grasp")


class Release(ActionAtom):

    def __init__(self) -> None:
        super().__init__("release")


class OptionsNicoBlocksWorld(SymbolicOptionsEnvironment):

    __constants__ = ["_table", "_use_top", "_semantics"]

    env: NicoBlocksWorldBase

    _table: str
    
    _use_top: bool
    _semantics: FuzzySemantics
    _blocks: List[str]
    _current_option: ActionAtom
    _dist: float
    _dist2: float

    def __init__(
        self, env: NicoBlocksWorldBase,
        use_top: bool = True, semantics: FuzzySemantics = FuzzySemantics()
    ) -> None:
        super().__init__(env)

        self._blocks = list(env._blocks)
        self._use_top = use_top
        self._semantics = semantics
        self._table = env._table
        
        self._init_state_atoms()
        self._init_option_atoms()

    def _init_state_atoms(self) -> None:
        self.state_atoms = []

        for b1 in self._blocks + [self._table]:
            if self._use_top and b1 != self._table:
                self.state_atoms.append(Top(b1))

            for b2 in self._blocks:
                if b1 != b2:
                    self.state_atoms.append(On(b2, b1))

        self.state_atoms.append(Contradiction())
    
    def _init_option_atoms(self) -> None:
        self.option_atoms = []

        # for b in self._blocks:
        #     self.option_atoms.append(ReachForGrasp(b))
        #     self.option_atoms.append(ReachForRelease(b))

        # self.option_atoms.append(ReachForRelease(self._table))
        self.option_atoms.append(ReachForGrasp("b"))
        # self.option_atoms.append(Grasp())
        # self.option_atoms.append(Release())
    
    def _can_grasp(self, block: str) -> bool:
        return self.env._find_closest_graspable_block() == self.env._blocks[block]
    
    def _can_release(self, block: str) -> bool:
        if block == self._table:
            return self.env._find_closest_table_position() is not None
        
        return self.env._find_closest_target_block() == self.env._blocks[block]
    
    def _is_grasped(self) -> bool:
        return self.env._grasped_block is not None
    
    def _is_released(self) -> bool:
        return self.env._grasped_block is None

    def _evaluate_last_transition(self, observation: PhysicalObservation, reward: float, info: Dict[str, Any]) -> Dict[str, Any]:
        if not info["is_valid"]:
            return {"intrinsic_reward": reward, "is_option_goal": False}
        
        if ((isinstance(self._current_option, ReachForGrasp)
             and self._can_grasp(self._current_option.args[0]))
            or (isinstance(self._current_option, ReachForRelease)
                and self._can_release(self._current_option.args[0]))
            or (isinstance(self._current_option, Grasp) and self._is_grasped())
            or (isinstance(self._current_option, Release) and self._is_released())):
            return {"intrinsic_reward": 1.0, "is_option_goal": True}
        
        if (isinstance(self._current_option, ReachForGrasp)
            or (isinstance(self._current_option, ReachForRelease)
                and self._current_option.args[0] != self._table)):
            dist = length(observation[self._current_option.args[0]][:3])
            last_dist = dist if self._dist is None else self._dist
            self._dist = dist

            # dist2 = sorted(
            #     [length(observation[b][:3]) for b in observation
            #      if b != self._current_option.args[0] and b != self.env.eeff_obs_key and not b.startswith("joint")]
            # )[0]
            # last_dist2 = dist2 if self._dist2 is None else self._dist2
            # self._dist2 = dist
            
            delta = (last_dist - dist)
            # delta2 = (last_dist2 - dist2)
            # delta *= np.exp(-dist)
            # if delta < 0:
            #     delta = min(-1e-3, delta)
            # if delta < 0:
            #     delta *= 2
            
            return {"intrinsic_reward": delta, "is_option_goal": False}
        
        if (isinstance(self._current_option, ReachForRelease)
            and self._current_option.args[0] == self._table):
            dist = observation[self.env.eeff_obs_key][2] - 0.645
            last_dist = dist if self._dist is None else self._dist
            self._dist = dist
            
            delta = last_dist - dist
            delta *= 2 * np.exp(-dist)
            # if -0.01 < delta < 0:
            #     delta = -0.01
            
            return {"intrinsic_reward": delta, "is_option_goal": False}

        return {"intrinsic_reward": reward, "is_option_goal": False}

    def get_valuation(self) -> Valuation:
        valuation = {}
        
        for atom in filter(lambda a: isinstance(a, On), self.state_atoms):
            if self.env._is_on(*atom.args):
                self._semantics.set_true(atom, valuation)
            else:
                self._semantics.set_false(atom, valuation)

        for atom in filter(lambda a: isinstance(a, Top), self.state_atoms):
            if all(self._semantics.is_false(On(b, atom.args[0]), valuation)
                   for b in self._blocks if b != atom.args[0]):
                self._semantics.set_true(atom, valuation)
            else:
                self._semantics.set_false(atom, valuation)

        self._semantics.set_false(Contradiction(), valuation)

        return valuation

    def activate_option(self, option: int) -> None:
        self._current_option = self.option_atoms[option]
        self._dist = None
        self._dist2 = None
    
    def step(self, action: PhysicalAction) -> Tuple[PhysicalObservation, float, bool, bool, Dict[str, Any]]:
        observation, reward, terminated, truncated, info = self.env.step(action)
        info |= self._evaluate_last_transition(observation, reward, info)

        return observation, reward, terminated, truncated, info
