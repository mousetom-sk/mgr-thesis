from typing import List, Dict, Tuple, Any

import numpy as np

from tianshou.data import Batch
from tianshou.env import VectorEnvNormObs
from tianshou.policy import BasePolicy

from nesyrl.logic.propositional import FuzzySemantics, Valuation
from nesyrl.envs.symbolic import BlocksWorld, SymbolicEnvironment, ActionAtom, On, Top


class BlocksWorldHierarchical(BlocksWorld):

    _physical_policy: BasePolicy
    _physical_env: VectorEnvNormObs

    def __init__(
        self, physical_policy: BasePolicy, physical_env: VectorEnvNormObs,
        horizon: int, blocks: List[str], goal_state: List[List[str]], use_top: bool = True,
        semantics: FuzzySemantics = FuzzySemantics(), reward_subgoals: bool = False
    ) -> None:
        super().__init__(horizon, blocks, goal_state, None,
                         use_top, semantics, reward_subgoals)

        self._physical_policy = physical_policy
        self._physical_env = physical_env
        
    def _perform_action(self, action: ActionAtom) -> None:
        obs = self._physical_env.get_env_attr("prepare_move")[0](*action.args)
        batch = Batch(obs=np.atleast_2d(obs), info={})
        done = False

        while not done:
            a = self._physical_policy(batch, None).act
            a = self._physical_policy.map_action(a).cpu().numpy()

            obs, _, ter, trun, info = self._physical_env.step(a)
            batch = Batch(obs=np.atleast_2d(obs), info={})
            done = ter[0] or trun[0]
        
        if not info[0]["is_goal"]:
            self._current_state = self._invalid_state
            return
        
        self._current_step += 1
        block1, block2 = action.args

        for below in [self._table] + self._blocks:
            if below == block1:
                continue

            if self.semantics.is_true(On(block1, below), self._current_state):
                self.semantics.set_false(On(block1, below), self._current_state)

                if self._use_top and below != self._table:
                    self.semantics.set_true(Top(below), self._current_state)

                break

        self.semantics.set_true(On(block1, block2), self._current_state)

        if self._use_top and block2 != self._table:
            self.semantics.set_false(Top(block2), self._current_state)

    def reset(self, *, seed: int | None = None, options: Dict[str, Any] | None = None) -> Tuple[Valuation, Dict[str, Any]]:
        super(SymbolicEnvironment, self).reset(seed=seed, options=options)
        
        self._physical_env.reset(None, seed=seed, options=options)

        self._current_step = 0
        self._current_state = self._parse_raw_state(
            [s for s in self._physical_env.get_env_attr("_current_blocks_state")[0]
             if len(s) > 0]
        )
        self._reached_subgoals = self._get_reached_subgoals()

        observation = self._get_observation()

        return observation, {}
