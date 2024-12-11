from __future__ import annotations
from typing import Tuple, List, Dict, Any

import numpy as np
import gymnasium as gym
from gymnasium import spaces
from gymnasium.utils import EzPickle
from gymnasium.envs.mujoco import MujocoEnv


class NicoBlocksWorld(MujocoEnv, EzPickle):
    
    metadata = {"render_modes": ["human", "rgb_array", "depth_array"]}
    robot_xml = "./envs/physical/models/robots/nico-new/nico_upper_rh6d.urdf"

    def __init__(self, frame_skip=1, **kwargs):
        EzPickle.__init__(self, self.robot_xml, frame_skip, **kwargs)

        MujocoEnv.__init__(
            self,
            self.robot_xml,
            frame_skip=frame_skip,
            observation_space=None,  # needs to be defined after
            default_camera_config={},
            **kwargs,
        )

        print(self.data)
        exit()

        # obs_size = self.data.qpos.size + self.data.qvel.size

        # self.observation_space = Box(
        #     low=-np.inf, high=np.inf, shape=(obs_size,), dtype=np.float64
        # )

    def step(self, action) -> Tuple[Dict[str, float], float, bool, Dict[str, Any]]:
        x_position_before = self.data.qpos[0]
        self.do_simulation(action, self.frame_skip)
        x_position_after = self.data.qpos[0]

        observation = self._get_obs()
        reward = x_position_after - x_position_before
        info = {}

        if self.render_mode == "human":
            self.render()
        return observation, reward, False, False, info
    
    def render(self):
        pass

    def reset(self, *, seed: int | None = None, options: Dict[str, Any] | None = None) -> Tuple[Dict[str, float], Dict[str, Any]]:
        pass

    def close(self):
        pass

    def _get_obs(self):
        position = self.data.qpos.flat.copy()
        velocity = self.data.qvel.flat.copy()
        return np.concatenate((position, velocity))

    def reset_model(self):
        qpos = self.init_qpos
        qvel = self.init_qvel
        self.set_state(qpos, qvel)

        observation = self._get_obs()

        return observation

    def _get_reset_info(self):
        return {"works": True}


# class BlocksWorld(gym.Env):

#     __constants__ = ["_TABLE", "_goal_state", "horizon", "current_step"]

#     _blocks: List[str]
#     _goal_state: State
#     _initial_state: State
#     _all_subgoals: List[State]

#     horizon: int
#     current_step: int

#     def __init__(
#         self, horizon: int, blocks: List[str],
#         goal_state: List[List[str]], initial_state: List[List[str]] | None = None,
#     ) -> None:
#         super().__init__()

#         if self._TABLE in blocks:
#             raise ValueError(f"No block can be named {self._TABLE}")

#         self._blocks = blocks[:]
#         self._generate_state_atoms()
#         self._generate_action_atoms()
#         self._goal_state = self._parse_raw_state(goal_state)
#         self._initial_state = (self._parse_raw_state(initial_state) if initial_state is not None
#                                else None)
#         self._generate_subgoals(goal_state)

#         self.horizon = horizon
#         self.current_step = 0
#         self.observation_space = spaces.Dict(
#             {f: spaces.Box(0, 1, dtype=float) for f in self.state_atoms}
#         )
#         self.action_space = spaces.Discrete(len(self.action_atoms))

#     def _generate_state_atoms(self) -> None:
#         self.state_atoms = []

#         for b1 in self._blocks + [self._TABLE]:
#             if use_top and b1 != self._TABLE:
#                 self.state_atoms.append(Top(b1))

#             for b2 in self._blocks:
#                 if b1 != b2:
#                     self.state_atoms.append(On(b2, b1))

#         self.state_atoms.append(On(self._blocks[0], self._blocks[0]))
    
#     def _generate_action_atoms(self) -> None:
#         self.action_atoms = []

#         for b1 in self._blocks + [self._TABLE]:
#             for b2 in self._blocks:
#                 if b1 != b2:
#                     self.action_atoms.append(Move(b2, b1))
    
#     def _generate_subgoals(self, raw_goal_state: List[List[str]]) -> None:
#         blocks_backup = list(self._blocks)
#         self._all_subgoals = []

#         for stack in raw_goal_state:
#             for i in range(1, len(stack)):
#                 substack = stack[:i]
#                 self._blocks = substack
#                 self._all_subgoals.append(self._parse_raw_state([substack]))

#                 if use_top: self._all_subgoals[-1][Top(substack[-1])] = 0

#         self._blocks = blocks_backup 

#     def _parse_raw_state(self, raw_state: List[List[str]]) -> State:
#         remaining_blocks = set(self._blocks)
#         true_state_atoms = set()

#         for stack in raw_state:
#             for b1, b2 in zip([self._TABLE] + stack, stack + [None]):
#                 if b2 is None and b1 != self._TABLE:
#                     if use_top: true_state_atoms.add(Top(b1))
#                     continue

#                 if b2 not in remaining_blocks:
#                     if b2 not in self._blocks:
#                         raise ValueError(f"Unknown block {b2}")
                    
#                     raise ValueError(f"Multiple occurences of the block {b2} in a single state.")
                
#                 true_state_atoms.add(On(b2, b1))
#                 remaining_blocks.discard(b2)

#         if len(remaining_blocks) > 0:
#             raise ValueError(f"These blocks are not positioned: {b2}")
        
#         return {atom: int(atom in true_state_atoms) for atom in self.state_atoms}

#     def _is_state_subsumed(self, state: State, other: State) -> bool:
#         return all(other[atom] >= value for (atom, value) in state.items())

#     def action_to_idx(self, action: ActionAtom) -> int:
#         return self.action_atoms.index(action)

#     def idx_to_action(self, index: int) -> ActionAtom:
#         return self.action_atoms[index]

#     def step(self, action: int) -> Tuple[State, float, bool, Dict[str, Any]]:
#         action = self.action_atoms[action]
#         print(action, end=" ")
        
#         if not self._is_valid_action(action):
#             return self._current_state, -2.0, False, True, {"is_goal": False}
        
#         if isinstance(action, Move):
#             next_state = dict(self._current_state)
#             block1, block2 = action.args

#             for below in [self._TABLE] + self._blocks:
#                 if below == block1:
#                     continue

#                 if next_state[On(block1, below)]:
#                     next_state[On(block1, below)] = 0

#                     if use_top and below != self._TABLE:
#                         next_state[Top(below)] = 1

#                     break

#             next_state[On(block1, block2)] = 1

#             if use_top and block2 != self._TABLE:
#                 next_state[Top(block2)] = 0

#             self._current_state = next_state

#         terminated = self._is_state_subsumed(self._goal_state, self._current_state) # validity: False
        
#         self.current_step += 1
#         truncated = self.current_step >= self.horizon

#         if terminated:
#             reward = 2.0
#         else:
#             reward = -0.1
#             # reached = set(sg for sg in self._subgoals if sg.is_subset(self._current_state))
#             # subgoal_diff = len(reached - self._subgoals_reached) - 1.1 * len(self._subgoals_reached - reached)

#             # if subgoal_diff != 0:
#             #     reward = subgoal_diff
#             # else:
#             #     reward = -0.1

#             # self._subgoals_reached = reached

#             # generic (option): reward = -0.1

#         # validity: reward = 0.1

#         return self._current_state, reward, terminated, truncated, {"is_goal": terminated}
    
#     def _is_valid_action(self, action: ActionAtom) -> bool:
#         if isinstance(action, Move):
#             block1, block2 = action.args

#             if self._current_state[On(block1, block2)]:
#                 return False
            
#             if ((use_top and not self._current_state[Top(block1)])
#                 or any(self._current_state[On(b, block1)] for b in self._blocks if b != block1)):
#                 return False
            
#             if block2 != self._TABLE and ((use_top and not self._current_state[Top(block2)])
#                                                 or any(self._current_state[On(b, block2)] for b in self._blocks if b != block2)):
#                 return False

#             return True
        
#         raise ValueError(f"Unknown action {action}")

#     def reset(self, *, seed: int | None = None, options: Dict[str, Any] | None = None) -> Tuple[State, Dict[str, Any]]:
#         super().reset(seed=seed)
#         # self._subgoals_reached = set()

#         if self._initial_state is None:
#             while True:
#                 self._current_state = self._generate_random_state()

#                 if not self._is_state_subsumed(self._goal_state, self._current_state):
#                     break
#         else:
#             self._current_state = self._initial_state

#         self.current_step = 0
        
#         # self._subgoals = list(filter(lambda sg: not sg.is_subset(self._current_state), self._all_subgoals))

#         return self._current_state, {}

#     def _generate_random_state(self) -> State:
#         shuffled_blocks = self.np_random.permutation(self._blocks)
#         stack_ends = np.concatenate((self.np_random.integers(2, size=len(self._blocks) - 1), np.array([1])))

#         raw_state = []
#         stack = []

#         for (block, end_stack) in zip(shuffled_blocks, stack_ends):
#             stack.append(block)

#             if end_stack:
#                 raw_state.append(stack)
#                 stack = []

#         return self._parse_raw_state(raw_state)
