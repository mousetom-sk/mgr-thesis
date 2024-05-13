from __future__ import annotations
from typing import List, Dict, Tuple, Optional

import numpy as np
import numpy.typing as nptype

from lib import Agent, Environment
from lib.lmlp_new import *


class LMLPAgent(Agent):

    _trained: bool = False

    def _setup(self, environment: Environment):
        self._all_features = environment.feature_space
        self._all_actions = environment.action_space

        units = []
        for _ in self._all_actions:
            a_lmlp = LMLP(len(self._all_features),
                          [LMLPLayer(1, And(), UniformInitializer(0.25))])
            units.append(a_lmlp)

        self._lmlp = LMLP(len(self._all_features),
                          [LMLPCompositeLayer(units, Normalization())])
        
        self._optimizer = Adam(0.01)
        self._optimizer.prepare(self._lmlp)

    def _vectorize_state(self, state: Environment.State) -> nptype.NDArray:
        return np.array(list(map(lambda f: state.features[f], self._all_features)))
    
    def _sample_episode(self, environment: Environment) -> Tuple[float, List[Tuple[Environment.State, Environment.Action, nptype.NDArray, float]]]:
        trajectory = []
        steps_left = 50
        total_reward = 0

        state = environment.reset()

        while not environment.is_final() and steps_left > 0:
            state_vec = self._vectorize_state(state)
            intermediate = self._lmlp.forward(state_vec)
            action_dist = intermediate[-1]
            # if steps_left == 48:
            #     print(state_vec)
            #     print(action_dist)
            #     print()
            action_idx = np.random.choice(len(self._all_actions), p=action_dist)
            action_prob = action_dist[action_idx]

            new_state, reward = environment.step(self._all_actions[action_idx])

            trajectory.append((action_idx, action_prob, reward, intermediate))
            total_reward += reward

            state = new_state
            steps_left -= 1
        
        if environment.is_goal():
            return total_reward, True, trajectory
        
        return total_reward, False, trajectory
    
    def _update(self, trajectory: List[Tuple[Environment.State, Environment.Action, nptype.NDArray, float]]):
        gamma = 0.99
        t = len(trajectory) - 1
        G = 0

        for action_idx, action_prob, reward, inter in reversed(trajectory):
            G *= gamma
            G += reward

            grad_obj_out = np.zeros(len(self._all_actions))
            grad_obj_out[action_idx] = (gamma ** t) * G / action_prob
            self._optimizer.optimize(grad_obj_out, inter)

            # if t == 2:
            #     print(self._lmlp.forward(inter[0][0][0])[-1])
            #     print()

            t -= 1

    def train(self, environment: Environment, episodes: int) -> List[float]:
        reward_history = []
        goal_history = []

        self._setup(environment)

        for i in range(episodes):
            print(f"Episode {i}:", end=" ")

            # if i > 200 and i % 10 == 0:
            #     print(i)

            total_reward, is_goal, trajectory = self._sample_episode(environment)
            print(total_reward, is_goal)
            reward_history.append(total_reward)
            goal_history.append(is_goal)
            self._update(trajectory)

        self._trained = True

        print(self._lmlp, file=open("test_lmlp_val.txt", "w"))

        return reward_history, goal_history

    def evaluate(self, environment: Environment, episodes: int) -> List[float]:
        if not self._trained:
            raise RuntimeError("Agent is not trained")
        
        reward_history = []

        for _ in range(episodes):
            total_reward = self._sample_episode(environment)[0]
            reward_history.append(total_reward)

        return reward_history
