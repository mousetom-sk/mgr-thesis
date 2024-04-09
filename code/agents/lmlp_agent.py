from __future__ import annotations
from typing import List, Dict, Tuple, Optional

import numpy as np
import numpy.typing as nptype

from lib import Agent, Environment
from lib.lmlp import *


class LMLPAgent(Agent):

    _trained: bool = False

    def _setup(self, environment: Environment):
        self._all_features = environment.feature_space
        self._all_actions = environment.action_space

        # feature_mask = [0, 1, 4, 6, 8, 11, 12, 16]
        # self._all_features = list(map(lambda x: x[1],
        #                               filter(lambda x: x[0] in feature_mask,
        #                                      zip(range(len(self._all_features)),
        #                                          self._all_features))))

        units = []
        for _ in self._all_actions:
            a_lmlp = LMLP(len(self._all_features))
            a_lmlp.add_layer(LMLPLayer(1, And(), True))
            a_lmlp.add_layer(LMLPLayer(1, Or(), False))
            units.append(a_lmlp)

        self._lmlp = LMLP(len(self._all_features))
        self._lmlp.add_layer(LMLPCompositeLayer(units, Normalization()))

    def _vectorize_state(self, state: Environment.State) -> nptype.NDArray:
        return np.array(list(map(lambda f: state.features[f], self._all_features)))
    
    def _sample_episode(self, environment: Environment) -> Tuple[float, List[Tuple[Environment.State, Environment.Action, nptype.NDArray, float]]]:
        trajectory = []
        steps_left = 50
        total_reward = 0

        state = environment.reset()

        while not environment.is_final() and steps_left > 0:
            state_vec = self._vectorize_state(state)
            action_dist = self._lmlp.evaluate(state_vec)
            if steps_left == 50:
                print(state_vec)
                print(action_dist)
                print()
            action_idx = np.random.choice(len(self._all_actions), p=action_dist)
            action_prob = action_dist[action_idx]

            new_state, reward = environment.step(self._all_actions[action_idx])

            trajectory.append((action_idx, action_prob, reward))
            total_reward += reward

            state = new_state
            steps_left -= 1
        
        return total_reward, trajectory
    
    def _update(self, trajectory: List[Tuple[Environment.State, Environment.Action, nptype.NDArray, float]]):
        alpha = 0.3
        gamma = 0.999
        G = 0
        t = len(trajectory) - 1

        for action_idx, action_prob, reward in reversed(trajectory):
            G *= gamma
            G += reward

            self._lmlp.layers[-1].activation.dprocess = lambda x: x[:, action_idx]
            self._lmlp.backpropagate(alpha, np.repeat(G / action_prob, len(self._all_actions)))
            t -= 1

    def train(self, environment: Environment, episodes: int) -> List[float]:
        reward_history = []

        self._setup(environment)

        for _ in range(episodes):
            total_reward, trajectory = self._sample_episode(environment)
            reward_history.append(total_reward)
            self._update(trajectory)

        self._trained = True

        print(self._lmlp, file=open("test_lmlp.txt", "w"))

        return reward_history

    def evaluate(self, environment: Environment, episodes: int) -> List[float]:
        if not self._trained:
            raise RuntimeError("Agent is not trained")
        
        reward_history = []

        for _ in range(episodes):
            total_reward = self._sample_episode(environment)[0]
            reward_history.append(total_reward)

        return reward_history
