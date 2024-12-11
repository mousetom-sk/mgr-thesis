from __future__ import annotations
from typing import List, Dict, Tuple, Optional

import numpy.typing as nptype

import torch
from torch import Tensor

from ..base import Agent
from envs.symbolic import SymbolicEnvironment
from lmlp import *


torch.set_default_dtype(torch.float64)


class LMLPReinforceBaseline(Agent):

    _trained: bool = False

    def _setup(self, environment: SymbolicEnvironment):
        self._all_features = environment.feature_space
        self._all_actions = environment.action_space
        
        self._a_lmlps = []
        
        for _ in self._all_actions:
            a_lmlp = LMLPSequential(LMLPLayer(len(self._all_features), 1, And(), ConstantInitializer(0)))
            self._a_lmlps.append(a_lmlp)
        
        self._device = "cuda" if torch.cuda.is_available() else "cpu"
        
        self._lmlp = LMLPSequential(LMLPParallel(len(self._a_lmlps), Normalization(), EyeInitializer(), False, *self._a_lmlps))
        self._lmlp = self._lmlp.to(self._device)

        self._optimizer = torch.optim.Adam(self._lmlp.parameters(), lr=1e-2, maximize=True)
        
        self._v = dict()

    def _vectorize_state(self, state: SymbolicEnvironment.State) -> Tensor:
        return torch.tensor(list(map(lambda f: state.features[f], self._all_features)))
    
    def _sample_episode(self, environment: SymbolicEnvironment, e_cnt=0) -> Tuple[float, List[Tuple[SymbolicEnvironment.State, SymbolicEnvironment.Action, nptype.NDArray, float]]]:
        trajectory = []
        steps_left = 50
        total_reward = 0

        state = environment.reset()

        while not environment.is_final() and steps_left > 0:
            state_vec = self._vectorize_state(state).to(self._device)
            action_dist = self._lmlp.forward(state_vec)
            # if e_cnt == 999 and steps_left == 49:
            #     print(state_vec)
            #     print(action_dist)
            #     print()

            action_idx = torch.multinomial(action_dist, 1)
            action_prob = action_dist[action_idx]

            new_state, reward = environment.step(self._all_actions[action_idx])

            trajectory.append((state, action_prob, reward))
            total_reward += reward

            state = new_state
            steps_left -= 1

        print()
        
        if environment.is_goal():
            return total_reward, True, trajectory
        
        return total_reward, False, trajectory
    
    def _update(self, trajectory: List[Tuple[SymbolicEnvironment.State, SymbolicEnvironment.Action, nptype.NDArray, float]]):
        gamma = 0.99
        t = len(trajectory) - 1
        G = 0

        for state, action_prob, reward in reversed(trajectory):
            G *= gamma
            G += reward

            if state not in self._v:
                self._v[state] = 0
            
            objective = (gamma ** t) * (G - self._v[state]) * torch.log(action_prob)

            self._optimizer.zero_grad()
            objective.backward()
            self._optimizer.step()

            self._v[state] += 0.05 * (G - self._v[state])

            t -= 1

    def train(self, environment: SymbolicEnvironment, episodes: int) -> List[float]:
        reward_history = []
        goal_history = []

        self._setup(environment)
        self._lmlp.train()

        for i in range(episodes):
            print(f"Episode {i}:", end=" ")

            # if i == 68:
            #     print("a")

            total_reward, is_goal, trajectory = self._sample_episode(environment, i)
            print(total_reward, is_goal)
            reward_history.append(total_reward)
            goal_history.append(is_goal)
            self._update(trajectory)

        self._trained = True

        print(self._lmlp, file=open("torch.txt", "w"))

        return reward_history, goal_history

    def evaluate(self, environment: SymbolicEnvironment, episodes: int) -> List[float]:
        if not self._trained:
            raise RuntimeError("Agent is not trained")
        
        reward_history = []
        goal_history = []

        self._lmlp.eval()

        for i in range(episodes):
            print(f"Episode {i}:", end=" ")
            
            total_reward, is_goal = self._sample_episode(environment)[:2]
            reward_history.append(total_reward)
            goal_history.append(is_goal)

        return reward_history, goal_history
