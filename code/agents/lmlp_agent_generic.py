from __future__ import annotations
from typing import List, Dict, Tuple, Optional

import numpy as np
import numpy.typing as nptype

from lib import Agent, Environment
from lib.lmlp_generic import *
from envs import BlocksWorld


class LMLPAgent(Agent):

    _trained: bool = False

    def _setup(self, environment: Environment):
        self._all_features = environment.feature_space
        self._all_actions = environment.action_space

        self._lmlp = LMLP(
            None,
            [LMLPCompositeLayer(1, [
                LMLP(4, [LMLPLayer(1, And(), UniformInitializer(0.25))]),
                LMLP(8, [LMLPLayer(1, And(), UniformInitializer(0.25))])
            ], And(), OneSigmoidInitializer())]
        )
        
        self._optimizer = Adam(0.05)
        self._optimizer.prepare(self._lmlp)

    def _vectorize_state(self, state: Environment.State, x: str, y: str) -> nptype.NDArray:
        features = [BlocksWorld.On(x, y), BlocksWorld.On(y, x), BlocksWorld.Top(x), BlocksWorld.Top(y)]

        return np.array(list(map(lambda f: state.features[f] if f in state.features else 0, features)))
    
    def _sample_episode(self, environment: Environment) -> Tuple[float, List[Tuple[Environment.State, Environment.Action, nptype.NDArray, float]]]:
        trajectory = []
        steps_left = 50
        total_reward = 0

        state = environment.reset()
        norm = Normalization()

        while not environment.is_final() and steps_left > 0:
            inter = []
            out = []

            for a in self._all_actions:
                curr_state_vec = self._vectorize_state(state, a.block1, a.block2)
                goal_state_vec = self._vectorize_state(environment._goal_state, a.block1, a.block2)
                
                inter.append(self._lmlp.forward([curr_state_vec, np.hstack((curr_state_vec, goal_state_vec))]))
                out.append(inter[-1][-1])

            action_dist = norm.evaluate(np.hstack(out)).flatten()
            action_idx = np.random.choice(len(self._all_actions), p=action_dist)
            action_prob = action_dist[action_idx]

            # if steps_left == 50:
            #     print(" ".join(f"{p:.6f}" for p in action_dist))

            new_state, reward = environment.step(self._all_actions[action_idx])

            trajectory.append((action_prob, reward, inter[action_idx]))
            total_reward += reward

            state = new_state
            steps_left -= 1

        print()
        
        if environment.is_goal():
            return total_reward, True, trajectory
        
        return total_reward, False, trajectory
    
    def _update(self, trajectory: List[Tuple[Environment.State, Environment.Action, nptype.NDArray, float]]):
        gamma = 0.99
        t = len(trajectory) - 1
        G = 0

        for action_prob, reward, inter in reversed(trajectory):
            G *= gamma
            G += reward

            grad_obj_out = np.array([(gamma ** t) * G / action_prob])
            self._optimizer.optimize(self._ep, grad_obj_out, inter)

            t -= 1

    def train(self, environment: Environment, episodes: int) -> List[float]:
        reward_history = []
        goal_history = []

        self._setup(environment)

        for i in range(episodes):
            print(f"Episode {i}:", end=" ")
            self._ep = i + 1

            total_reward, is_goal, trajectory = self._sample_episode(environment)
            print(total_reward, is_goal)
            reward_history.append(total_reward)
            goal_history.append(is_goal)
            self._update(trajectory)

        self._trained = True

        print(self._lmlp, file=open("test_lmlp_generic2.txt", "w"))

        return reward_history, goal_history

    def evaluate(self, environment: Environment, episodes: int) -> List[float]:
        if not self._trained:
            raise RuntimeError("Agent is not trained")
        
        reward_history = []

        for _ in range(episodes):
            total_reward = self._sample_episode(environment)[0]
            reward_history.append(total_reward)

        return reward_history
