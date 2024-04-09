from __future__ import annotations
from typing import List, Dict, Tuple, Optional

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable

import numpy as np
import numpy.typing as nptype

from lib import Agent, Environment


class PolicyNetwork(nn.Module):

    def __init__(self, num_features, num_options, num_actions):
        super(PolicyNetwork, self).__init__()

        self.num_actions = num_actions
        units = []
        for _ in range(num_actions):
            units.append(nn.Sequential(
                nn.Linear(num_features, num_options),
                nn.ReLU(),
                nn.Linear(num_options, 1),
                nn.ReLU()
            ))

        self.linears = nn.ModuleList(units)
        self.optimizer = optim.Adam(self.parameters(), lr=0.001)

    def forward(self, state):
        units_output = tuple(m.forward(state) for m in self.linears)
        dist = F.softmax(torch.cat(units_output), dim=-1)
        
        return dist
    
    def get_action(self, state):
        state = torch.from_numpy(state).float()
        probs = self.forward(Variable(state))
        highest_prob_action = np.random.choice(self.num_actions, p=probs.detach().numpy())
        log_prob = torch.log(probs[highest_prob_action])
        return highest_prob_action, log_prob


class TorchAgent(Agent):

    _trained: bool = False

    def _setup(self, environment: Environment):
        self._all_features = environment.feature_space
        self._all_actions = environment.action_space

        self._mlp = PolicyNetwork(len(self._all_features), 5, len(self._all_actions))

    def _vectorize_state(self, state: Environment.State) -> nptype.NDArray:
        return np.array(list(map(lambda f: state.features[f], self._all_features)))
        return np.random.choice(self._all_actions, p=action_dist)
    
    def _sample_episode(self, environment: Environment) -> Tuple[float, List[Tuple[Environment.State, Environment.Action, nptype.NDArray, float]]]:
        trajectory = []
        steps_left = 50
        total_reward = 0

        state = environment.reset()

        while not environment.is_final() and steps_left > 0:
            state_vec = self._vectorize_state(state)
            action, log_prob = self._mlp.get_action(state_vec)
            action = self._all_actions[action]
            new_state, reward = environment.step(action)

            trajectory.append((state, action, log_prob, reward))
            total_reward += reward

            state = new_state
            steps_left -= 1
        
        return total_reward, trajectory
    
    def _update(self, trajectory: List[Tuple[Environment.State, Environment.Action, nptype.NDArray, float]]):
        discounted_rewards = []
        log_probs = []

        gamma = 0.999
        G = 0

        for _, _, log_prob, reward in reversed(trajectory):
            G *= gamma
            G += reward

            discounted_rewards.append(G)
            log_probs.append(log_prob)
            
        discounted_rewards = torch.tensor(discounted_rewards[::-1])
        discounted_rewards = (discounted_rewards - discounted_rewards.mean()) / (discounted_rewards.std() + 1e-9) # normalize discounted rewards
        log_probs = log_probs[::-1]

        policy_gradient = []
        for log_prob, Gt in zip(log_probs, discounted_rewards):
            policy_gradient.append(-log_prob * Gt)
        
        self._mlp.optimizer.zero_grad()
        policy_gradient = torch.stack(policy_gradient).sum()
        policy_gradient.backward()
        self._mlp.optimizer.step()

        # gamma = 0.999
        # G = 0
        # t = len(trajectory) - 1

        # for state, action, action_dist, reward in reversed(trajectory):
        #     G *= gamma
        #     G += reward

        #     state_vec = self._vectorize_state(state)
        #     # action_dist = self._lmlp.evaluate(state_vec) # TODO: already updated lmlp, is it ok?
        #     action_idx = self._all_actions.index(action)
        #     action_prob = action_dist[action_idx]

        #     # self._lmlp.layers[-1].activation.dprocess = lambda x: x[:, :, action_idx]
        #     self._lmlp.backpropagate(alpha * (gamma ** t) * G,
        #                              state_vec,
        #                              np.repeat(1 / action_prob, len(self._all_actions)))
        #     t -= 1

    def train(self, environment: Environment, episodes: int) -> List[float]:
        reward_history = []

        self._setup(environment)

        for _ in range(episodes):
            total_reward, trajectory = self._sample_episode(environment)
            reward_history.append(total_reward)
            self._update(trajectory)

        self._trained = True

        return reward_history

    def evaluate(self, environment: Environment, episodes: int) -> List[float]:
        if not self._trained:
            raise RuntimeError("Agent is not trained")
        
        reward_history = []

        for _ in range(episodes):
            total_reward = self._sample_episode(environment)[0]
            reward_history.append(total_reward)

        return reward_history
