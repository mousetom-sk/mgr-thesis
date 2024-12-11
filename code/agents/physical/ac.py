from __future__ import annotations
from typing import List, Tuple

import torch
from torch import Tensor
from gymnasium import spaces

from ..base import Agent
from envs.physical import PhysicalEnvironment, PhysicalObservation


torch.set_default_dtype(torch.float64)


class ActorCritic(Agent):

    _trained: bool = False
    _one_of: torch.nn.Module = None
    _policies: List[torch.nn.Module] = None

    def _setup(self, environment: PhysicalEnvironment):
        self._device = "cuda" if torch.cuda.is_available() else "cpu"
        
        if isinstance(environment.observation_space, spaces.Dict):
            num_inputs = sum(s.shape[0] for s in environment.observation_space.spaces.values())

        if isinstance(environment.action_space, spaces.OneOf):
            self._one_of = torch.nn.Sequential(
                torch.nn.Linear(num_inputs, 10),
                torch.nn.ReLU(),
                torch.nn.Linear(10, 10),
                torch.nn.ReLU(),
                torch.nn.Linear(10, 10),
                torch.nn.ReLU(),
                torch.nn.Linear(10, 2),
                torch.nn.Softmax(dim=-1)
            ).to(self._device)

            self._policies = [
                torch.nn.Sequential(
                    torch.nn.Linear(num_inputs, 10),
                    torch.nn.ReLU(),
                    torch.nn.Linear(10, 10),
                    torch.nn.ReLU(),
                    torch.nn.Linear(10, 10),
                    torch.nn.ReLU(),
                    torch.nn.Linear(10, environment.action_space.spaces[0].n),
                    torch.nn.Softmax(dim=-1)
                ).to(self._device),
                torch.nn.Sequential(
                    torch.nn.Linear(num_inputs, 10),
                    torch.nn.ReLU(),
                    torch.nn.Linear(10, 10),
                    torch.nn.ReLU(),
                    torch.nn.Linear(10, 10),
                    torch.nn.ReLU(),
                    torch.nn.Linear(10, environment.action_space.spaces[1].shape[0])
                ).to(self._device)
            ]

        self._one_of_optimizer = torch.optim.Adam(self._one_of.parameters())
        self._optimizers = [torch.optim.Adam(p.parameters()) for p in self._policies]
        
        self._v = torch.nn.Sequential(
            torch.nn.Linear(num_inputs, 1)
        ).to(self._device)

        self._v_optimizer = torch.optim.Adam(self._v.parameters())

    def _vectorize_observation(self, observation: PhysicalObservation) -> Tensor:
        return torch.concatenate(tuple(torch.tensor(val) for val in observation.values()))
    
    def _sample_episode(self, environment: PhysicalEnvironment) -> Tuple[float, bool]:
        total_reward = 0

        obs = environment.reset()[0]
        done = False

        while not done:
            obs_vec = self._vectorize_observation(obs).to(self._device)
            action_probs = self._one_of.forward(obs_vec)
            action_dist = torch.distributions.Multinomial(1, action_probs)
            action = action_dist.sample()

            if action == 0:
                subaction_probs = self._policies[action].forward(obs_vec)
                subaction_dist = torch.distributions.Multinomial(1, subaction_probs)
                subaction = subaction_dist.sample()
            else:
                subaction_dist_params = self._policies[action].forward(obs_vec)
                subaction_dist = torch.distributions.Normal(subaction_dist_params)
                subaction = subaction_dist.sample()

            next_obs, reward, terminated, truncated, info = environment.step([action, subaction])

            total_reward += reward
            obs = next_obs
            done = terminated or truncated

        print()
        
        return total_reward, info["is_goal"]
    
    def _train_episode(self, environment: PhysicalEnvironment) -> Tuple[float, bool]:
        gamma = 0.99
        total_reward = 0
        I = 1

        obs = environment.reset()[0]
        done = False

        while not done:
            obs_vec = self._vectorize_observation(obs).to(self._device)
            action_probs = self._one_of.forward(obs_vec)
            action_dist = torch.distributions.Multinomial(1, action_probs)
            action = action_dist.sample()
            action_log_prob = action_dist.log_prob(action)
            action = (action * torch.arange(len(action)).to(action)).sum(dtype=torch.int64)

            if action == 0:
                subaction_probs = self._policies[action].forward(obs_vec)
                subaction_dist = torch.distributions.Multinomial(1, subaction_probs)
                subaction = subaction_dist.sample()
                subaction_log_prob = subaction_dist.log_prob(subaction)
                subaction = (subaction * torch.arange(len(subaction)).to(subaction)).sum(dtype=torch.int64)
            else:
                subaction_dist_params = self._policies[action].forward(obs_vec)
                subaction_dist = torch.distributions.Normal(subaction_dist_params, 0.01)
                subaction = subaction_dist.sample()
                subaction_log_prob = subaction_dist.log_prob(subaction)

            next_obs, reward, terminated, truncated, info = environment.step([action, subaction.cpu().numpy()])

            if not terminated:
                next_obs_vec = self._vectorize_observation(next_obs).to(self._device)
                next_v = self._v.forward(next_obs_vec)
            else:
                next_v = 0

            delta = reward + gamma * next_v - self._v.forward(obs_vec)
            
            self._v_optimizer.zero_grad()
            loss = delta
            loss.backward(retain_graph=True)
            self._v_optimizer.step()

            self._optimizers[action].zero_grad()
            loss = - I * delta * (action_log_prob + subaction_log_prob).sum(dim=-1)
            loss.backward(retain_graph=True)
            self._optimizers[action].step()

            self._one_of_optimizer.zero_grad()
            loss = - I * delta * action_log_prob
            loss.backward()
            self._one_of_optimizer.step()
            
            I *= gamma
            total_reward += reward
            obs = next_obs
            done = terminated or truncated

        print()
            
        return total_reward, info["is_goal"]

    def train(self, environment: PhysicalEnvironment, episodes: int) -> List[float]:
        reward_history = []
        goal_history = []

        self._setup(environment)
        
        self._one_of.train()
        self._v.train()
        for p in self._policies:
            p.train()

        for i in range(episodes):
            print(f"Episode {i}:", end=" ")

            total_reward, is_goal = self._train_episode(environment)

            reward_history.append(total_reward)
            goal_history.append(is_goal)
            print(f"{total_reward:+.3f} {is_goal}")

        self._trained = True

        print(self._lmlp, file=open("_my/logs/physical.txt", "w"))

        return reward_history, goal_history

    def evaluate(self, environment: PhysicalEnvironment, episodes: int) -> List[float]:
        if not self._trained:
            raise RuntimeError("Agent is not trained")
        
        reward_history = []
        goal_history = []

        self._one_of.eval()
        self._v.eval()
        for p in self._policies:
            p.eval()

        for i in range(episodes):
            print(f"Episode {i}:", end=" ")
            
            total_reward, is_goal = self._sample_episode(environment)
            reward_history.append(total_reward)
            goal_history.append(is_goal)

        return reward_history, goal_history
