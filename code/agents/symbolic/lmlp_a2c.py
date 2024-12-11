from __future__ import annotations
from typing import List, Tuple

import numpy.typing as nptype

import torch
from torch import Tensor

from ..base import Agent
from envs.symbolic import SymbolicEnvironment
from lmlp import *


torch.set_default_dtype(torch.float64)


class LMLPA2C(Agent):

    _trained: bool = False

    def _setup(self, environment: SymbolicEnvironment):
        self._all_features = environment.state_atoms
        self._all_actions = environment.action_atoms
        
        self._a_lmlps = []
        
        for _ in self._all_actions:
            a_lmlp = LMLPSequential(
                LMLPLayer(len(self._all_features), 1, And3(),
                          ConstantInitializer(0), NoRegularizer())#UncertaintyRegularizer(1e-3)) #L1AdaptiveRegularizer(0.15, 1e-4) #UncertaintyRegularizer(1e-3) #SimilarityRegularizer(1e-2)), #ZeroingPostprocessor(0.5, 1000)),
                # LMLPLayer(2, 1, Or3(), ConstantInitializer(10), trainable=False)
            )
            self._a_lmlps.append(a_lmlp)
        
        self._device = "cuda" if torch.cuda.is_available() else "cpu"
        
        self._lmlp = LMLPSequential(LMLPParallel(len(self._a_lmlps), ScaledSoftmax(10), EyeInitializer(), NoRegularizer(), NoPostprocessor(), False, *self._a_lmlps))
        self._lmlp = self._lmlp.to(self._device)

        self._optimizer = torch.optim.RMSprop(self._lmlp.parameters(), lr=1e-2)
        
        self._v = dict()

    def _vectorize_state(self, state: SymbolicEnvironment.State) -> Tensor:
        return torch.tensor(list(map(lambda f: state[f], self._all_features)))
    
    def _sample_episode(self, environment: SymbolicEnvironment) -> Tuple[float, List[Tuple[SymbolicEnvironment.State, SymbolicEnvironment.Action, nptype.NDArray, float]]]:
        total_reward = 0
        gamma = 0.99
        I = 1

        state = environment.reset()[0]
        init = environment.get_raw_observation()
        done = False

        while not done:
            state_vec = self._vectorize_state(state).to(self._device)
            action_probs = self._lmlp.forward(state_vec)
            action_dist = torch.distributions.Multinomial(1, action_probs)
            action = torch.argmax(action_dist.sample())

            print(self._all_actions[action], end=" ")
            next_state, reward, terminated, truncated, info = environment.step(action)

            total_reward += I * reward
            I *= gamma
            state = next_state
            done = terminated or truncated

        print()
        
        return total_reward, info["is_goal"], init
    
    def _train_episode(self, environment: SymbolicEnvironment, steps: int):
        gamma = 0.99
        total_reward = 0
        I = 1

        state = environment.reset()[0]
        init = environment.get_raw_observation()
        done = False

        while not done:
            state_vec = self._vectorize_state(state).to(self._device)
            action_probs = self._lmlp.forward(state_vec)
            action_dist = torch.distributions.Multinomial(1, action_probs)
            action = action_dist.sample()
            action_log_prob = action_dist.log_prob(action)
            action = torch.argmax(action)

            print(self._all_actions[action], end=" ")
            next_state, reward, terminated, truncated, info = environment.step(action)

            state_tuple = tuple(state.items())
            next_state_tuple = tuple(next_state.items())

            if state_tuple not in self._v:
                self._v[state_tuple] = 0
            if next_state_tuple not in self._v:
                self._v[next_state_tuple] = 0

            advantage = reward + gamma * self._v[next_state_tuple] - self._v[state_tuple]
            self._v[state_tuple] += (5e-2) * advantage

            self._optimizer.zero_grad()
            task_loss = - I * advantage * action_log_prob
            regularization_loss = self._lmlp.compute_regularization_loss().to(self._device)
            loss = task_loss + regularization_loss
            loss.backward()
            self._optimizer.step()
            self._lmlp.postprocess_parameters()

            total_reward += I * reward
            I *= gamma
            state = next_state
            steps -= 1
            done = terminated or truncated or (steps <= 0)

        print()
            
        return total_reward, info["is_goal"], init, steps

    def _train_eval_episode(
        self, training_env: SymbolicEnvironment, training_steps: int,
        eval_env: SymbolicEnvironment, eval_episodes: int, before_eval: int, eval_freq: int
    ):
        avg_returns = []
        avg_goals = []
        gamma = 0.99
        I = 1

        state = training_env.reset()[0]
        done = False

        while not done:
            if before_eval == 0:
                returns, goals = self.evaluate(eval_env, eval_episodes)
                avg_returns.append(sum(returns) / eval_episodes)
                avg_goals.append(sum(goals) / eval_episodes)
                before_eval = eval_freq

            state_vec = self._vectorize_state(state).to(self._device)
            action_probs = self._lmlp.forward(state_vec)
            action_dist = torch.distributions.Multinomial(1, action_probs)
            action = action_dist.sample()
            action_log_prob = action_dist.log_prob(action)
            action = torch.argmax(action)

            next_state, reward, terminated, truncated, _ = training_env.step(action)

            state_tuple = tuple(state.items())
            next_state_tuple = tuple(next_state.items())

            if state_tuple not in self._v:
                self._v[state_tuple] = 0
            if next_state_tuple not in self._v:
                self._v[next_state_tuple] = 0

            advantage = reward + gamma * self._v[next_state_tuple] - self._v[state_tuple]
            self._v[state_tuple] += (5e-2) * advantage

            self._optimizer.zero_grad()
            task_loss = - I * advantage * action_log_prob
            regularization_loss = self._lmlp.compute_regularization_loss().to(self._device)
            loss = task_loss + regularization_loss
            loss.backward()
            self._optimizer.step()
            self._lmlp.postprocess_parameters()

            I *= gamma
            state = next_state
            training_steps -= 1
            before_eval -= 1
            done = terminated or truncated or (training_steps <= 0)
            
        return training_steps, before_eval, avg_returns, avg_goals

    def train(self, environment: SymbolicEnvironment, steps: int) -> List[float]:
        reward_history = []
        goal_history = []
        inits = []

        self._setup(environment)
        self._lmlp.train()

        i = 1
        while steps > 0:
            print(f"Episode {i}:", end=" ")

            total_reward, is_goal, init, steps = self._train_episode(environment, steps)

            i += 1
            reward_history.append(total_reward)
            goal_history.append(is_goal)
            inits.append(init)
            print(f"{total_reward:+.3f} {is_goal}")

        self._trained = True

        print(self._lmlp, file=open("_my/logs/torch_a2cu.txt", "w"))

        return reward_history, goal_history, inits
    
    def train_eval(
        self, training_env: SymbolicEnvironment, training_steps: int,
        eval_env: SymbolicEnvironment, eval_episodes: int, eval_freq: int,
        save_dir: str
    ):
        return_history = []
        goal_history = []
        before_eval = eval_freq

        self._setup(training_env)
        self._lmlp.train()

        i = 1
        while training_steps > 0:
            print(f"Episode {i}")

            training_steps, before_eval, avg_returns, avg_goals = self._train_eval_episode(
                training_env, training_steps, eval_env, eval_episodes, before_eval, eval_freq
            )
            
            return_history.extend(avg_returns)
            goal_history.extend(avg_goals)
            i += 1

        if before_eval == 0:
            returns, goals = self.evaluate(eval_env, eval_episodes)
            return_history.append(sum(returns) / eval_episodes)
            goal_history.append(sum(goals) / eval_episodes)

        self._trained = True

        print(self._lmlp, file=open(f"{save_dir}/weights.txt", "w"))
        
        return return_history, goal_history

    def evaluate(self, environment: SymbolicEnvironment, episodes: int):
        # if not self._trained:
        #     raise RuntimeError("Agent is not trained")
        
        return_history = []
        goal_history = []

        self._lmlp.eval()

        for i in range(episodes):
            print(f"Episode {i}:", end=" ")
            
            total_reward, is_goal = self._sample_episode(environment)[:2]
            return_history.append(total_reward)
            goal_history.append(is_goal)

        return return_history, goal_history
