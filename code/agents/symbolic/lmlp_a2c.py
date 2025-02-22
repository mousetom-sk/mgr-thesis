from __future__ import annotations
from typing import List, Tuple

import numpy.typing as nptype

import torch
from torch import Tensor

import matplotlib.pyplot as plt

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
                          ConstantInitializer(0), NoRegularizer()), #UncertaintyRegularizer3(1e-4)), #UncertaintyRegularizer3(1e-3)), #L1DynamicRegularizer(1e-4)), #UncertaintyRegularizer(1e-3)), #L1AdaptiveRegularizer(0.15, 1e-4) #UncertaintyRegularizer(1e-3) #SimilarityRegularizer(1e-2)), #ZeroingPostprocessor(0.5, 1000)),
                LMLPLayer(1, 1, Xor3Fixed(), ConstantInitializer(0), trainable=False)
            )
            # with torch.no_grad():
            #     a_lmlp.lmlp_modules[0].weight[:, -1] = 2
            self._a_lmlps.append(a_lmlp)
        
        self._device = "cuda" if torch.cuda.is_available() else "cpu"
        
        self._lmlp = LMLPSequential(
            LMLPParallel(len(self._a_lmlps), Identity(), EyeInitializer(), OrRegularizer(0), NoPostprocessor(), False, *self._a_lmlps),
            LMLPLayer(len(self._a_lmlps), len(self._a_lmlps), ScaledSoftmax(10), EyeInitializer(), trainable=False)
        )
        self._lmlp = self._lmlp.to(self._device)

        self._optimizer = torch.optim.RMSprop(self._lmlp.parameters(), lr=1e-2)
        
        self._v = dict()

        self._probs = []

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

            if terminated and not info["is_goal"]:
                self._diag.append([state, action_probs, action])

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

            self._probs.append(torch.max(action_probs))

            next_state, reward, terminated, truncated, info = training_env.step(action)
            print(self._all_actions[action], end=" ")

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

        print()
        print(info["is_goal"])
            
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

        print(self._lmlp, file=open("torch_a2cr.txt", "w"))

        return reward_history, goal_history, inits
    
    def _plot_weights(self):
        scaled_weight = torch.tanh(self._a_lmlps[0].lmlp_modules[0].weight).cpu().detach()

        for f, w in enumerate(scaled_weight):
            if self._bars[f]:
                self._bars[f].remove()
            
            self._bars[f] = self._axs[f].bar([str(atom) for atom in self._all_features], w, color="tab:blue")

        plt.pause(0.05)

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

        # self._save_dir = save_dir

        plt.ion()
        plt.rcParams['figure.figsize'] = [18, 8]
        fig, self._axs = plt.subplots(2, 1)
        plt.get_current_fig_manager().set_window_title("move(b, a)")
        
        self._bars = [None] * 2

        for f in range(2):
            self._axs[f].set_xlabel("State Atoms")
            self._axs[f].set_ylabel("Scaled Weight")
            self._axs[f].set_yticks(torch.arange(-10, 11) / 10)
            self._axs[f].set_ylim(-1, 1)
            self._axs[f].grid()
        
        fig.tight_layout()

        branch_added = False

        i = 1
        while training_steps > 0:
            print(f"Episode {i}")

            if (i - 1) % 20 == 0:
                self._plot_weights()

            training_steps, before_eval, avg_returns, avg_goals = self._train_eval_episode(
                training_env, training_steps, eval_env, eval_episodes, before_eval, eval_freq
            )
            
            return_history.extend(avg_returns)
            goal_history.extend(avg_goals)
            i += 1

            # if training_steps < 10000 and not branch_added:
            #     for a_lmlp in self._a_lmlps:
            #         b = torch.zeros_like(a_lmlp.lmlp_modules[0].weight)
            #         b[:, -1] = 2
            #         a_lmlp.lmlp_modules[0].weight = torch.nn.Parameter(torch.cat((a_lmlp.lmlp_modules[0].weight, b)))

            #     self._optimizer = torch.optim.RMSprop(self._lmlp.parameters(), lr=1e-2)
            #     branch_added = True

        if before_eval == 0:
            returns, goals = self.evaluate(eval_env, eval_episodes)
            return_history.append(sum(returns) / eval_episodes)
            goal_history.append(sum(goals) / eval_episodes)

        self._trained = True

        print(self._lmlp, file=open(f"{save_dir}/weights.txt", "w"))

        # for i in range(len(self._a_lmlps)):
        #     print(self._a_lmlps[i].lmlp_modules[0].activation._cnt)

        print(", ".join(f"{p:.6f}" for p in self._probs), file=open(f"{save_dir}/probs.txt", "w"))

        print(", ".join(f"{p:.6f}" for p in self._lmlp.lmlp_modules[0].weight_regularizer._or_history), file=open(f"{save_dir}/or.txt", "w"))
        print(", ".join(f"{p:.6f}" for p in self._lmlp.lmlp_modules[0].weight_regularizer._xor_history), file=open(f"{save_dir}/xor.txt", "w"))
        
        return return_history, goal_history

    def evaluate(self, environment: SymbolicEnvironment, episodes: int, save_dir: str):
        # if not self._trained:
        #     raise RuntimeError("Agent is not trained")
        
        return_history = []
        goal_history = []

        self._lmlp.eval()

        self._diag = []

        for i in range(episodes):
            print(f"Episode {i}:", end=" ")
            
            total_reward, is_goal = self._sample_episode(environment)[:2]
            return_history.append(total_reward)
            goal_history.append(is_goal)

        with open(f"{save_dir}/issues.txt", "w") as out:
            for desc in self._diag:
                print(" ".join(str(atom) for atom, val in desc[0].items() if val > 0.7), file=out)
                print(desc[1], file=out)
                print(desc[2], file=out)
                print(file=out)

        return return_history, goal_history
