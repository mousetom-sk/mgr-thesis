from __future__ import annotations
from typing import List, Tuple, Any

import torch
from torch import Tensor

from ..base import Agent
from envs.hierarchical import SymbolicOptionsEnvironment, Grasp, Release, ReachForGrasp, ReachForRelease, ActionAtom, PhysicalObservation, Valuation
from lmlp import *
from envs.physical.util import squared_length


torch.set_default_dtype(torch.float64)


class PPO(Agent):

    _trained: bool = False

    def _setup(self, environment: SymbolicOptionsEnvironment):
        self._device = "cuda" if torch.cuda.is_available() else "cpu"
        
        self._setup_lmlp(environment)
        self._setup_physical(environment)

    def _setup_lmlp(self, environment: SymbolicOptionsEnvironment):
        self._a_lmlps = []

        for _ in environment.option_atoms:
            a_lmlp = LMLPSequential(
                LMLPLayer(len(environment.state_atoms), 1, And3(),
                          ConstantInitializer(0), NoRegularizer())
            )
            self._a_lmlps.append(a_lmlp)
        
        self._lmlp = LMLPSequential(LMLPParallel(len(self._a_lmlps), ScaledSoftmax(10),
                                                 EyeInitializer(), NoRegularizer(), NoPostprocessor(),
                                                 False, *self._a_lmlps))
        self._lmlp = self._lmlp.to(self._device)

        self._lmlp_optimizer = torch.optim.Adam(self._lmlp.parameters(), lr=1e-2)

        self._lmlp_v = dict()

    def _setup_physical(self, environment: SymbolicOptionsEnvironment):
        self._physical_extractors = {
            "reach_grasp":
                torch.nn.Sequential(
                    torch.nn.Linear(len(environment.env._active_joints) + 7 + 7 + 7, 64),
                    torch.nn.Tanh(),
                    torch.nn.Linear(64, 64),
                    torch.nn.Tanh(),
                    torch.nn.Linear(64, 64),
                    torch.nn.Tanh(),
                    torch.nn.Linear(64, 64),
                    torch.nn.Tanh(),
                ).to(self._device),
            "reach_release":
                torch.nn.Sequential(
                    torch.nn.Linear(7 + 7 + 7, 64),
                    torch.nn.Tanh(),
                    torch.nn.Linear(64, 64),
                    torch.nn.Tanh(),
                    torch.nn.Linear(64, 64),
                    torch.nn.Tanh(),
                    torch.nn.Linear(64, 64),
                    torch.nn.Tanh(),
                ).to(self._device),
            "reach_release_table":
                torch.nn.Sequential(
                    torch.nn.Linear(7 + 7, 64),
                    torch.nn.Tanh(),
                    torch.nn.Linear(64, 64),
                    torch.nn.Tanh(),
                    torch.nn.Linear(64, 64),
                    torch.nn.Tanh(),
                    torch.nn.Linear(64, 64),
                    torch.nn.Tanh(),
                ).to(self._device)
        }
        
        self._physical_policies = {
            "reach_grasp":
                torch.nn.Sequential(
                    self._physical_extractors["reach_grasp"],
                    torch.nn.Linear(64, 64),
                    torch.nn.Tanh(),
                    torch.nn.Linear(64, environment.action_space.spaces[1].shape[0])
                ).to(self._device),
            "reach_release":
                torch.nn.Sequential(
                    self._physical_extractors["reach_release"],
                    torch.nn.Linear(64, 64),
                    torch.nn.Tanh(),
                    torch.nn.Linear(64, environment.action_space.spaces[1].shape[0])
                ).to(self._device),
            "reach_release_table":
                torch.nn.Sequential(
                    self._physical_extractors["reach_release_table"],
                    torch.nn.Linear(64, 64),
                    torch.nn.Tanh(),
                    torch.nn.Linear(64, environment.action_space.spaces[1].shape[0])
                ).to(self._device)
        }

        self._physical_vfs = {
            "reach_grasp":
                torch.nn.Sequential(
                    self._physical_extractors["reach_grasp"],
                    torch.nn.Linear(64, 1)
                ).to(self._device),
            "reach_release":
                torch.nn.Sequential(
                    self._physical_extractors["reach_release"],
                    torch.nn.Linear(64, 1)
                ).to(self._device),
            "reach_release_table":
                torch.nn.Sequential(
                    self._physical_extractors["reach_release_table"],
                    torch.nn.Linear(64, 1)
                ).to(self._device)
        }

        self._physical_extractor_optimizers = {
            key: torch.optim.RMSprop(p.parameters(), 1e-5) for key, p in self._physical_extractors.items()
        }

        self._physical_policy_optimizers = {
            key: torch.optim.RMSprop(p.parameters(), 1e-5) for key, p in self._physical_policies.items()
        }

        self._physical_vf_optimizers = {
            key: torch.optim.RMSprop(p.parameters(), 1e-5) for key, p in self._physical_vfs.items()
        }

    def _vectorize_valuation(self, valuation: Valuation) -> Tensor:
        return torch.tensor(list(valuation.values()))
    
    def _vectorize_observation(self, environment: SymbolicOptionsEnvironment, option_atom: ActionAtom, observation: PhysicalObservation) -> Tensor:
        closest_non_target_block = sorted(
            list(b for b in observation if b != option_atom.args[0] and b != environment.env.eeff_obs_key and not b.startswith("joint")),
            key=lambda b: squared_length(observation[b][:3] - observation[environment.env.eeff_obs_key][:3])
        )[0]
        
        joints = torch.tensor([observation[j] for j in observation if j.startswith("joint")])
        
        if isinstance(option_atom, ReachForGrasp):
            return torch.concatenate((
                joints,
                torch.tensor(observation[environment.env.eeff_obs_key]),
                torch.tensor(observation[option_atom.args[0]]),
                torch.tensor(observation[closest_non_target_block])
            ))
        
        if isinstance(option_atom, ReachForRelease):
            if option_atom.args[0] == environment._table:
                return torch.concatenate((
                    torch.tensor(observation[environment.env.eeff_obs_key]),
                    torch.tensor(observation[closest_non_target_block])
                ))

            return torch.concatenate((
                torch.tensor(observation[environment.env.eeff_obs_key]),
                torch.tensor(observation[option_atom.args[0]]),
                torch.tensor(observation[closest_non_target_block])
            ))
    
    def _train_episode(self, environment: SymbolicOptionsEnvironment, subepisodes: int, substeps: int, batch_size: int, n_iter: int):
        gamma = 0.99
        total_reward = 0
        I = 1

        environment.reset()
        done = False

        while not done:
            val = environment.get_valuation()
            val_vec = self._vectorize_valuation(val).to(self._device)
            option_probs = self._lmlp.forward(val_vec)
            option_dist = torch.distributions.Multinomial(1, option_probs)
            option = option_dist.sample()
            option_log_prob = option_dist.log_prob(option)
            option = (option * torch.arange(len(option)).to(option)).sum(dtype=torch.int64)

            print("   ", environment.option_atoms[option], flush=True)
            environment.activate_option(option)

            option_atom = environment.option_atoms[option]

            if isinstance(option_atom, Grasp):
                _, reward, terminated, truncated, info = environment.step([0, 0])
            elif isinstance(option_atom, Release):
                _, reward, terminated, truncated, info = environment.step([0, 1])
            else:
                if isinstance(option_atom, ReachForGrasp):
                    executor = "reach_grasp"
                elif isinstance(option_atom, ReachForRelease):
                    if option_atom.args[0] == environment._table:
                        executor = "reach_release_table"
                    else:
                        executor = "reach_release"

                snapshot = environment.create_snapshot()
                option_goals = 0

                print("    0", end="\r", flush=True)
                for i in range(subepisodes):
                    reward, terminated, truncated, info = self._train_subepisode(environment, snapshot, option_atom, executor, substeps, batch_size, n_iter)
                    option_goals += info["is_option_goal"]
                    print(f"    {i+1}", end="\r", flush=True)
                print(f"    {option_goals / subepisodes:.3f}", flush=True)
            
            next_val = environment.get_valuation()
            next_val_vec = self._vectorize_valuation(next_val).to(self._device)

            val_tuple = tuple(val_vec)
            next_val_tuple = tuple(next_val_vec)

            if val_tuple not in self._lmlp_v:
                self._lmlp_v[val_tuple] = 0
            if next_val_tuple not in self._lmlp_v:
                self._lmlp_v[next_val_tuple] = 0

            delta = reward + gamma * self._lmlp_v[next_val_tuple] - self._lmlp_v[val_tuple]

            self._lmlp_optimizer.zero_grad()
            task_loss = - I * delta * option_log_prob
            task_loss.backward()
            self._lmlp_optimizer.step()

            self._lmlp_v[val_tuple] += (5e-2) * delta

            I *= gamma
            total_reward += reward
            done = terminated or truncated or info["is_option_goal"]
            
        return total_reward, info["is_goal"]
    
    def _train_subepisode(self, environment: SymbolicOptionsEnvironment, snapshot: Any, option_atom: ActionAtom, executor: str, substeps: int, batch_size: int, n_iter: int):
        clip = 0.2
        total_reward = 0
        I = 1

        obs = environment.restore_snapshot(snapshot)
        done = False

        while not done:
            batch_obs, batch_actions, batch_log_probs, batch_intrinsic_rewards, reward, I_delta, next_obs, terminated, truncated, info = self._sample_batch(environment, obs, option_atom, executor, min(batch_size, substeps))

            advantages, returns = self._calculate_advantages(environment, option_atom, executor, batch_obs, batch_intrinsic_rewards, terminated, next_obs)

            for _ in range(n_iter):
                vs, log_probs = self._get_vs_log_probs(environment, option_atom, executor, batch_obs, batch_actions)
                
                ratios = torch.exp(log_probs - batch_log_probs)
                
                surr1 = ratios * advantages
                surr2 = torch.clamp(ratios, 1 - clip, 1 + clip) * advantages

                self._physical_extractor_optimizers[executor].zero_grad()
                self._physical_policy_optimizers[executor].zero_grad()
                self._physical_vf_optimizers[executor].zero_grad()
            
                loss = (- torch.min(surr1, surr2)).mean() - 0.5 * ((vs - returns) ** 2).mean()
                loss.backward()

                self._physical_extractor_optimizers[executor].step()
                self._physical_policy_optimizers[executor].step()
                self._physical_vf_optimizers[executor].step()
            
            total_reward += I * reward
            I *= I_delta
            obs = next_obs
            substeps -= 1
            done = terminated or truncated or info["is_option_goal"] or (substeps <= 0)
        
        del info["intrinsic_reward"]

        return total_reward, terminated, truncated, info
    
    def _sample_batch(self, environment: SymbolicOptionsEnvironment, obs: PhysicalObservation, option_atom: ActionAtom, executor: str, batch_size: int):
        batch_obs = []
        batch_actions = []
        batch_log_probs = []
        batch_intrinsic_rewards = []

        gamma = 0.99
        total_reward = 0
        I = 1
        
        done = False

        while not done:
            obs_vec = self._vectorize_observation(environment, option_atom, obs).to(self._device)
            action_dist_mean = self._physical_policies[executor].forward(obs_vec)
            action_dist_corr = 1e-6 * torch.eye(action_dist_mean.shape[0]).to(action_dist_mean)
            action_dist = torch.distributions.MultivariateNormal(action_dist_mean, action_dist_corr)
            action = action_dist.sample()
            action_log_prob = action_dist.log_prob(action)

            next_obs, reward, terminated, truncated, info = environment.step([1, action.cpu().numpy()])

            batch_obs.append(obs)
            batch_actions.append(action)
            batch_log_probs.append(action_log_prob)
            batch_intrinsic_rewards.append(info["intrinsic_reward"])

            total_reward += I * reward
            I *= gamma
            obs = next_obs
            batch_size -= 1
            done = terminated or truncated or info["is_option_goal"] or (batch_size <= 0)

        return (batch_obs, batch_actions, torch.stack(batch_log_probs).detach(), batch_intrinsic_rewards,
                total_reward, I, next_obs, terminated, truncated, info)
    
    def _calculate_advantages(self, environment, option_atom, executor, batch_obs, batch_intrinsic_rewards, terminated, next_obs):
        advantages = []
        returns = []

        gamma = 0.99
        R = 0
        
        if not terminated:
            next_obs_vec = self._vectorize_observation(environment, option_atom, next_obs).to(self._device)
            R = self._physical_vfs[executor].forward(next_obs_vec)
        
        for obs, r in reversed(list(zip(batch_obs, batch_intrinsic_rewards))):
            obs_vec = self._vectorize_observation(environment, option_atom, obs).to(self._device)
            v = self._physical_vfs[executor].forward(obs_vec)
            R = r + gamma * R

            advantages.append((R - v).detach())
            returns.append(R.detach())

        return torch.cat(advantages[::-1]), torch.cat(returns[::-1])
    
    def _get_vs_log_probs(self, environment, option_atom, executor, batch_obs, batch_actions):
        vs = []
        log_probs = []
        
        for obs, a in zip(batch_obs, batch_actions):
            obs_vec = self._vectorize_observation(environment, option_atom, obs).to(self._device)
            v = self._physical_vfs[executor].forward(obs_vec)
            
            action_dist_mean = self._physical_policies[executor].forward(obs_vec)
            action_dist_corr = 1e-6 * torch.eye(action_dist_mean.shape[0]).to(action_dist_mean)
            action_dist = torch.distributions.MultivariateNormal(action_dist_mean, action_dist_corr)
            action_log_prob = action_dist.log_prob(a)

            vs.append(v)
            log_probs.append(action_log_prob)

        return torch.cat(vs), torch.stack(log_probs)
    
    def _sample_episode(self, environment: SymbolicOptionsEnvironment):
        pass
    
    def _sample_subepisode(self, environment: SymbolicOptionsEnvironment, snapshot: Any, option_atom: ActionAtom, executor: str):
        pass

    def train(self, environment: SymbolicOptionsEnvironment, episodes: int, subepisodes: int, substeps: int, batch_size: int, n_iter: int) -> List[float]:
        reward_history = []
        goal_history = []

        self._setup(environment)

        self._lmlp.train()
        for e in self._physical_extractors.values():
            e.train()
        for p in self._physical_policies.values():
            p.train()
        for v in self._physical_vfs.values():
            v.train()

        for i in range(episodes):
            print(f"Episode {i}:")

            total_reward, is_goal = self._train_episode(environment, subepisodes, substeps, batch_size, n_iter)

            reward_history.append(total_reward)
            goal_history.append(is_goal)
            print(f"{total_reward:+.3f} {is_goal}")

        self._trained = True

        print(self._lmlp, file=open("_my/logs/torch_ttt.txt", "w"))

        return reward_history, goal_history

    def evaluate(self, environment: SymbolicOptionsEnvironment, episodes: int) -> List[float]:
        if not self._trained:
            raise RuntimeError("Agent is not trained")
        
        reward_history = []
        goal_history = []

        self._lmlp.eval()
        for e in self._physical_extractors.values():
            e.eval()
        for p in self._physical_policies.values():
            p.eval()
        for v in self._physical_vfs.values():
            v.eval()

        for i in range(episodes):
            print(f"Episode {i}:", end=" ")
            
            total_reward, is_goal = self._sample_episode(environment)
            reward_history.append(total_reward)
            goal_history.append(is_goal)

        return reward_history, goal_history
