from __future__ import annotations
from typing import List, Tuple, Any

import torch
from torch import Tensor
import collections

from ..base import Agent
from envs.hierarchical import SymbolicOptionsEnvironment, Grasp, Release, ReachForGrasp, ReachForRelease, ActionAtom, PhysicalObservation, Valuation
from lmlp import *
from envs.physical.util import squared_length
from .util import *


torch.set_default_dtype(torch.float64)
torch.serialization.add_safe_globals([torch.nn.Sequential, torch.nn.Linear, torch.nn.Tanh, torch.optim.RMSprop, set, dict, collections.defaultdict])

class A2CNCE(Agent):

    _trained: bool = False

    def setup(self, environment: SymbolicOptionsEnvironment):
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

        self._lmlp_optimizer = torch.optim.RMSprop(self._lmlp.parameters(), lr=1e-2)

        self._lmlp_v = dict()

    def _setup_physical(self, environment: SymbolicOptionsEnvironment):
        self._physical_extractors = {
            "reach_grasp":
                torch.nn.Sequential(
                    torch.nn.Linear(len(environment.env._active_joints) + 7 + 3 + 3, 64),
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
                    torch.nn.Linear(64, environment.action_space.spaces[1].shape[0]),
                    # ScaledSigmoid(environment.action_space.spaces[1].low, environment.action_space.spaces[1].high),
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
            key=lambda b: squared_length(observation[b][:3])
        )[0]

        joints = torch.tensor([observation[j] for j in observation if j.startswith("joint")])
        
        if isinstance(option_atom, ReachForGrasp):
            return torch.concatenate((
                joints,
                torch.tensor(observation[environment.env.eeff_obs_key]),
                torch.tensor(observation[option_atom.args[0]][:3]),
                torch.tensor(observation[closest_non_target_block][:3])
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
    
    def _sample_episode(self, environment: SymbolicOptionsEnvironment):
        total_reward = 0
        
        environment.reset()
        done = False

        while not done:
            val = environment.get_valuation()
            val_vec = self._vectorize_valuation(val).to(self._device)
            option_probs = self._lmlp.forward(val_vec)
            option_dist = torch.distributions.Multinomial(1, option_probs)
            option = option_dist.sample()
            option = (option * torch.arange(len(option)).to(option)).sum(dtype=torch.int64)

            print(environment.option_atoms[option], end=" ")
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

                reward, terminated, truncated, info = self._sample_subepisode(environment, snapshot, option_atom, executor)
            
            total_reward += reward
            done = terminated or truncated

        print()
        
        return total_reward, info["is_goal"]
    
    def _sample_subepisode(self, environment: SymbolicOptionsEnvironment, snapshot: Any, option_atom: ActionAtom, executor: str):
        total_reward = 0
        
        obs = environment.restore_snapshot(snapshot)
        done = False

        while not done:
            obs_vec = self._vectorize_observation(environment, option_atom, obs).to(self._device)
            action_dist_mean = self._physical_policies[executor].forward(obs_vec)
            action_dist_corr = 1e-6 * torch.eye(action_dist_mean.shape[0]).to(action_dist_mean)
            action_dist = torch.distributions.MultivariateNormal(action_dist_mean, action_dist_corr)
            action = action_dist.sample()

            next_obs, reward, terminated, truncated, info = environment.step([1, action.cpu().numpy()])

            total_reward += reward
            obs = next_obs
            done = terminated or truncated or info["is_option_goal"]
        
        del info["intrinsic_reward"]
        del info["is_option_goal"]

        return total_reward, terminated, truncated, info
    
    def _train_metacontroller(self, environment: SymbolicOptionsEnvironment, steps: int, substeps: int, subrollout_steps: int):
        reward_history = []
        goal_history = []
        gamma = 0.99

        i = 1
        while steps > 0:
            print(f"Episode {i}:")

            environment.reset()
            total_reward = 0
            I = 1
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
                    reward, terminated, truncated, info = self._train_controller(environment, snapshot, option_atom, executor, substeps)
                
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
                steps -= 1
                i += 1
                done = terminated or truncated or (steps <= 0) or True

            reward_history.append(total_reward)
            goal_history.append(info["is_goal"])

            print(f"{total_reward:+.3f} {info["is_goal"]}")
        
        return reward_history, goal_history
    
    def _train_controller(self, environment: SymbolicOptionsEnvironment, snapshot: Any, option_atom: ActionAtom, executor: str, substeps: int):
        option_goals = 0
        subepisodes = 1

        while substeps > 0:
            print(f"    {subepisodes}", end="\r", flush=True)

            obs = environment.restore_snapshot(snapshot)
            total_reward = 0
            done = False

            while not done:
                obs_vec = self._vectorize_observation(environment, option_atom, obs).to(self._device)
                action_dist_mean = self._physical_policies[executor].forward(obs_vec)
                action_dist_corr = 1e-6 * torch.eye(action_dist_mean.shape[0]).to(action_dist_mean)
                action_dist = torch.distributions.MultivariateNormal(action_dist_mean, action_dist_corr)
                action = action_dist.sample()
                action_log_prob = action_dist.log_prob(action)

                next_obs, reward, terminated, truncated, info = environment.step([1, action.cpu().numpy()])

                if not terminated and not info["is_option_goal"]:
                    next_obs_vec = self._vectorize_observation(environment, option_atom, next_obs).to(self._device)
                    next_v = self._physical_vfs[executor].forward(next_obs_vec)
                else:
                    next_v = 0
                
                v = self._physical_vfs[executor].forward(obs_vec)
                advantage = (info["intrinsic_reward"] + next_v - v).detach()

                self._physical_extractor_optimizers[executor].zero_grad()
                self._physical_policy_optimizers[executor].zero_grad()
                self._physical_vf_optimizers[executor].zero_grad()

                loss = - advantage * action_log_prob.sum(dim=-1) - advantage * v
                loss.backward()

                self._physical_extractor_optimizers[executor].step()
                self._physical_policy_optimizers[executor].step()
                self._physical_vf_optimizers[executor].step()
                
                total_reward += reward
                obs = next_obs
                substeps -= 1
                done = terminated or truncated or info["is_option_goal"] or (substeps <= 0)

            option_goals += info["is_option_goal"]
            subepisodes += 1
        
        del info["intrinsic_reward"]
        
        print(f"    {option_goals}/{subepisodes}", flush=True)

        return total_reward, terminated, truncated, info
    
    # def _train_controller(self, environment: SymbolicOptionsEnvironment, snapshot: Any, option_atom: ActionAtom, executor: str, substeps: int, subrollout_steps: int):
    #     option_goals = 0
    #     subepisodes = 0
    #     done = True

    #     while substeps > 0:
    #         print(f"    {subepisodes}", end="\r", flush=True)

    #         if done:
    #             obs = environment.restore_snapshot(snapshot)

    #         steps_delta = min(substeps, subrollout_steps)
    #         obs, total_reward, terminated, truncated, info, returns, values, log_probs, og, se = self._subrollout(environment, snapshot, option_atom, executor, obs, steps_delta)

    #         advantages = (returns.to(values) - values)

    #         self._physical_extractor_optimizers[executor].zero_grad()
    #         self._physical_policy_optimizers[executor].zero_grad()
    #         self._physical_vf_optimizers[executor].zero_grad()

    #         loss = - (advantages.detach() * log_probs).mean() + (advantages ** 2).mean()
    #         print(loss)
    #         loss.backward()

    #         self._physical_extractor_optimizers[executor].step()
    #         self._physical_policy_optimizers[executor].step()
    #         self._physical_vf_optimizers[executor].step()

    #         option_goals += og
    #         subepisodes += se
    #         substeps -= steps_delta
    #         done = terminated or truncated or info["is_option_goal"]
        
    #     print(f"    {option_goals}/{subepisodes}", flush=True)

    #     return total_reward, terminated, truncated, info
    
    # def _subrollout(self, environment: SymbolicOptionsEnvironment, snapshot: Any, option_atom: ActionAtom, executor: str, obs, subrollout_steps: int):
    #     option_goals = 0
    #     subepisodes = 0
    #     returns = []
    #     values = []
    #     log_probs = []

    #     while subrollout_steps > 0:
    #         total_reward = 0
    #         done = False

    #         ep_returns = []

    #         while not done and subrollout_steps > 0:
    #             obs_vec = self._vectorize_observation(environment, option_atom, obs).to(self._device)
    #             action_dist_mean = self._physical_policies[executor].forward(obs_vec)
    #             action_dist_corr = 1e-6 * torch.eye(action_dist_mean.shape[0]).to(action_dist_mean)
    #             action_dist = torch.distributions.MultivariateNormal(action_dist_mean, action_dist_corr)
    #             action = action_dist.sample()
    #             action_log_prob = action_dist.log_prob(action)

    #             next_obs, reward, terminated, truncated, info = environment.step([1, action.cpu().numpy()])
                
    #             values.append(self._physical_vfs[executor].forward(obs_vec))
    #             log_probs.append(action_log_prob)
                
    #             ep_returns.append(0)
    #             for t in range(len(ep_returns)):
    #                 ep_returns[t] += info["intrinsic_reward"]
                
    #             total_reward += reward
    #             obs = next_obs
    #             subrollout_steps -= 1
    #             done = terminated or truncated or info["is_option_goal"]

    #         if not terminated and not info["is_option_goal"]:
    #             obs_vec = self._vectorize_observation(environment, option_atom, obs).to(self._device)
    #             v = self._physical_vfs[executor].forward(obs_vec)

    #             for t in range(len(ep_returns)):
    #                 ep_returns[t] += float(v)

    #         returns.extend(ep_returns)

    #         option_goals += info["is_option_goal"]
    #         subepisodes += done

    #         if subrollout_steps > 0:
    #             obs = environment.restore_snapshot(snapshot)
        
    #     del info["intrinsic_reward"]

    #     return obs, total_reward, terminated, truncated, info, torch.tensor(returns), torch.stack(values), torch.stack(log_probs), option_goals, subepisodes

    def train(self, environment: SymbolicOptionsEnvironment, steps: int, substeps: int, subrollout_steps: int) -> List[float]:
        self._lmlp.train()
        for e in self._physical_extractors.values():
            e.train()
        for p in self._physical_policies.values():
            p.train()
        for v in self._physical_vfs.values():
            v.train()

        reward_history, goal_history = self._train_metacontroller(environment, steps, substeps, subrollout_steps)

        self._trained = True

        # print(self._lmlp, file=open("_my/logs/torch_ttt.txt", "w"))
        
        torch.save(self._physical_extractors["reach_grasp"], "e.model")
        torch.save(self._physical_policies["reach_grasp"], "p.model")
        torch.save(self._physical_vfs["reach_grasp"], "v.model")

        torch.save(self._physical_extractor_optimizers["reach_grasp"], "e.opt")
        torch.save(self._physical_policy_optimizers["reach_grasp"], "p.opt")
        torch.save(self._physical_vf_optimizers["reach_grasp"], "v.opt")

        return reward_history, goal_history
    
    def load(self, physical_extractors, physical_policies, physical_vfs, physical_extractor_optimizers, physical_policy_optimizers, physical_vf_optimizers):
        for key in physical_extractors:
            self._physical_extractors[key] = torch.load(physical_extractors[key], weights_only=True)
        for key in physical_policies:
            self._physical_policies[key] = torch.load(physical_policies[key], weights_only=True)
        for key in physical_vfs:
            self._physical_vfs[key] = torch.load(physical_vfs[key], weights_only=True)

        for key in physical_extractor_optimizers:
            self._physical_extractor_optimizers[key] = torch.load(physical_extractor_optimizers[key], weights_only=True)
        for key in physical_policy_optimizers:
            self._physical_policy_optimizers[key] = torch.load(physical_policy_optimizers[key], weights_only=True)
        for key in physical_vf_optimizers:
            self._physical_vf_optimizers[key] = torch.load(physical_vf_optimizers[key], weights_only=True)

    def evaluate(self, environment: SymbolicOptionsEnvironment, episodes: int) -> List[float]:
        # if not self._trained:
        #     raise RuntimeError("Agent is not trained")
        
        reward_history = []
        goal_history = []

        self._lmlp.eval()
        for e in self._physical_extractors.values():
            e.eval()
        for p in self._physical_policies.values():
            p.eval()
        for v in self._physical_vfs.values():
            v.eval()

        log = environment.env._pbc.startStateLogging(environment.env._pbc.STATE_LOGGING_VIDEO_MP4, "eval_b.mp4")

        for i in range(episodes):
            print(f"Episode {i}:", end=" ")
            
            total_reward, is_goal = self._sample_episode(environment)
            reward_history.append(total_reward)
            goal_history.append(is_goal)

            if i == 9:
                environment.env._pbc.stopStateLogging(log)

        return reward_history, goal_history
