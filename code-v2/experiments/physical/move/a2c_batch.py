import time
import argparse
import json
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch
from torch.optim import Optimizer, RMSprop, Adam

from tianshou.data import Collector, ReplayBuffer
from tianshou.env import DummyVectorEnv, VectorEnvNormObs
from tianshou.trainer import OnpolicyTrainer
from tianshou.policy import A2CPolicy
from tianshou.utils.net.common import Net, ActorCritic
from tianshou.utils.net.continuous import ActorProb, Critic
from tianshou.utils.statistics import RunningMeanStd

from nesyrl.envs.physical import NicoBlocksWorldMove
from nesyrl.util.logging import FileLogger
from nesyrl.util.collecting import SuccessCollector

# TODO: ref. https://arxiv.org/pdf/2006.05990

def init_weights(module: torch.nn.Module) -> None:
    if isinstance(module, torch.nn.Linear):
        torch.nn.init.orthogonal_(module.weight, gain=np.sqrt(2))
        torch.nn.init.zeros_(module.bias)

def rescale_weights(module: torch.nn.Module, scale: float) -> None:
    if isinstance(module, torch.nn.Linear):
        module.weight.data.copy_(scale * module.weight.data)

def save_obs_norm_params(rms: RunningMeanStd, path: str) -> None:
    with open(path, "w") as out:
        for attr in (rms.mean, rms.var, rms.clip_max, rms.count, rms.eps):
            print(attr, file=out)

def save_run(
    run: int, actor: ActorProb, critic: Critic, optim: Optimizer, rms: RunningMeanStd
) -> None:
    torch.save(actor, f"{log_dir}/run_{run}_actor.model")
    torch.save(critic, f"{log_dir}/run_{run}_critic.model")
    torch.save(optim, f"{log_dir}/run_{run}.optim")
    save_obs_norm_params(rms, f"{log_dir}/run_{run}.norm")

def next_epoch(
    ep: int, run: int, actor: ActorProb, critic: Critic, optim: Optimizer,
    train_venv: VectorEnvNormObs, test_venv: VectorEnvNormObs
) -> None:
    if ep % 10 == 0:
        save_run(run, actor, critic, optim, train_venv.get_obs_rms())
    
    test_venv.set_obs_rms(train_venv.get_obs_rms())


# Basic configuration
optimizers = {
    "rms": RMSprop,
    "adam": Adam
}

parser = argparse.ArgumentParser()
parser.add_argument("--num-runs", help="number of times to run the experiment", type=int, choices=range(1, 11), required=True)
parser.add_argument("--actor-hidden", help="sizes of hidden layers in the actor net", nargs='+', type=int, required=True)
parser.add_argument("--critic-hidden", help="sizes of hidden layers in the critic net", nargs='+', type=int, required=True)
parser.add_argument("--optim", help="optimizer of both the actor's and the critic's parameters", choices=optimizers, required=True)
parser.add_argument("--lr", help="learning rate for both the actor and the critic", type=float, required=True)

log_dir = f"results/physical/move/a2c_batch_{time.strftime('%d%m%Y_%H%M%S', time.gmtime(time.time()))}"
device = "cuda" if torch.cuda.is_available() else "cpu"
torch.set_default_dtype(torch.float32)

gym.register(id="nesyrl-physical/NicoBlocksWorldMove-v0", entry_point=NicoBlocksWorldMove)

args = {
    "train_env_kwargs":  {
        "horizon": 4096,
        "blocks": ["a", "b", "c", "d"],
        "simulation_steps": 1
    },
    "test_env_kwargs":  {
        "horizon": 4096,
        "blocks": ["a", "b", "c", "d"],
        "simulation_steps": 1,
        "render_mode": None
    },
    "test_train_seed": 42,
    "test_test_seed": 47,
    "test_episides": 100,
    "actor": {
        "mu_scale": 0.01,
        "sigma_param": -0.5
    },
    "policy" : {
        "gae_lambda": 0.95,
        "ent_coef": 0,
        "max_batchsize": 64,
        # "reward_normalization": True,
    },
    "trainer": {
        "max_epoch": 50,
        "step_per_epoch": 16384,
        "repeat_per_collect": 10,
        "episode_per_test": 1,
        "step_per_collect": 1024,
        "batch_size": 64
    }
}


if __name__ == "__main__":
    args |= vars(parser.parse_args())

    # Prepare log directory
    path = Path(log_dir)
    path.mkdir(parents=True, exist_ok=True)

    with open(f"{log_dir}/args.txt", "w") as out:
        json.dump(args, out, indent=4)

    for run in range(args["num_runs"]):
        # Prepare environments
        train_env = gym.make("nesyrl-physical/NicoBlocksWorldMove-v0", **args["train_env_kwargs"])
        test_env = gym.make("nesyrl-physical/NicoBlocksWorldMove-v0", **args["test_env_kwargs"])
        test_env.reset(seed=args["test_train_seed"])

        train_venv = VectorEnvNormObs(DummyVectorEnv([lambda: train_env]))
        test_venv = VectorEnvNormObs(DummyVectorEnv([lambda: test_env]), False)
        test_venv.set_obs_rms(train_venv.get_obs_rms())

        # Prepare agent
        net = Net(state_shape=train_env.observation_space.shape,
                  hidden_sizes=args["actor_hidden"],
                  activation=torch.nn.Tanh, device=device).to(device)
        actor = ActorProb(preprocess_net=net, action_shape=train_env.action_space.shape,
                          unbounded=True, device=device).to(device)
        
        net = Net(state_shape=train_env.observation_space.shape,
                  hidden_sizes=args["critic_hidden"],
                  activation=torch.nn.Tanh, device=device).to(device)
        critic = Critic(preprocess_net=net, device=device).to(device)
        
        actor_critic = ActorCritic(actor, critic)

        actor_critic.apply(init_weights)
        actor.mu.apply(lambda m: rescale_weights(m, args["actor"]["mu_scale"]))
        torch.nn.init.constant_(actor.sigma_param, args["actor"]["sigma_param"])
        
        optim = optimizers[args["optim"]](actor_critic.parameters(), lr=args["lr"])

        policy = A2CPolicy(
            actor=actor,
            critic=critic,
            optim=optim,
            dist_fn=torch.distributions.Normal,
            action_scaling=True,
            action_space=train_env.action_space,
            action_bound_method="tanh",
            **args["policy"]
        )

        # Prepare training
        train_collector = Collector(policy, train_venv, ReplayBuffer(args["trainer"]["step_per_collect"]))
        test_collector = SuccessCollector(policy, test_venv)

        logger = FileLogger(f"{log_dir}/run_{run}_log.txt")

        trainer = OnpolicyTrainer(
            policy=policy,
            train_collector=train_collector,
            test_collector=test_collector,
            logger=logger,
            test_fn=lambda ep, _: next_epoch(ep, run, actor, critic, optim, train_venv, test_venv),
            **args["trainer"]
        )

        # Train
        trainer.run()

        # Save models
        save_run(run, actor, critic, optim, train_venv.get_obs_rms())

        # Test
        policy.eval()
        test_collector.reset(gym_reset_kwargs={"seed": args["test_test_seed"]})
        test_venv.set_obs_rms(train_venv.get_obs_rms())
        result = test_collector.collect(n_episode=args["test_episides"], render=False)
        
        logger = FileLogger(f"{log_dir}/test.txt")
        logger.log_test_data(result, 0)
