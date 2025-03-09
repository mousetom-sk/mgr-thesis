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

def load_obs_norm_params(path: str) -> RunningMeanStd:
    rms = RunningMeanStd()
    keys = ["mean", "var", "clip_max", "count", "eps"]
    attrs = []
    is_array = False

    with open(path, "r") as src:
        for line in src:
            line = line[:-1]
            if line.startswith("["):
                attrs.append([])
                is_array = True
                attrs[-1].extend([float(x) for x in line[1:].split()])
            elif line.endswith("]"):
                is_array = False
                attrs[-1].extend([float(x) for x in line[:-1].split()])
                attrs[-1] = np.array(attrs[-1])
            elif is_array:
                attrs[-1].extend([float(x) for x in line.split()])
            else:
                attrs.append(float(line))

    for k, v in zip(keys, attrs):
        rms.__setattr__(k, v)
    
    return rms

def save_obs_norm_params(rms: RunningMeanStd, path: str) -> None:
    with open(path, "w") as out:
        for attr in (rms.mean, rms.var, rms.clip_max, rms.count, rms.eps):
            print(attr, file=out)

def save_run(
    run: int, ep: int | None, actor: ActorProb, critic: Critic, optim: Optimizer, rms: RunningMeanStd
) -> None:
    ep_str = f"_ep_{ep}" if ep is not None else ""

    torch.save(actor, f"{log_dir}/run_{run}{ep_str}_actor.model")
    torch.save(critic, f"{log_dir}/run_{run}{ep_str}_critic.model")
    torch.save(optim, f"{log_dir}/run_{run}{ep_str}.optim")

    save_obs_norm_params(rms, f"{log_dir}/run_{run}{ep_str}_norm.txt")

def next_epoch(
    ep: int, run: int, actor: ActorProb, critic: Critic, optim: Optimizer,
    train_venv: VectorEnvNormObs, test_venv: VectorEnvNormObs
) -> None:
    if ep % 10 == 0:
        save_run(run, ep, actor, critic, optim, train_venv.get_obs_rms())
    
    test_venv.set_obs_rms(train_venv.get_obs_rms())


# Basic configuration
optimizers = {
    "rms": RMSprop,
    "adam": Adam
}

parser = argparse.ArgumentParser()
parser.add_argument("--dir", help="path to the directory with the model to continue training", type=str, required=True)
parser.add_argument("--run", help="index of the run to continue training", type=int, required=True)
parser.add_argument("--ep", help="index of the run to continue training", type=int, required=True)

log_dir = f"results/physical/move/a2c_cont_{time.strftime('%d%m%Y_%H%M%S', time.gmtime(time.time()))}"
device = "cuda:1" if torch.cuda.is_available() else "cpu"
torch.set_default_dtype(torch.float32)

gym.register(id="nesyrl-physical/NicoBlocksWorldMove-v0", entry_point=NicoBlocksWorldMove)

args = {
    "train_env_kwargs":  {
        "horizon": 8192,
        "blocks": ["a", "b", "c", "d"],
        "simulation_steps": 1,
        "render_mode": None
    },
    "test_env_kwargs":  {
        "horizon": 8192,
        "blocks": ["a", "b", "c", "d"],
        "simulation_steps": 1,
        "render_mode": None
    },
    "test_train_seed": 43,
    "test_test_seed": 47,
    "test_episides": 100,
    "actor": {
        "mu_scale": 0.01,
        "sigma_param": -0.5
    },
    "policy" : {
        "gae_lambda": 1,
        "ent_coef": 0,
        "max_batchsize": 1
    },
    "trainer": {
        "max_epoch": 50,
        "step_per_epoch": 32768,
        "repeat_per_collect": 1,
        "episode_per_test": 3,
        "step_per_collect": 1,
        "batch_size": 1
    }
}


if __name__ == "__main__":
    args |= vars(parser.parse_args())

    # Prepare log directory
    path = Path(log_dir)
    path.mkdir(parents=True, exist_ok=True)

    with open(f"{log_dir}/args.txt", "w") as out:
        json.dump(args, out, indent=4)

    run = 0

    # Prepare environments
    train_env = gym.make("nesyrl-physical/NicoBlocksWorldMove-v0", **args["train_env_kwargs"])
    test_env = gym.make("nesyrl-physical/NicoBlocksWorldMove-v0", **args["test_env_kwargs"])
    test_env.reset(seed=args["test_train_seed"])

    train_venv = VectorEnvNormObs(DummyVectorEnv([lambda: train_env]))
    train_venv.set_obs_rms(load_obs_norm_params(f"{args['dir']}/run_{args['run']}_ep_{args['ep']}_norm.txt"))
    test_venv = VectorEnvNormObs(DummyVectorEnv([lambda: test_env]), False)
    test_venv.set_obs_rms(train_venv.get_obs_rms())

    # Prepare agent
    actor = torch.load(f"{args['dir']}/run_{args['run']}_ep_{args['ep']}_actor.model")
    critic = torch.load(f"{args['dir']}/run_{args['run']}_ep_{args['ep']}_critic.model")
    
    actor_critic = ActorCritic(actor, critic)

    optim = torch.load(f"{args['dir']}/run_{args['run']}_ep_{args['ep']}.optim")

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
    save_run(run, None, actor, critic, optim, train_venv.get_obs_rms())

    # Test
    policy.eval()
    test_collector.reset(gym_reset_kwargs={"seed": args["test_test_seed"]})
    test_venv.set_obs_rms(train_venv.get_obs_rms())
    result = test_collector.collect(n_episode=args["test_episides"], render=False)
    
    logger = FileLogger(f"{log_dir}/test.txt")
    logger.log_test_data(result, 0)
