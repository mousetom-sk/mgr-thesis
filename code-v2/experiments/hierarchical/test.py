import time
import argparse
import json
from pathlib import Path

import gymnasium as gym

import torch
import numpy as np

from tianshou.env import DummyVectorEnv, VectorEnvNormObs
from tianshou.policy import A2CPolicy
from tianshou.utils.net.common import ActorCritic
from tianshou.utils.statistics import RunningMeanStd

from nesyrl.envs.hierarchical import BlocksWorldHierarchical
from nesyrl.envs.physical.blocks_world_hierarchical import NicoBlocksWorld
from nesyrl.agents.symbolic import PPOPolicy
from nesyrl.util.logging import FileLogger
from nesyrl.util.collecting import SuccessCollector


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


parser = argparse.ArgumentParser()
parser.add_argument("--sym-dir", help="path to the directory with the symbolic models to test", type=str, required=True)
parser.add_argument("--phys-dir", help="path to the directory with the physical model to use", type=str, required=True)
parser.add_argument("--phys-run", help="the run of the physical model to use", type=str, required=True)

log_dir = f"results/hierarchical/test_{time.strftime('%d%m%Y_%H%M%S', time.gmtime(time.time()))}"
device = "cuda" if torch.cuda.is_available() else "cpu"
torch.set_default_dtype(torch.float32)

gym.register(id="nesyrl-hierarchical/BlocksWorld-v0", entry_point=BlocksWorldHierarchical)

args = {
    "test_env_kwargs":  {
        "horizon": 50,
        "blocks": ["a", "b", "c", "d"],
        "goal_state": [["a", "b", "c", "d"]]
    },
    "phys_env_kwargs" : {
        "horizon": 4096,
        "blocks": ["a", "b", "c", "d"],
        "simulation_steps": 1,
        "render_mode": None
    },
    "test_test_seed": 47,
    "test_episides": 100,
    "policy" : {
        "gae_lambda": 0.9,
        "ent_coef": 0,
        "max_batchsize": 1
    },
    "phys_policy" : {
        "gae_lambda": 1,
        "ent_coef": 0,
        "max_batchsize": 1
    }
}


if __name__ == "__main__":
    args |= vars(parser.parse_args())

    # Prepare log directory
    path = Path(log_dir)
    path.mkdir(parents=True, exist_ok=True)

    with open(f"{log_dir}/args.txt", "w") as out:
        json.dump(args, out, indent=4)

    logger = FileLogger(f"{log_dir}/test.txt")

    env = NicoBlocksWorld(**args["phys_env_kwargs"])
    venv = VectorEnvNormObs(DummyVectorEnv([lambda: env]), False)
    venv.set_obs_rms(load_obs_norm_params(f'{args["phys_dir"]}/run_{args["phys_run"]}_norm.txt'))

    actor = torch.load(f'{args["phys_dir"]}/run_{args["phys_run"]}_actor.model', device)
    critic = torch.load(f'{args["phys_dir"]}/run_{args["phys_run"]}_critic.model', device)

    to_device = lambda m: setattr(m, "device", device) if hasattr(m, "device") else None
    actor.apply(to_device)
    critic.apply(to_device)

    actor_critic = ActorCritic(actor, critic)

    optim = torch.load(f'{args["phys_dir"]}/run_{args["phys_run"]}.optim', device)

    policy = A2CPolicy(
        actor=actor,
        critic=critic,
        optim=optim,
        dist_fn=torch.distributions.Normal,
        action_scaling=True,
        action_space=venv.action_space,
        action_bound_method="tanh",
        **args["phys_policy"]
    )

    args["test_env_kwargs"]["physical_env"] = venv
    args["test_env_kwargs"]["physical_policy"] = policy

    i = 0
    path = Path(f'{args["sym_dir"]}/run_{i}_actor.model')

    while path.exists():
        # Prepare environment
        test_env = gym.make("nesyrl-hierarchical/BlocksWorld-v0", **args["test_env_kwargs"])
        test_venv = DummyVectorEnv([lambda: test_env])

        # Prepare agent
        actor = torch.load(f'{args["sym_dir"]}/run_{i}_actor.model', device)
        critic = torch.load(f'{args["sym_dir"]}/run_{i}_critic.model', device)

        to_device = lambda m: setattr(m, "device", device) if hasattr(m, "device") else None
        actor.apply(to_device)
        critic.apply(to_device)
        
        optim = torch.load(f'{args["sym_dir"]}/run_{i}.optim', device)

        policy = PPOPolicy(
            actor=actor,
            critic=critic,
            optim=optim,
            dist_fn=torch.distributions.Categorical,
            action_scaling=False,
            action_space=test_env.action_space,
            **args["policy"]
        )

        # Test
        test_collector = SuccessCollector(policy, test_venv)

        policy.eval()
        test_collector.reset(gym_reset_kwargs={"seed": args["test_test_seed"]})
        result = test_collector.collect(n_episode=args["test_episides"], render=False)
        
        logger.log_test_data(result, 0)

        i += 1
        path = Path(f'{args["sym_dir"]}/run_{i}_actor.model')
