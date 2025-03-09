import argparse

import gymnasium as gym
import numpy as np
import torch

from tianshou.env import DummyVectorEnv, VectorEnvNormObs
from tianshou.policy import A2CPolicy
from tianshou.utils.net.common import ActorCritic
from tianshou.utils.statistics import RunningMeanStd

from nesyrl.envs.physical import NicoBlocksWorldMove
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
parser.add_argument("--dir", help="path to the directory with the model to test", type=str, required=True)
parser.add_argument("--run", help="index of the run to test", type=int, required=True)
parser.add_argument("--ep", help="index of the run to test", type=int, default=None)

device = "cuda" if torch.cuda.is_available() else "cpu"
torch.set_default_dtype(torch.float32)

gym.register(id="nesyrl-physical/NicoBlocksWorldMove-v0", entry_point=NicoBlocksWorldMove)

env_kwargs = {
    "horizon": 8192,
    "blocks": ["a", "b", "c", "d"],
    "simulation_steps": 1
}

if __name__ == "__main__":
    args = parser.parse_args()
    ep_str = f"_ep_{args.ep}" if args.ep is not None else ""

    # Prepare environment
    test_env = gym.make("nesyrl-physical/NicoBlocksWorldMove-v0", **env_kwargs)
    test_venv = VectorEnvNormObs(DummyVectorEnv([lambda: test_env]), False)
    test_venv.set_obs_rms(load_obs_norm_params(f"{args.dir}/run_{args.run}{ep_str}_norm.txt"))

    # Prepare agent
    actor = torch.load(f"{args.dir}/run_{args.run}{ep_str}_actor.model")
    critic = torch.load(f"{args.dir}/run_{args.run}{ep_str}_critic.model")
    
    actor_critic = ActorCritic(actor, critic)

    optim = torch.load(f"{args.dir}/run_{args.run}{ep_str}.optim")

    policy = A2CPolicy(
        actor=actor,
        critic=critic,
        optim=optim,
        dist_fn=torch.distributions.Normal,
        action_scaling=True,
        action_bound_method="tanh",
        gae_lambda=1,
        ent_coef=0,
        max_batchsize=1
    )

    # Test
    test_collector = SuccessCollector(policy, test_venv)
    
    policy.eval()
    test_collector.reset(gym_reset_kwargs={"seed": 47})
    result = test_collector.collect(n_episode=1000, render=False)

    print(result)
