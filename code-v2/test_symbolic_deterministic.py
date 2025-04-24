import argparse
from pathlib import Path

import gymnasium as gym

import torch

from tianshou.env import DummyVectorEnv

from nesyrl.envs.symbolic import BlocksWorld
from nesyrl.agents.symbolic import A2CPolicy
from nesyrl.util.logging import FileLogger
from nesyrl.util.collecting import SuccessCollector


parser = argparse.ArgumentParser()
parser.add_argument("--dir", help="path to the directory with models to test", type=str, required=True)

device = "cuda" if torch.cuda.is_available() else "cpu"
torch.set_default_dtype(torch.float32)

gym.register(id="nesyrl-symbolic/BlocksWorld-v0", entry_point=BlocksWorld)

args = {
    "test_env_kwargs":  {
        "horizon": 50,
        "blocks": ["a", "b", "c", "d"],
        "goal_state": [["a", "b", "c", "d"]]
    },
    "test_test_seed": 47,
    "test_episides": 1000,
    "policy" : {
        "gae_lambda": 0.9,
        "ent_coef": 0,
        "max_batchsize": 1,
        "deterministic_eval": True
    }
}


if __name__ == "__main__":
    args |= vars(parser.parse_args())

    logger = FileLogger(f"test.txt")

    i = 0
    path = Path(f'{args["dir"]}/run_{i}_actor.model')

    while path.exists():
        # Prepare environment
        test_env = gym.make("nesyrl-symbolic/BlocksWorld-v0", **args["test_env_kwargs"])
        test_venv = DummyVectorEnv([lambda: test_env])

        # Prepare agent
        actor = torch.load(f'{args["dir"]}/run_{i}_actor.model', device)
        critic = torch.load(f'{args["dir"]}/run_{i}_critic.model', device)

        to_device = lambda m: setattr(m, "device", device) if hasattr(m, "device") else None
        actor.apply(to_device)
        critic.apply(to_device)
        
        optim = torch.load(f'{args["dir"]}/run_{i}.optim', device)

        policy = A2CPolicy(
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
        path = Path(f'{args["dir"]}/run_{i}_actor.model')
