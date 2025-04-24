import time
import json
from pathlib import Path

import gymnasium as gym

from tianshou.env import DummyVectorEnv
from tianshou.trainer.utils import test_episode

from nesyrl.envs.symbolic import BlocksWorld
from nesyrl.planning import BlocksWorldASPPlanner
from nesyrl.util.logging import FileLogger
from nesyrl.util.collecting import SuccessCollector


# Basic configuration
log_dir = f"results/symbolic/10_blocks/planning_{time.strftime('%d%m%Y_%H%M%S', time.gmtime(time.time()))}"

gym.register(id="nesyrl-symbolic/BlocksWorld-v0", entry_point=BlocksWorld)

args = {
    "test_env_kwargs":  {
        "horizon": 50,
        "blocks": ["a", "b", "c", "d", "e", "f"],
        "goal_state": [["a", "b", "c", "d", "e", "f"]]
    },
    "test_train_seed": 42,
    "test_test_seed": 47,
    "test_episides": 1000,
    "train_epochs": 300,
    "episode_per_epoch": 100
}


if __name__ == "__main__":
    # Prepare log directory
    path = Path(log_dir)
    path.mkdir(parents=True, exist_ok=True)

    with open(f"{log_dir}/args.txt", "w") as out:
        json.dump(args, out, indent=4)

    # Prepare environments
    test_env = gym.make("nesyrl-symbolic/BlocksWorld-v0", **args["test_env_kwargs"])
    test_env.reset(seed=args["test_train_seed"])

    test_venv = DummyVectorEnv([lambda: test_env])

    # Prepare agent
    planner = BlocksWorldASPPlanner(test_env.unwrapped, log_dir)

    # Prepare collector
    test_collector = SuccessCollector(planner, test_venv)

    # Plan
    logger = FileLogger(f"{log_dir}/log.txt")

    for ep in range(args["train_epochs"] + 1):
        test_episode(
            planner, test_collector, None, ep, args["episode_per_epoch"],
            logger, 0
        )
    
    test_collector.reset(gym_reset_kwargs={"seed": args["test_test_seed"]})
    result = test_collector.collect(n_episode=args["test_episides"], render=False)
    
    logger = FileLogger(f"{log_dir}/test.txt")
    logger.log_test_data(result, 0)
