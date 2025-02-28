import time
import argparse
from pathlib import Path

import gymnasium as gym

import torch

from tianshou.data import Collector
from tianshou.env import DummyVectorEnv
from tianshou.trainer import OnpolicyTrainer

from nesyrl.envs.symbolic import BlocksWorld
from nesyrl.agents.symbolic import Actor, CriticTab, ActorCriticOptimizer, PGPolicy
from nesyrl.logic.neural import ConstantInitializer, UniformInitializer
from nesyrl.util.logging import FileLogger
from nesyrl.util.collecting import SuccessCollector


def log_params(ep: int, actor_path: str, critic_path: str) -> None:
    with open(actor_path, "a") as out:
        print(f"\n>>> {ep}\n", file=out)
        print(actor.params_str(), file=out)

    with open(critic_path, "a") as out:
        print(f"\n>>> {ep}\n", file=out)
        print(critic.params_str(), file=out)


# Basic configuration
initializers = {
    "constant": ConstantInitializer,
    "uniform": UniformInitializer
}

optimizers = {
    "rms": torch.optim.RMSprop,
    "adam": torch.optim.Adam
}

parser = argparse.ArgumentParser()
parser.add_argument("--num-runs", help="number of times to run the experiment", type=int, choices=range(1, 11), required=True)
parser.add_argument("--actor-init", help="actor's NLN weight initializer", choices=initializers, required=True)
parser.add_argument("--actor-init-arg", help="argument for the actor's NLN weight initializer", type=float, required=True)
parser.add_argument("--actor-assume-false", help="if the actor should set a high weight to false initially", action="store_true")
parser.add_argument("--actor-optim", help="optimizer of the actor's parameters", choices=optimizers, required=True)
parser.add_argument("--actor-lr", help="learning rate for the actor's optimizer", type=float, required=True)
parser.add_argument("--critic-lr", help="learning rate for the critic", type=float, required=True)

log_dir = f"results/symbolic/exploring_starts/01_algorithms/nln_a2c_tab_{int(time.time())}"
device = "cuda" if torch.cuda.is_available() else "cpu"
torch.set_default_dtype(torch.float64)

gym.register(id="nesyrl-symbolic/BlocksWorld-v0", entry_point=BlocksWorld)

env_kwargs = {
    "horizon": 50,
    "blocks": ["a", "b", "c", "d"],
    "goal_state": [["a", "b", "c", "d"]]
}

if __name__ == "__main__":
    args = parser.parse_args()

    # Prepare log directory
    path = Path(log_dir)
    path.mkdir(parents=True, exist_ok=True)

    with open(f"{log_dir}/args.txt", "w") as out:
        print(args, file=out)

    for run in range(args.num_runs):
        # Prepare environments
        train_env = gym.make("nesyrl-symbolic/BlocksWorld-v0", **env_kwargs)
        test_env = gym.make("nesyrl-symbolic/BlocksWorld-v0", **env_kwargs)
        test_env.reset(seed=42)

        # Prepare agent
        actor = Actor(
            train_env.unwrapped,
            initializers[args.actor_init](args.actor_init_arg),
            args.actor_assume_false,
            device
        )
        critic = CriticTab(train_env.unwrapped, device)
        
        optim = ActorCriticOptimizer(
            optimizers[args.actor_optim](actor.parameters(), lr=args.actor_lr),
            torch.optim.SGD(critic.parameters(), lr=args.critic_lr)
        )

        policy = PGPolicy(
            actor=actor,
            optim=optim,
            dist_fn=torch.distributions.Categorical,
            action_scaling=False
        )

        # Prepare training
        train_collector = Collector(policy, DummyVectorEnv([lambda: train_env]))
        test_collector = SuccessCollector(policy, DummyVectorEnv([lambda: test_env]))

        logger = FileLogger(f"{log_dir}/run_{run}_log.txt")

        trainer = OnpolicyTrainer(
            policy=policy,
            batch_size=1,
            train_collector=train_collector,
            test_collector=test_collector,
            max_epoch=150,
            step_per_epoch=100,
            repeat_per_collect=1,
            episode_per_test=100,
            step_per_collect=1,
            logger=logger,
            test_fn=lambda ep, _: (ep % 10 == 0) and log_params(
                ep, f"{log_dir}/run_{run}_actor.txt", f"{log_dir}/run_{run}_critic.txt"
            )
        )

        # Train
        trainer.run()

        # Save models
        torch.save(actor, f"{log_dir}/run_{run}_actor.model")
        torch.save(optim.actor_optimizer, f"{log_dir}/run_{run}_actor.optim")
        torch.save(critic, f"{log_dir}/run_{run}_critic.model")
        torch.save(optim.critic_optimizer, f"{log_dir}/run_{run}_critic.optim")

        # Test
        policy.eval()
        test_collector.reset(seed=47)
        result = test_collector.collect(n_episode=1000, render=False)
        
        logger = FileLogger(f"run_{run}_test.txt")
        logger.log_test_data(result, 0)
