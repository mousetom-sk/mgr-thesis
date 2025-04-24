import time
import argparse
import json
from pathlib import Path

import gymnasium as gym

import torch
import numpy as np

from tianshou.data import Collector, ReplayBuffer
from tianshou.env import DummyVectorEnv
from tianshou.policy import A2CPolicy
from tianshou.trainer import OnpolicyTrainer
from tianshou.utils.net.common import Net

from nesyrl.envs.symbolic import BlocksWorld
from nesyrl.agents.symbolic import ActorMLP, CriticTab, ActorCriticOptimizer
from nesyrl.util.logging import FileLogger
from nesyrl.util.collecting import SuccessCollector


def init_weights(module: torch.nn.Module) -> None:
    if isinstance(module, torch.nn.Linear):
        torch.nn.init.orthogonal_(module.weight, gain=np.sqrt(2))
        torch.nn.init.zeros_(module.bias)

def save_run(
    run: int, ep: int | None, actor: ActorMLP, critic: CriticTab, optim: ActorCriticOptimizer
) -> None:
    ep_str = f"_ep_{ep}" if ep is not None else ""

    torch.save(actor, f"{log_dir}/run_{run}{ep_str}_actor.model")
    torch.save(critic, f"{log_dir}/run_{run}{ep_str}_critic.model")
    torch.save(optim, f"{log_dir}/run_{run}{ep_str}.optim")

def next_epoch(
    ep: int, run: int, actor: ActorMLP, critic: CriticTab, optim: ActorCriticOptimizer
) -> None:
    if ep % 10 == 0:
        save_run(run, ep, actor, critic, optim)


# Basic configuration
optimizers = {
    "rms": torch.optim.RMSprop,
    "adam": torch.optim.Adam
}

parser = argparse.ArgumentParser()
parser.add_argument("--num-runs", help="number of times to run the experiment", type=int, choices=range(1, 11), required=True)
parser.add_argument("--actor-hidden", help="sizes of hidden layers in the actor net", nargs='+', type=int, required=True)
parser.add_argument("--actor-optim", help="optimizer of the actor's parameters", choices=optimizers, required=True)
parser.add_argument("--actor-lr", help="learning rate for the actor's optimizer", type=float, required=True)
parser.add_argument("--critic-lr", help="learning rate for the critic", type=float, required=True)

log_dir = f"results/symbolic/01_algorithms/mlp_a2c_tab_{time.strftime('%d%m%Y_%H%M%S', time.gmtime(time.time()))}"
device = "cuda" if torch.cuda.is_available() else "cpu"
torch.set_default_dtype(torch.float32)

gym.register(id="nesyrl-symbolic/BlocksWorld-v0", entry_point=BlocksWorld)

args = {
    "train_env_kwargs":  {
        "horizon": 50,
        "blocks": ["a", "b", "c", "d"],
        "goal_state": [["a", "b", "c", "d"]]
    },
    "test_env_kwargs":  {
        "horizon": 50,
        "blocks": ["a", "b", "c", "d"],
        "goal_state": [["a", "b", "c", "d"]]
    },
    "test_train_seed": 42,
    "test_test_seed": 47,
    "test_episides": 1000,
    "policy" : {
        "gae_lambda": 0.9,
        "ent_coef": 0,
        "max_batchsize": 1
    },
    "trainer": {
        "max_epoch": 300,
        "step_per_epoch": 100,
        "repeat_per_collect": 1,
        "episode_per_test": 100,
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

    for run in range(args["num_runs"]):
        # Prepare environments
        train_env = gym.make("nesyrl-symbolic/BlocksWorld-v0", **args["train_env_kwargs"])
        test_env = gym.make("nesyrl-symbolic/BlocksWorld-v0", **args["test_env_kwargs"])
        test_env.reset(seed=args["test_train_seed"])

        train_venv = DummyVectorEnv([lambda: train_env])
        test_venv = DummyVectorEnv([lambda: test_env])

        # Prepare agent
        net = Net(state_shape=len(train_env.observation_space.spaces),
                  action_shape=train_env.action_space.n,
                  hidden_sizes=args["actor_hidden"],
                  activation=torch.nn.Tanh, softmax=True, device=device).to(device)
        actor = ActorMLP(net)
        critic = CriticTab(train_env.unwrapped, device)

        net.apply(init_weights)
        
        optim = ActorCriticOptimizer(
            optimizers[args["actor_optim"]](actor.parameters(), lr=args["actor_lr"]),
            torch.optim.SGD(critic.parameters(), lr=args["critic_lr"])
        )

        policy = A2CPolicy(
            actor=actor,
            critic=critic,
            optim=optim,
            dist_fn=torch.distributions.Categorical,
            action_scaling=False,
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
            test_fn=lambda ep, _: next_epoch(ep, run, actor, critic, optim),
            **args["trainer"]
        )

        # Train
        trainer.run()

        # Save models
        save_run(run, None, actor, critic, optim)

        # Test
        policy.eval()
        test_collector.reset(gym_reset_kwargs={"seed": args["test_test_seed"]})
        result = test_collector.collect(n_episode=args["test_episides"], render=False)
        
        logger = FileLogger(f"{log_dir}/test.txt")
        logger.log_test_data(result, 0)
