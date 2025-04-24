import time
import argparse
import json
from pathlib import Path

import gymnasium as gym

import torch

from tianshou.data import Collector, ReplayBuffer
from tianshou.env import DummyVectorEnv
from tianshou.trainer import OnpolicyTrainer

from nesyrl.envs.symbolic import BlocksWorld
from nesyrl.agents.symbolic import Actor, CriticTab, ActorCriticOptimizer, PPOPolicy
from nesyrl.logic.neural import ConstantInitializer, UniformInitializer, NLAndBiProd, NLAndBiLuka, UncertaintyRegularizer, NoRegularizer
from nesyrl.util.logging import FileLogger
from nesyrl.util.collecting import SuccessCollector


def save_run(
    run: int, ep: int | None, actor: Actor, critic: CriticTab, optim: ActorCriticOptimizer
) -> None:
    ep_str = f"_ep_{ep}" if ep is not None else ""

    torch.save(actor, f"{log_dir}/run_{run}{ep_str}_actor.model")
    torch.save(critic, f"{log_dir}/run_{run}{ep_str}_critic.model")
    torch.save(optim, f"{log_dir}/run_{run}{ep_str}.optim")

    with open(f"{log_dir}/run_{run}{ep_str}_actor.weights", "w") as out:
        print(actor.params_str(), file=out)

    with open(f"{log_dir}/run_{run}{ep_str}_critic.weights", "w") as out:
        print(critic.params_str(), file=out)

def next_epoch(
    ep: int, run: int, actor: Actor, critic: CriticTab, optim: ActorCriticOptimizer
) -> None:
    if ep % 10 == 0:
        save_run(run, ep, actor, critic, optim)


# Basic configuration
initializers = {
    "constant": ConstantInitializer,
    "uniform": UniformInitializer
}

ands = {
    "biprod": NLAndBiProd,
    "biluka": NLAndBiLuka
}

parser = argparse.ArgumentParser()
parser.add_argument("--num-runs", help="number of times to run the experiment", type=int, choices=range(1, 11), required=True)
parser.add_argument("--actor-init", help="actor's NLN weight initializer", choices=initializers, required=True)
parser.add_argument("--actor-init-arg", help="argument for the actor's NLN weight initializer", type=float, required=True)
parser.add_argument("--actor-and", help="the implementation of conjuction to use", choices=ands, required=True)
parser.add_argument("--actor-reg-strength", help="strength of actor's regularizer", type=float, default=None)
parser.add_argument("--actor-reg-peak", help="peak of actor's regularizer", type=float, default=None)
parser.add_argument("--actor-reg-constant", help="whether actor's regularizer should be constant after peak", action="store_true")
parser.add_argument("--actor-lr", help="learning rate for the actor's optimizer", type=float, required=True)
parser.add_argument("--critic-lr", help="learning rate for the critic", type=float, required=True)

log_dir = f"results/symbolic/02_regularization/nln_ppo_tab_{time.strftime('%d%m%Y_%H%M%S', time.gmtime(time.time()))}"
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
        "max_batchsize": 1,
        "advantage_normalization": False
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
        actor = Actor(
            train_env.unwrapped,
            ands[args["actor_and"]],
            initializers[args["actor_init"]](args["actor_init_arg"]),
            and_regularizer=UncertaintyRegularizer(args["actor_reg_strength"], args["actor_reg_peak"], args["actor_reg_constant"])
                            if args["actor_reg_strength"] else NoRegularizer(),
            drop_false=True,
            device=device
        )
        critic = CriticTab(train_env.unwrapped, device)
        
        optim = ActorCriticOptimizer(
            torch.optim.Adam(actor.parameters(), lr=args["actor_lr"]),
            torch.optim.SGD(critic.parameters(), lr=args["critic_lr"])
        )

        policy = PPOPolicy(
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
