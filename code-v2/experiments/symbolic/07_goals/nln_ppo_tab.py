import time
import argparse
import json
from pathlib import Path

import gymnasium as gym

import torch

from tianshou.data import Collector, ReplayBuffer
from tianshou.env import DummyVectorEnv
from tianshou.trainer import OnpolicyTrainer

from nesyrl.envs.symbolic import BlocksWorldMultiGoal
from nesyrl.agents.symbolic import ActorMulti, CriticTab, ActorCriticOptimizer, PPOPolicy
from nesyrl.logic.neural import ConstantInitializer, UniformInitializer, NLOrClamped, NLOrLukaClamped, NLXorClamped, NLXorLukaClamped, NLAndBiLuka, NLAndBiProd
from nesyrl.util.logging import FileLogger
from nesyrl.util.collecting import SuccessCollector


def save_run(
    run: int, ep: int | None, actor: ActorMulti, critic: CriticTab, optim: ActorCriticOptimizer
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
    ep: int, run: int, actor: ActorMulti, critic: CriticTab, optim: ActorCriticOptimizer
) -> None:
    if ep % 10 == 0:
        save_run(run, ep, actor, critic, optim)


# Basic configuration
initializers = {
    "constant": ConstantInitializer,
    "uniform": UniformInitializer
}

ors = {
    "or": (NLOrClamped, NLAndBiProd),
    "orluka": (NLOrLukaClamped, NLAndBiLuka),
    "xor": (NLXorClamped, NLAndBiProd),
    "xorluka": (NLXorLukaClamped, NLAndBiLuka)
}

all_goals = [[["a", "b", "c", "d"]],
             [["d", "c", "b", "a"]]]

parser = argparse.ArgumentParser()
parser.add_argument("--num-runs", help="number of times to run the experiment", type=int, choices=range(1, 11), required=True)
parser.add_argument("--goals", help="which goals to learn", nargs="+", type=int, choices=range(len(all_goals)), required=True)
parser.add_argument("--goals-switch", help="how often to switch goals", type=int, default=1)
parser.add_argument("--reward-subgoals", help="whether to reward the agent for sub-stacks", action="store_true")
parser.add_argument("--actor-init", help="actor's NLN weight initializer", choices=initializers, required=True)
parser.add_argument("--actor-init-arg", help="argument for the actor's NLN weight initializer", type=float, required=True)
parser.add_argument("--actor-num-ands", help="number of rules for each action", type=int, required=True)
parser.add_argument("--actor-or", help="disjunction implementation", type=str, choices=ors, required=True)
parser.add_argument("--actor-validity", help="whether to inject prior knwoledge on action validity", action="store_true")
parser.add_argument("--actor-lr", help="learning rate for the actor's optimizer", type=float, required=True)
parser.add_argument("--critic-lr", help="learning rate for the critic", type=float, required=True)

log_dir = f"results/symbolic/07_goals/nln_ppo_tab_{time.strftime('%d%m%Y_%H%M%S', time.gmtime(time.time()))}"
device = "cuda" if torch.cuda.is_available() else "cpu"
torch.set_default_dtype(torch.float32)

gym.register(id="nesyrl-symbolic/BlocksWorldMultiGoal-v0", entry_point=BlocksWorldMultiGoal)

args = {
    "train_env_kwargs":  {
        "horizon": 50,
        "blocks": ["a", "b", "c", "d"]
    },
    "test_env_kwargs":  {
        "horizon": 50,
        "blocks": ["a", "b", "c", "d"]
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
        "max_epoch": 600,
        "step_per_epoch": 100,
        "repeat_per_collect": 1,
        "episode_per_test": 100,
        "step_per_collect": 1,
        "batch_size": 1
    }
}


if __name__ == "__main__":
    args |= vars(parser.parse_args())
    args["train_env_kwargs"]["goal_states"] = [g for i, g in enumerate(all_goals) if i in args["goals"]]
    args["test_env_kwargs"]["goal_states"] = [g for i, g in enumerate(all_goals) if i in args["goals"]]
    args["train_env_kwargs"]["switch_period"] = args["goals_switch"]
    args["test_env_kwargs"]["switch_period"] = args["goals_switch"]
    args["train_env_kwargs"]["reward_subgoals"] = args["reward_subgoals"]
    args["test_env_kwargs"]["reward_subgoals"] = args["reward_subgoals"]

    # Prepare log directory
    path = Path(log_dir)
    path.mkdir(parents=True, exist_ok=True)

    with open(f"{log_dir}/args.txt", "w") as out:
        json.dump(args, out, indent=4)

    for run in range(args["num_runs"]):
        # Prepare environments
        train_env = gym.make("nesyrl-symbolic/BlocksWorldMultiGoal-v0", **args["train_env_kwargs"])
        test_env = gym.make("nesyrl-symbolic/BlocksWorldMultiGoal-v0", **args["test_env_kwargs"])
        test_env.reset(seed=args["test_train_seed"])

        train_venv = DummyVectorEnv([lambda: train_env])
        test_venv = DummyVectorEnv([lambda: test_env])

        # Prepare agent
        actor = ActorMulti(
            train_env.unwrapped,
            ors[args["actor_or"]][1],
            initializers[args["actor_init"]](args["actor_init_arg"]),
            args["actor_num_ands"], ors[args["actor_or"]][0],
            inject_validity=args["actor_validity"],
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
