import gymnasium as gym

import torch

from tianshou.data import Collector
from tianshou.policy import A2CPolicy
from tianshou.trainer import OnpolicyTrainer

from envs.symbolic import BlocksWorld
from agents.symbolic import Actor, Critic, CriticTab, ActorCriticOptimizer
from util.logging import FileLogger
from util.collecting import SuccessCollector


device = "cuda" if torch.cuda.is_available() else "cpu"
torch.set_default_dtype(torch.float64)


# Prepare environments
gym.register(
    id="nesyrl-symbolic/BlocksWorld-v0",
    entry_point=BlocksWorld,
)

env_kwargs = {
    "horizon": 50,
    "blocks": ["a", "b", "c", "d"],
    "goal_state": [["a", "b", "c", "d"]]
}

train_env = gym.make("nesyrl-symbolic/BlocksWorld-v0", **env_kwargs)
test_env = gym.make("nesyrl-symbolic/BlocksWorld-v0", **env_kwargs)


# Prepare policy
actor = Actor(train_env.unwrapped, device)
# critic = Critic(test_env.unwrapped, 20, device)
critic = CriticTab(train_env.unwrapped, device)
optim = ActorCriticOptimizer(
    torch.optim.RMSprop(actor.parameters(), lr=1e-2),
    torch.optim.SGD(critic.parameters(), lr=5e-2)
)

dist = torch.distributions.Categorical
policy = A2CPolicy(
    actor=actor,
    critic=critic,
    optim=optim,
    dist_fn=dist,
    action_scaling=False,
    gae_lambda=1,
    ent_coef=0,
    max_batchsize=1
)


# Prepare training
train_collector = Collector(policy, train_env)
test_collector = SuccessCollector(policy, test_env)

logger = FileLogger("training.txt")

trainer = OnpolicyTrainer(
    policy=policy,
    batch_size=1, #256
    train_collector=train_collector,
    test_collector=test_collector,
    max_epoch=150,
    step_per_epoch=100,
    repeat_per_collect=1, #10
    episode_per_test=100,
    step_per_collect=1,
    logger=logger
)


# Train
train_result = trainer.run()

with open("actor_weights.txt", "w") as out:
    print(actor.params_str(), file=out)

with open("critic_weights.txt", "w") as out:
    print(critic.params_str(), file=out)
