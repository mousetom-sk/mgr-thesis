from envs.physical import SimulatedNicoBlocksWorld
from envs.hierarchical import OptionsNicoBlocksWorld
from agents.hierarchical import *


if __name__ == "__main__":
    training_env = OptionsNicoBlocksWorld(SimulatedNicoBlocksWorld(
        200,
        ["a", "b", "c", "d"],
        [["a", "b", "c", "d"]],
        [["a"], ["b"], ["c"], ["d"]]
    ))
    
    training_env.reset(seed=5)
    agent = A2CNCE()

    # print("Training")
    agent.setup(training_env)
    # agent.load(
    #     {"reach_grasp": "e.model"},
    #     {"reach_grasp": "p.model"},
    #     {"reach_grasp": "v.model"},
    #     {"reach_grasp": "e.opt"},
    #     {"reach_grasp": "p.opt"},
    #     {"reach_grasp": "v.opt"}
    # )
    rewards, goals = agent.train(training_env, 20, 10000, 100)

    # print("Testing")
    # rewards, goals = agent.evaluate(training_env, 1000)
    # print(sum(goals) / len(goals))
