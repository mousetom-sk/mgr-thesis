from envs.physical import SimulatedNicoBlocksWorld
from agents.physical import *


if __name__ == "__main__":
    training_env = SimulatedNicoBlocksWorld(10000,
                                            ["a", "b", "c", "d"],
                                            [["a", "b", "c", "d"]],
                                            [["a"], ["b"], ["c"], ["d"]])
    
    training_env.reset(seed=5)
    agent = ActorCritic()

    print("Training")
    rewards, goals = agent.train(training_env, 5000)

    print("Testing")
    rewards, goals = agent.evaluate(training_env, 1000)
    print(sum(goals) / len(goals))
