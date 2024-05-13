import numpy as np

from envs import BlocksWorld
from agents.lmlp_agent_new import LMLPAgent


# def moving_average(x, w):
#     return np.convolve(x, np.ones(w), 'valid') / w

if __name__ == "__main__":
    training_env = BlocksWorld(["a", "b", "c", "d"],
                               [["a", "b", "c", "d"]],
                               [["a"], ["b"], ["c"], ["d"]])
    # validity:
    # training_env = BlocksWorld(["a", "b", "c"],
    #                            [["a", "b", "c"]])
    
    agent = LMLPAgent()

    rewards, goals = agent.train(training_env, 200)
    print(rewards)
    print(goals)
    
    # print()
    # ma = moving_average(rewards, 100)
    # print(ma)
