from envs import BlocksWorld
from agents.lmlp_agent_new import LMLPAgent
from agents.torch_agent import TorchAgent

import numpy as np

# import numpy as np

# i = np.array([1, 2, 3, 4])
# m = np.array([[1, 2, 3],
#               [1, 2, 3],
#               [1, 2, 3],
#               [1, 2, 3]])
# n = np.apply_along_axis(lambda c: np.outer(c, i), 0, m)

# print(n[:, :, 1])


def moving_average(x, w):
    return np.convolve(x, np.ones(w), 'valid') / w

if __name__ == "__main__":
    training_env = BlocksWorld(["a", "b", "c", "d", "e"],
                               [["a", "b", "c", "d", "e"]],
                               [["a"], ["b"], ["c"], ["d"], ["e"]])
    # training_env = BlocksWorld(["a", "b", "c"],
    #                            [["a", "b", "c"]],
    #                            [["a"], ["b"], ["c"]])
    agent = LMLPAgent()
    # agent = TorchAgent()

    rewards, goals = agent.train(training_env, 400)
    print(rewards)
    print(goals)
    print()

    # ma = moving_average(rewards, 100)
    # print(ma)
