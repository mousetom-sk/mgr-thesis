import os

from envs.symbolic import BlocksWorld
from agents.symbolic import *


horizon = 50
blocks = ["a", "b", "c", "d"]
goal = [["a", "b", "c", "d"]]
save_dir = "models/lmlp_a2c_reg2_rnd_xor_test"


if __name__ == "__main__":
    try:
        os.mkdir(save_dir)
    except:
        pass

    training_env = BlocksWorld(horizon, blocks[:], goal[:], None)
    training_env.reset(seed=5)

    eval_env = BlocksWorld(horizon, blocks[:], goal[:], None)
    eval_env.reset(seed=10)

    agent = LMLPA2C()

    print("Training")
    returns, goals = agent.train_eval(
        training_env, 15000, eval_env, 100, 100000, save_dir
    )

    with open(f"{save_dir}/training_results.txt", "w") as out:
        print(", ".join(f"{r:.6f}" for r in returns), file=out)
        print(", ".join(f"{r:.6f}" for r in goals), file=out)

    print()
    print("Testing")
    returns, goals = agent.evaluate(eval_env, 1000)
    print(sum(goals) / len(goals))

    with open(f"{save_dir}/test_results.txt", "w") as out:
        print(", ".join(f"{r:.6f}" for r in returns), file=out)
        print(", ".join(f"{r:.6f}" for r in goals), file=out)
