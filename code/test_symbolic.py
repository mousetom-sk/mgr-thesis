import matplotlib.pyplot as plt
import numpy as np

from envs.symbolic import BlocksWorld
from agents.symbolic import *

from planning.blocks_world import BlocksWorldASPPlanner


horizon = 50
blocks = ["a", "b", "c", "d"]
goal = [["a", "b", "c", "d"]]


if __name__ == "__main__":
    env = BlocksWorld(horizon, blocks[:], goal[:], None) #[["a"], ["b"], ["c"], ["d"]])
    env.reset(seed=5)

    agent = LMLPA2C()

    print("Training")
    rewards, goals, inits = agent.train(env, 10000)

    print()
    print("Planning")

    best_rewards = []

    for ep, init in enumerate(inits, 1):
        print(f"Episode {ep}", end="\r")

        planner = BlocksWorldASPPlanner(horizon, blocks[:], goal[:], init)
        plan = planner.solve()

        gamma = 0.99
        total_reward = 0
        I = 1
        done = False
        env.reset(options={"initial_state": init})
        
        i = 0
        while not done:
            _, reward, terminated, truncated, _ = env.step(env.action_atoms.index(plan[i]))
            done = terminated or truncated
            total_reward += I * reward
            I *= gamma
            i += 1
        
        best_rewards.append(total_reward)

    print()
    print(rewards)
    print(best_rewards)

    rewards = np.array(rewards)
    best_rewards = np.array(best_rewards)
    # ratios = rewards / best_rewards
    episodes = np.arange(1, len(inits) + 1)

    fig, ax = plt.subplots()

    ax.plot(episodes, rewards, label="Ours")
    ax.plot(episodes, best_rewards, label="Symbolic Planning")
    ax.legend()

    plt.show()


    # print("Testing")
    # rewards, goals, inits = agent.evaluate(training_env, 1000)
    # print(sum(goals) / len(goals))
