from envs.symbolic import BlocksWorld
from agents.symbolic import *

from planning.blocks_world import BlocksWorldASPPlanner


horizon = 50
blocks = ["a", "b", "c", "d"]
goal = [["a", "b", "c", "d"]]
save_dir = "models/planning"


def plan_episode():
    print(f"Episode {ep+1}", end="\r")

    eval_env.reset()
    init = eval_env.get_raw_observation()
    planner = BlocksWorldASPPlanner(horizon, blocks[:], goal[:], init)
    plan = planner.solve()

    gamma = 0.99
    total_reward = 0
    I = 1
    done = False
    
    i = 0
    while not done:
        _, reward, terminated, truncated, info = eval_env.step(eval_env.action_atoms.index(plan[i]))
        done = terminated or truncated
        total_reward += I * reward
        I *= gamma
        i += 1

    return total_reward, info["is_goal"]


if __name__ == "__main__":
    eval_env = BlocksWorld(horizon, blocks[:], goal[:], None)
    eval_env.reset(seed=10)

    print("Planning")

    avg_returns = []
    avg_goals = []

    for batch in range(150):
        print(f"Batch {batch+1}  ")

        returns = []
        goals = []

        for ep in range(100):
            total_reward, is_goal = plan_episode()
            
            returns.append(total_reward)
            goals.append(is_goal)
        
        avg_returns.append(sum(returns) / len(returns))
        avg_goals.append(sum(goals) / len(goals))

    with open(f"{save_dir}/training_results.txt", "w") as out:
        print(", ".join(f"{r:.6f}" for r in avg_returns), file=out)
        print(", ".join(f"{r:.6f}" for r in avg_goals), file=out)

    print(20 * " ")
    print("Testing")
    
    returns = []
    goals = []

    for ep in range(100):
        total_reward, is_goal = plan_episode()
        
        returns.append(total_reward)
        goals.append(is_goal)

    print(sum(goals) / len(goals), 10 * " ")

    with open(f"{save_dir}/test_results.txt", "w") as out:
        print(", ".join(f"{r:.6f}" for r in returns), file=out)
        print(", ".join(f"{r:.6f}" for r in goals), file=out)

