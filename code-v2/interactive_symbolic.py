from nesyrl.envs.symbolic import BlocksWorld


if __name__ == "__main__":
    env = BlocksWorld(50, ["a", "b", "c", "d"],
                      [["a", "b", "c", "d"]],
                      [["a"], ["b"], ["c"], ["d"]], reward_subgoals=True)
    
    print("All actions:")
    print([(i, str(a)) for i, a in enumerate(env.action_atoms)])

    state, _ = env.reset()
    total_reward = 0
    done = False
    
    while not done:
        print("Current state:", list(map(str, filter(lambda a: state[a], state))))
        action_idx = int(input("Choose action: "))
        state, reward, ter, trun, _ = env.step(action_idx)

        print("Reward:", reward)
        total_reward += reward
        done = ter or trun

        print(env._reached_subgoals)

    print("End. Total reward:", total_reward)
