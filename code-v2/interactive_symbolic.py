from envs.symbolic import BlocksWorld


if __name__ == "__main__":
    training_env = BlocksWorld(["a", "b", "c", "d"],
                               [["a", "b", "c", "d"]],
                               [["a"], ["b"], ["c"], ["d"]])
    
    print("All actions:")
    print(list(zip(range(len(training_env.action_space)), map(str, training_env.action_space))))

    state = training_env.reset()
    total_reward = 0
    
    while not training_env.is_final():
        print("Current state:", list(map(str, filter(lambda f: state.features[f], state.features))))
        action_idx = int(input("Choose action: "))
        action = training_env.action_space[action_idx]
        state, reward = training_env.step(action)

        print("Reward:", reward)
        total_reward += reward

    print("End. Total reward:", total_reward)
