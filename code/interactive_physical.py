from envs.physical import SimulatedNicoBlocksWorld


env = SimulatedNicoBlocksWorld(10000,
                               ["a", "b", "c", "d"],
                               [["a", "b", "c", "d"]],
                               [["a"], ["b"], ["c", "d"]])

while True:
    env.reset()
    done = False

    while not done:
        obs, _, ter, trun, _ = env.step_human()
        
        done = ter or trun
