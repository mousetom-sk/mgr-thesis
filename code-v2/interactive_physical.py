from nesyrl.envs.physical import NicoBlocksWorldMove


env = NicoBlocksWorldMove(10000, ["a", "b", "c", "d"], 1)

while True:
    env.reset()
    done = False

    while not done:
        obs, _, ter, trun, _ = env.step_human()
        
        done = ter or trun

    input("cont")
