from nesyrl.envs.physical.blocks_world_ned_mag import NedBlocksWorldMove


env = NedBlocksWorldMove(10000, ["a", "b", "c", "d"], 1)

while True:
    env.reset()
    done = False

    while not done:
        obs, _, ter, trun, _ = env.step_human()
        
        done = ter or trun

    input("cont")
