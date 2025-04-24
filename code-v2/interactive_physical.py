from nesyrl.envs.physical.blocks_world_mag import NicoBlocksWorldMove
# import numpy as np

env = NicoBlocksWorldMove(10000, ["a", "b", "c", "d"], 1)

while True:
    env.reset(seed=47)
    done = False

    while not done:
        obs, _, ter, trun, _ = env.step_human()
        # orn = np.array(obs[3:6])
        # orn /= np.linalg.norm(orn)
        # ref = np.array([-np.sqrt(2) / 2, 0, np.sqrt(2) / 2])
        # dist = np.linalg.norm(orn - ref)
        # dist = min(dist, np.linalg.norm(orn + ref))

        # print(dist)
        # print(orn, abs(orn @ ref))
        
        done = ter or trun

    input("cont")
