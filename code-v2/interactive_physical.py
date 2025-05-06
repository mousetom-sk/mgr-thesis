from nesyrl.envs.physical.blocks_world_mag import NicoBlocksWorldMove
# import numpy as np

env = NicoBlocksWorldMove(10000, ["a", "b", "c", "d"], 1)

while True:
    env.reset(seed=322)
    done = False

    log = env._pbc.startStateLogging(loggingType=env._pbc.STATE_LOGGING_VIDEO_MP4, fileName="test.mp4")

    i = 0
    while not done:
        obs, _, ter, trun, _ = env.step_human()
        # orn = np.array(obs[3:6])
        # orn /= np.linalg.norm(orn)
        # ref = np.array([-np.sqrt(2) / 2, 0, np.sqrt(2) / 2])
        # dist = np.linalg.norm(orn - ref)
        # dist = min(dist, np.linalg.norm(orn + ref))

        # print(dist)
        # print(orn, abs(orn @ ref))

        if i == 100:
            env._pbc.stopStateLogging(log)
            break

        done = ter or trun
        i += 1

    input("cont")
