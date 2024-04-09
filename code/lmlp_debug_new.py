from __future__ import annotations

import numpy as np
import numpy.typing as nptype

from lib import Environment
from lib.lmlp_new import *
from envs import BlocksWorld


env = BlocksWorld(["a", "b", "c", "d"],
                  [["a", "b", "c", "d"]],
                  [["a"], ["b"], ["c"], ["d"]])

def vectorize_state(state: Environment.State) -> nptype.NDArray:
    return np.array(list(map(lambda f: state.features[f], env.feature_space)))

lmlp = LMLP(len(env.feature_space),
            [LMLPLayer(2, And(), UniformInitializer(0.25)),
             LMLPLayer(1, Or(), UniformInitializer(0.25), trainable=False)])


precons1 = [BlocksWorld.Top("a")]
precons2 = [BlocksWorld.Top("b")]
free_atoms = set(env.feature_space) - set(precons1) - set(precons2)

states = []
targets = []

for _ in range(1000):
    s = env._generate_random_state()
    states.append(s)
    targets.append([int(all(s.features[p] > 0 for p in precons1) or all(s.features[p] > 0 for p in precons2))])

targets = np.array(targets)
print(np.sum(targets))

np.set_printoptions(formatter={'float': lambda x: "{0:0.5f}".format(x)})

optimizer = Adam(-0.01)
optimizer.prepare(lmlp)

for ep in range(30):
    error = 0

    for idx in np.random.permutation(len(states)):
        s = states[idx]
        t = targets[idx]

        o = lmlp.forward(vectorize_state(s))
        optimizer.optimize(-(t - o[-1]), o)

        error += np.sum(((t - o[-1])**2)/2)

    print(ep, error)

print(lmlp)
