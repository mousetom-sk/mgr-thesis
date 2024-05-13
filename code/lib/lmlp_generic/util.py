# Neural Networks (2-AIN-132/15), FMFI UK BA
# (c) Tomas Kuzma, Juraj Holas, Peter Gergel, Endre Hamerlik, Stefan Pócoš, Iveta Bečková 2017-2024


import numpy as np
import numpy.typing as nptype


def add_bias(x: nptype.NDArray) -> nptype.NDArray:
    """
    Add bias term to vector, or to every (column) vector in a matrix.
    """

    if x.ndim == 1:
        return np.concatenate((x, [1]))
    else:
        pad = np.ones((1, x.shape[1]))
        return np.concatenate((x, pad), axis=0)

def expand(x: nptype.NDArray) -> nptype.NDArray:
        return np.atleast_2d(x.T).T
    
def squeeze(x: nptype.NDArray) -> nptype.NDArray:
    if x.shape[-1] == 1:
        return np.squeeze(x, axis=-1)
    
    return x
