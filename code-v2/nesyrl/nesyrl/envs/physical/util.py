from typing import Iterable
from numpy.typing import NDArray

import numpy as np


def squared_length(vector: NDArray | Iterable[int | float]) -> float:
    return sum(np.array(vector) ** 2)

def length(vector: NDArray | Iterable[int | float]) -> float:
    return np.sqrt(squared_length(vector))
