from typing import List, Tuple, Any, Iterable
from numpy.typing import NDArray

from enum import Enum

import numpy as np


# class SignedComponent(Enum):
#     X_POS = [1, 0, 0]
#     X_NEG = [-1, 0, 0]
#     Y_POS = [0, 1, 0]
#     Y_NEG = [0, -1, 0]
#     Z_POS = [0, 0, 1]
#     Z_NEG = [0, 0, -1]


# def to_ndarrays(*arrays: List[Any] | Tuple[Any]) -> Tuple[NDArray]:
#     return tuple(np.array(a) for a in arrays)

def squared_length(vector: NDArray | Iterable[int | float]) -> float:
    return sum(np.array(vector) ** 2)

def length(vector: NDArray | Iterable[int | float]) -> float:
    return np.sqrt(squared_length(vector))

# def is_main_component(vector: NDArray, component: SignedComponent) -> float:
#     proj = vector * np.array(component.value)

#     return squared_length(proj) > 0.98 * squared_length(vector)
