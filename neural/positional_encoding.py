import numpy as np

from .base import NamedOp

def _get_angles(pos, i, dim):
    angle_rates = 1 / np.power(10000, (2 * (i//2)) / np.float32(dim))
    return pos * angle_rates

def get_positional_encodings(max_pos: int, dim: int) -> np.array:
    angle_rads = _get_angles(np.arange(max_pos)[:, np.newaxis],
                             np.arange(dim)[np.newaxis, :],
                             dim)
    angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])
    angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])

    return angle_rads
    #return angle_rads[np.newaxis, ...]
