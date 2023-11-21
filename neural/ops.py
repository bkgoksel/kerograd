import math
import numpy as np

def softmax(x: np.array, axis: int = 0) -> np.array:
    e_x = np.exp(x - np.max(x))
    return e_x / np.expand_dims(e_x.sum(axis=axis), axis=axis)

def layer_norm(x: np.array, eps: float=1e-10) -> np.array:
    mean = x.mean(axis=-1, keepdims=True)
    var = ((x - mean) ** 2).mean(axis=-1, keepdims=True)
    std = np.sqrt((var + eps))
    return (x - mean) / std
