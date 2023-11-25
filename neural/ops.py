import math
import numpy as np

def softmax(x: np.array, axis: int = 0) -> np.array:
    e_x = np.exp(x - np.max(x))
    return e_x / np.expand_dims(np.sum(e_x, axis=axis), axis=axis)

def layer_norm(x: np.array, eps: float=1e-10) -> np.array:
    mean = np.mean(x, axis=-1, keepdims=True)
    var = np.mean(((x - mean) ** 2), axis=-1, keepdims=True)
    std = np.sqrt((var + eps))
    norm = x - mean
    div_norm = norm / std
    return div_norm
