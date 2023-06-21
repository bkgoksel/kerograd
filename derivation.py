import numpy as np
import math

DIFFERENTIABLE_FUNCTIONS = {}
    
def differentiates(numpy_function):
    """Register an __array_function__ implementation for MyArray objects."""
    def decorator(func):
        DIFFERENTIABLE_FUNCTIONS[numpy_function] = func
        return func
    return decorator

@differentiates(np.add)
def add(inputs: list[any], differentiate_wrt: str, rolling_partial: np.ndarray) -> np.ndarray:
    dims = [arg for arg in inputs if hasattr(arg, 'name') and arg.name==differentiate_wrt][0].shape
    return rolling_partial * (np.ones(dims))

@differentiates(np.subtract)
def subtract(inputs: list[any], differentiate_wrt: str, rolling_partial: np.ndarray) -> np.ndarray:
    for i, param in enumerate(inputs):
        if hasattr(param, 'name') and param.name == differentiate_wrt:
            res = np.ones(param.shape)
            return rolling_partial * ((res if i == 0 else -1 * res))

@differentiates(np.multiply)
def multiply(inputs: list[any], differentiate_wrt: str, rolling_partial: np.ndarray) -> np.ndarray:
    # multiply is element-wise, return the other input
    for i, param in enumerate(inputs):
        if hasattr(param, 'name') and param.name == differentiate_wrt:
            return rolling_partial * (inputs[(i-1)**2])

@differentiates(np.matmul)
def matmul(inputs: list[any], differentiate_wrt: str, rolling_partial: np.ndarray) -> np.ndarray:
    for i, param in enumerate(inputs):
        if hasattr(param, 'name') and param.name == differentiate_wrt:
            wrt_param, other_param = param, inputs[(i-1)**2]
            break
    if len(wrt_param.shape) == 1:
        return rolling_partial.dot(other_param)
    elif len(other_param.shape) == 1:
        return rolling_partial.dot(np.diagflat(np.ones(wrt_param.shape[0])).reshape((wrt_param.shape[0], wrt_param.shape[0], 1)).dot(other_param.reshape(1,-1)))
    else:
        raise NotImplementedError("matrix-matrix derivatives not yet implemented")

@differentiates(np.mean)
def mean(inputs: list[any], differentiate_wrt: str, rolling_partial: np.ndarray) -> np.ndarray:
    dims = [arg for arg in inputs if hasattr(arg, 'name') and arg.name==differentiate_wrt][0].shape
    return rolling_partial * (np.ones(dims))

@differentiates(np.transpose)
def transpose(inputs: list[any], differentiate_wrt: str, rolling_partial: np.ndarray) -> np.ndarray:
    return rolling_partial.transpose()

@differentiates(np.exp)
def exp(inputs: list[any], differentiate_wrt: str, rolling_partial: np.ndarray) -> np.ndarray:
    return rolling_partial * (inputs[0].exp(inputs[1]-1))

@differentiates(np.square)
def square(inputs: list[any], differentiate_wrt: str, rolling_partial: np.ndarray) -> np.ndarray:
    return rolling_partial * (2* inputs[0])

@differentiates(np.maximum)
def maximum(inputs: list[any], differentiate_wrt: str, rolling_partial: np.ndarray) -> np.ndarray:
    if len(inputs) >= 2 and np.isscalar(inputs[1]):
        return rolling_partial * ((inputs[0] >= inputs[1]))
    else:
        raise NotImplementedError("maximum can only be differentiated between an array and scalar at the moment")

