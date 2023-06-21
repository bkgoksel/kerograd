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
    if len(wrt_param.shape) == 1 or wrt_param.shape[-1] == 1:
        # We're taking a derivative w.r.t a vector: dAx/dx = A
        return rolling_partial.dot(other_param)
    elif len(other_param.shape) == 1 or other_param.shape[-1] == 1:
        # We're taking a derivative w.r.t a matrix: dAx/dA = a 3d matrix that has x^T on its 2d diagonal
        diagonal_shape = [wrt_param.shape[0], wrt_param.shape[0], 1]
        x_t = other_param
        grad_mul = 'i,imo->io'
        if len(other_param.shape) > 1:
            diagonal_shape = [1] + diagonal_shape
            x_t = other_param.reshape((other_param.shape[0], 1, 1, other_param.shape[-2]))
            grad_mul = 'bi,bimo->bio'
        diagonal = np.diagflat(np.ones(wrt_param.shape[0])).reshape(diagonal_shape)
        grad = diagonal * x_t
        return np.einsum(grad_mul, rolling_partial, grad)
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

