import numpy as np
import math

DIFFERENTIABLE_FUNCTIONS = {}


def differentiates(numpy_function):
    """Register an __array_function__ implementation for MyArray objects."""

    def decorator(func):
        DIFFERENTIABLE_FUNCTIONS[numpy_function] = func
        return func

    return decorator


@differentiates(np.concatenate)
def concatenate(
    inputs: list[any], differentiate_wrt: str, rolling_partial: np.ndarray
) -> np.ndarray:
    input_arrays, axis = inputs
    # concatenation, grads are 1 where the differentiate_wrt is and 0 elsewhere
    grad_arrays = []
    for i, param in enumerate(input_arrays):
        if hasattr(param, "name") and param.name == differentiate_wrt:
            grad_arrays.append(rolling_partial)
        else:
            grad_arrays.append(np.zeros(param.shape))
    return np.concatenate(grad_arrays, axis=axis)


@differentiates(np.add)
def add(
    inputs: list[any], differentiate_wrt: str, rolling_partial: np.ndarray
) -> np.ndarray:
    dims = [
        arg for arg in inputs if hasattr(arg, "name") and arg.name == differentiate_wrt
    ][0].shape
    return rolling_partial * (np.ones(dims))


@differentiates(np.subtract)
def subtract(
    inputs: list[any], differentiate_wrt: str, rolling_partial: np.ndarray
) -> np.ndarray:
    for i, param in enumerate(inputs):
        if hasattr(param, "name") and param.name == differentiate_wrt:
            res = np.ones(param.shape)
            return rolling_partial * ((res if i == 0 else -1 * res))


@differentiates(np.multiply)
def multiply(
    inputs: list[any], differentiate_wrt: str, rolling_partial: np.ndarray
) -> np.ndarray:
    # multiply is element-wise, return the other input
    for i, param in enumerate(inputs):
        if hasattr(param, "name") and param.name == differentiate_wrt:
            return rolling_partial * (inputs[(i - 1) ** 2])


@differentiates(np.divide)
def divide(
    inputs: list[any], differentiate_wrt: str, rolling_partial: np.ndarray
) -> np.ndarray:
    # divide is element-wise, return the negative of other input
    for i, param in enumerate(inputs):
        if hasattr(param, "name") and param.name == differentiate_wrt:
            if i == 0:
                return (rolling_partial * inputs[(i - 1) ** 2])
            else:
                return -1 * (rolling_partial * inputs[(i - 1) ** 2])



@differentiates(np.einsum)
def einsum(
    inputs: list[any], differentiate_wrt: str, rolling_partial: np.ndarray
) -> np.ndarray:
    # TODO: parse einsum string, figure out what the derivation is
    #   - Ellipses/broadcasting will likely be annoying
    summation, arrays = inputs[0], inputs[1:]
    if "..." in summation:
        raise NotImplementedError("brodcasting in einsum not supported yet")
    if "->" not in summation:
        raise NotImplementedError(
            "only explicit mode einsum differentiation is supported"
        )
    sum_in, sum_out = summation.split("->")
    sum_terms = sum_in.split(",")
    sum_dimensionality = {}
    for param, sum_term in zip(arrays, sum_terms):
        assert len(sum_term) == len(param.shape)
        sum_dimensionality.update(dict(zip(sum_term, param.shape)))
        if hasattr(param, "name") and param.name == differentiate_wrt:
            wrt_param = param
            wrt_indices = list(sum_term)

    grad_shape = tuple(
        [sum_dimensionality[index] for index in sum_out] + list(wrt_param.shape)
    )
    print(summation)
    print(", ".join(str(array.shape) for array in arrays))
    print(grad_shape)

    """
    GRAD = (out) x (in)
         = (out_term) x (in_term)
    A = (2, 3) b = (3)

    O = einsum("ij,j->i") => O_(i in 2) = sum_jin3 A_ij * b_j
    dO/dA = (2 x 2 x 3)
    [dO/dA]_(x,y,z) = dO_x / dA_(y,z) = d sum(j in 3) A_xj * bj / dA_(y,z)
                                      = d sum(j in 3) A_yj * bj if x == y else 0 / dA(y,z)
                                      = d Ayz * bz / dAyz
                                      = bz if x == y else 0
    d_[i,i] = b, elsewhere 0

    (i, i, j) => i)out=i_in -> grad is b_j up to j

    B = einsum("ij,j->j") => B_(j in 3) = sum_iin2 A_ij * b_j
    dB/dA = (3 x 2 x 3)
    [dB/dA]_(x,y,z) = dB_x / dA_(y,z) = d sum(i in 2) A_ix * b_x / dA_(y,z)
                                      = d sum(i in 2) A_iz * b_z / dA_(y,z) if x == z else 0
                                      = d Ayz * bz / dAyz
                                      = bz if x == z else 0
    d_[0,0,0] = b_0, d_[1,0,1] = b_1, d_[0,1,0] = b_0, d_[1,1,1] = b_1

    d_[i,:,i] = b[:2], elsewhere 0

    (j, i, j) => j_out=j_in -> grad is b_j up to i

    grad shape is (out shape) x (wrt shape)
    where the dimension from the output shape matches the corresponding dimension from the wrt param shape
    eg along the diagonal of out[n], wrt[n] where out[n] and wrt[n] are the same dimension in the einsum
        the gradient is the other param

    """
    raise NotImplementedError("einsum not implemented")


@differentiates(np.matmul)
def matmul(
    inputs: list[any], differentiate_wrt: str, rolling_partial: np.ndarray
) -> np.ndarray:
    for i, param in enumerate(inputs):
        if hasattr(param, "name") and param.name == differentiate_wrt:
            wrt_param, other_param = param, inputs[(i - 1) ** 2]
            wrt_index = i
            break
    if len(wrt_param.shape) == 1 or wrt_param.shape[-1] == 1:
        # We're taking a derivative w.r.t a vector: dAx/dx = A
        return rolling_partial.dot(other_param)
    elif len(other_param.shape) == 1 or other_param.shape[-1] == 1:
        # We're taking a derivative w.r.t a matrix: dAx/dA = a 3d matrix that has x^T on its 2d diagonal
        diagonal_shape = [wrt_param.shape[0], wrt_param.shape[0], 1]
        x_t = other_param
        grad_mul = "i,imo->io"
        if len(other_param.shape) > 1:
            diagonal_shape = [1] + diagonal_shape
            x_t = other_param.reshape(
                (other_param.shape[0], 1, 1, other_param.shape[-2])
            )
            grad_mul = "bi,bimo->bio"
        diagonal = np.diagflat(np.ones(wrt_param.shape[0])).reshape(diagonal_shape)
        grad = diagonal * x_t
        return np.einsum(grad_mul, rolling_partial, grad)
    else:
        if wrt_index == 0:
            return rolling_partial.dot(other_param.T)
        else:
            return other_param.T.dot(rolling_partial)


@differentiates(np.mean)
def mean(
    inputs: list[any], differentiate_wrt: str, rolling_partial: np.ndarray
) -> np.ndarray:
    dims = [
        arg for arg in inputs if hasattr(arg, "name") and arg.name == differentiate_wrt
    ][0].shape
    return rolling_partial * (np.ones(dims))


@differentiates(np.transpose)
def transpose(
    inputs: list[any], differentiate_wrt: str, rolling_partial: np.ndarray
) -> np.ndarray:
    return rolling_partial.transpose()


@differentiates(np.exp)
def exp(
    inputs: list[any], differentiate_wrt: str, rolling_partial: np.ndarray
) -> np.ndarray:
    return rolling_partial * (np.exp(inputs[0]))


@differentiates(np.square)
def square(
    inputs: list[any], differentiate_wrt: str, rolling_partial: np.ndarray
) -> np.ndarray:
    return rolling_partial * (2 * inputs[0])


@differentiates(np.maximum)
def maximum(
    inputs: list[any], differentiate_wrt: str, rolling_partial: np.ndarray
) -> np.ndarray:
    if len(inputs) >= 2 and np.isscalar(inputs[1]):
        return rolling_partial * ((inputs[0] >= inputs[1]))
    else:
        raise NotImplementedError(
            "maximum can only be differentiated between an array and scalar at the moment"
        )
