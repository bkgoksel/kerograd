from abc import abstractmethod
from dataclasses import dataclass
import numpy as np
import string
import random
from typing import Optional, Iterable

from derivation import DIFFERENTIABLE_FUNCTIONS
    
HANDLED_FUNCTIONS = {}

@dataclass
class Edge:
    """
    An Edge of a computation graph. Stores the function name that was applied and the arguments
    """
    func: str
    method: any
    inputs: iter
    kwargs: dict[str, any]
    trainable: bool = False

class Param(np.ndarray):
    """
    A Numpy ndarray that keeps track of computations it's been in.
    """

    def random_name(length: int = 12) -> str:
        return f"_auto_{''.join(random.choices(string.ascii_lowercase, k=length))}"

    def __new__(cls, value: np.ndarray, name: str = None, parent: Edge = None, trainable: bool = False):
        obj = np.asarray(value).view(cls)
        obj.name = name or Param.random_name()
        obj.parent = parent
        obj.trainable = trainable
        return obj

    def __array_finalize__(self, obj):
        if obj is None: return
        self.name = getattr(obj, 'name', None)
        self.parent = getattr(obj, 'parent', None)
        self.trainable = getattr(obj, 'trainable', False)

    def __array_ufunc__(self, ufunc, method, *inputs, out=None, **kwargs):
        if method in ("at", "reduceat"):
            """in place operations for parameters not supported"""
            return NotImplemented
        if out:
            """Specifying outputs not supported yet"""
            out_args = [output.view(np.ndarray) if isinstance(output, Param) else output for output in out]
            kwargs['out'] = tuple(out_args)
            outputs = out
        else:
            outputs = (None,) * ufunc.nout
        input_arrays = []
        trainable = self._is_upstream_of_trainable()
        for input_arg in inputs:
            if isinstance(input_arg, Param):
                trainable = trainable or input_arg._is_upstream_of_trainable()
                input_arrays.append(input_arg.view(np.ndarray))
            else:
                input_arrays.append(input_arg)
        results = super().__array_ufunc__(ufunc, method, *input_arrays, **kwargs)
        if results is NotImplemented:
            return NotImplemented
        if ufunc.nout == 1:
            results = (results,)

        
        # pass grads to the results here too
        results = tuple((np.asarray(result).view(Param) if output is None else output for result, output in zip(results, outputs)))
        parent = Edge(ufunc, method, inputs, kwargs, trainable=trainable)
        name = Param.random_name()
        if results:
            if isinstance(results[0], Param):
                results[0].parent = parent
                results[0].name = name
            if np.isscalar(results[0]):
                results[0] = Param(results[0], name, parent)

        return results[0] if len(results) == 1 else results


    def __array_function__(self, func, types, args, kwargs):
        if func not in HANDLED_FUNCTIONS:
            return NotImplemented
        # Note: this allows subclasses that don't override
        # __array_function__ to handle Param objects
        if not all(issubclass(t, Param) for t in types):
            return NotImplemented
        return HANDLED_FUNCTIONS[func](*args, **kwargs)

    def implements(numpy_function):
        """Register an __array_function__ implementation for MyArray objects."""
        def decorator(func):
            HANDLED_FUNCTIONS[numpy_function] = func
            return func
        return decorator

    @implements(np.transpose)
    def transpose(a: np.ndarray, axes: list[int] = None) -> np.ndarray:
        return Param(value=a.view(np.ndarray).transpose(), name=Param.random_name(), parent=Edge(np.transpose, method=None, inputs=(a,), kwargs={"axes": axes}, trainable=getattr(a, 'trainable', False)))

    @implements(np.mean)
    def mean(a: np.ndarray, axis: None|int|tuple[int] = None, dtype: None|np.dtype = None, keepdims: bool = False, where: None|np.ndarray = None) -> np.ndarray:
        result_arr = a.view(np.ndarray).mean()
        return Param(value=(result_arr,), name=Param.random_name(), parent=Edge(np.mean, method=None, inputs=(a,), kwargs={"axis": axis, "dtype": dtype, "keepdims":keepdims, "where": where}), trainable=getattr(a, 'trainable', False))

    def __iadd__(self, *args, **kwargs):
        np.ndarray.__iadd__(self.view(np.ndarray), *args, **kwargs)

    def grad_update(self, val: np.ndarray):
        self.__iadd__(-val)


    def _is_upstream_of_trainable(self) -> bool:
        return self.trainable or (self.parent and self.parent.trainable)

@dataclass
class ComputationGraph:
    root: Param
    params: dict[str, Param]
    grads: dict[str, np.ndarray]

    @classmethod
    def from_param(cls, root_param: Param, store_full_graph: bool = False):
        params, grads = zip(*cls.backwards(root_param, store_full_graph))

        param_names = [param.name for param in params]
        params = {param_name: param for param_name, param in zip(param_names, params)}
        grads = {param_name: grad for param_name, grad in zip(param_names, grads)}

        return ComputationGraph(root_param, params, grads)

    @classmethod
    def backwards(cls, root_param: Param, store_full_graph: bool = False, partial_grad: np.ndarray|None = None) -> Iterable[tuple[Param, np.ndarray]]:
        """
        Starting from the root parameter, yields all the upstream parameters and their gradients with respect to the root.
        Only returns the parameters and gradients for the trainable parameters, unless store_full_graph is true. 
        """
        if root_param.parent:
            if root_param.parent.func in DIFFERENTIABLE_FUNCTIONS:
                if partial_grad is None:
                    partial_grad = np.ones(root_param.shape)
                differential = DIFFERENTIABLE_FUNCTIONS[root_param.parent.func]
                for parent_param in root_param.parent.inputs: 
                    if isinstance(parent_param, Param):
                        param_grad = differential(root_param.parent.inputs, parent_param.name, partial_grad)
                        if store_full_graph or parent_param.trainable:
                            yield (parent_param, param_grad)
                        if store_full_graph or parent_param._is_upstream_of_trainable():
                            yield from cls.backwards(parent_param, store_full_graph, param_grad)
                
    @classmethod
    def _graph_string(cls, root_param: Param, prefix="") -> str:
        str = ""
        str += f"{prefix}Name: {root_param.name}\n"
        str += f"{prefix}Shape: {root_param.shape}\n"
        str += f"{prefix}Trainable: {root_param.trainable}\n"
        if root_param.parent:
            str+= f"{prefix}Parent: {root_param.parent.func.__name__}\n"
            if root_param.parent.func in DIFFERENTIABLE_FUNCTIONS:
                str += f"{prefix}-->differentiable\n"
            else:
                str += f"{prefix}-->NOT differentiable\n"
            for parent_param in root_param.parent.inputs:
                if isinstance(parent_param, Param):
                    str += f"{prefix}\tParam:\n"
                    str += cls._graph_string(parent_param, prefix + "\t  ")
                else:
                    str += f"{prefix}\tParam:\n{prefix}\t  {parent_param}\n"
        return str

    def __str__(self) -> str:
        return self._graph_string(self.root)

    def summary(self) -> None:
        print("\n".join([f"{param_name}: {param.shape} <-> {self.grads[param_name].shape}" for param_name, param in self.params.items()]))

        
