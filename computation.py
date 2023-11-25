from dataclasses import dataclass
import numpy as np
import string
import random
from typing import Iterable

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
    propagate_grads: bool = False

    def __str__(self) -> str:
        return f"Edge(func={self.func}, method={self.method}, propagate_grads={self.propagate_grads})"


class Param(np.ndarray):
    """
    A Numpy ndarray that keeps track of computations it's been in.
    """

    @staticmethod
    def random_name(length: int = 12) -> str:
        return f"_auto_{''.join(random.choices(string.ascii_lowercase, k=length))}"

    def __new__(
        cls,
        value: np.ndarray,
        name: str = None,
        parent: Edge = None,
        trainable: bool = False,
        propagate_grads: bool = False,
    ):
        obj = np.asarray(value).view(cls)
        obj.name = name or Param.random_name()
        obj.parent = parent
        if obj.parent and any(
            hasattr(param, "name") and param.name == obj.name
            for param in obj.parent.inputs
        ):
            raise Exception()
        obj.trainable = trainable
        obj.propagate_grads = propagate_grads
        return obj

    def __array_finalize__(self, obj):
        if obj is None:
            return
        self.name = getattr(obj, "name", None)
        self.parent = getattr(obj, "parent", None)
        self.trainable = getattr(obj, "trainable", False)
        self.propagate_grads = getattr(obj, "propagate_grads", False)
        if self.parent and any(
            hasattr(param, "name") and param.name == self.name
            for param in self.parent.inputs
        ):
            raise Exception()

    def __array_ufunc__(self, ufunc, method, *inputs, out=None, **kwargs):
        if method in ("at", "reduceat"):
            """in place operations for parameters not supported"""
            return NotImplemented
        if out:
            """Specifying outputs not supported yet"""
            out_args = [
                output.view(np.ndarray) if isinstance(output, Param) else output
                for output in out
            ]
            kwargs["out"] = tuple(out_args)
            outputs = out
        else:
            outputs = (None,) * ufunc.nout
        input_arrays = []
        propagate_grads = self._grads_from_upstream() or any(
            self._any_propagate_grads(arg) for arg in inputs
        )
        input_arrays = [self._normalize_arg(arg) for arg in inputs]
        try:
            results = super().__array_ufunc__(ufunc, method, *input_arrays, **kwargs)
        except RuntimeWarning as ex:
            print(results)
            print(ex)
            breakpoint()
        if results is NotImplemented:
            return NotImplemented
        if ufunc.nout == 1:
            results = (results,)

        # pass grads to the results here too
        results = tuple(
            (
                np.asarray(result).view(Param) if output is None else output
                for result, output in zip(results, outputs)
            )
        )
        parent = Edge(ufunc, method, inputs, kwargs, propagate_grads=propagate_grads)
        name = f"{ufunc.__name__}_{Param.random_name()}"
        if results:
            if isinstance(results[0], Param):
                results[0].parent = parent
                results[0].name = name
                results[0].propagate_grads = propagate_grads
            if np.isscalar(results[0]):
                results[0] = Param(
                    results[0], name, parent, propagate_grads=propagate_grads
                )

        return results[0] if len(results) == 1 else results

    def _normalize_arg(self, arg):
        if isinstance(arg, Param):
            return arg.view(np.ndarray)
        if isinstance(arg, list):
            return [self._normalize_arg(subarg) for subarg in arg]
        if isinstance(arg, tuple):
            return tuple((self._normalize_arg(subarg) for subarg in arg))
        return arg

    def _any_propagate_grads(self, arg):
        if isinstance(arg, Param):
            return arg.trainable or arg.propagate_grads
        if isinstance(arg, list) or isinstance(arg, tuple):
            return any(self._any_propagate_grads(subarg) for subarg in arg)

    def __array_function__(self, func, types, args, kwargs):
        # Note: this allows subclasses that don't override
        # __array_function__ to handle Param objects
        if not all(issubclass(t, Param) for t in types):
            return NotImplemented
        input_arrays = [self._normalize_arg(arg) for arg in args]
        return_arr = func(*input_arrays, **kwargs)
        propagate_grads = any(self._any_propagate_grads(arg) for arg in args)
        return Param(
            value=return_arr,
            name=f"{func.__name__}_{Param.random_name()}",
            parent=Edge(
                func,
                method=None,
                inputs=args,
                kwargs=kwargs,
                propagate_grads=propagate_grads,
            ),
            propagate_grads=propagate_grads,
        )

    def __iadd__(self, *args, **kwargs):
        np.ndarray.__iadd__(self.view(np.ndarray), *args, **kwargs)

    def grad_update(self, val: np.ndarray):
        if len(val.shape) == len(self.shape) + 1:
            # batched
            val = val.sum((0))
        self.__iadd__(-val)

    def _grads_from_upstream(self) -> bool:
        return (
            self.trainable
            or self.propagate_grads
            or (self.parent and self.parent.propagate_grads)
        )

    def __str__(self) -> str:
        return f"Param(name={self.name}, shape={self.shape}, parent={self.parent}, trainable={self.trainable}), propagate_grads={self.propagate_grads}"


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
    def backwards(
        cls,
        root_param: Param,
        store_full_graph: bool = False,
        partial_grad: np.ndarray | None = None,
    ) -> Iterable[tuple[Param, np.ndarray]]:
        """
        Starting from the root parameter, yields all the upstream parameters and their gradients with respect to the root.
        Only returns the parameters and gradients for the trainable parameters, unless store_full_graph is true.
        """
        if root_param.parent:
            parent = root_param.parent
            if parent.func in DIFFERENTIABLE_FUNCTIONS:
                if partial_grad is None:
                    partial_grad = np.ones(root_param.shape)
                differential = DIFFERENTIABLE_FUNCTIONS[parent.func]
                parent_input_params = (
                    parent.inputs[0] if parent.func == np.concatenate else parent.inputs
                )
                for parent_param in parent_input_params:
                    if isinstance(parent_param, Param):
                        try:
                            param_grad = differential(
                                root_param.parent.inputs,
                                root_param.parent.kwargs,
                                parent_param.name,
                                partial_grad,
                            )
                        except NotImplementedError as ex:
                            raise NotImplementedError(
                                f"{parent_param.name}'s function {root_param.parent.func} not fully implemented: {ex}"
                            )
                        if store_full_graph or parent_param.trainable:
                            yield (parent_param, param_grad)
                        if store_full_graph or parent_param.propagate_grads:
                            yield from cls.backwards(
                                parent_param, store_full_graph, param_grad
                            )
            else:
                print(f"{root_param.parent.func} not differentiable")

    def __str__(self) -> str:
        printer = GraphPrinter(printed=set())
        return printer._print_graph(self.root)

    def summary(self) -> None:
        print(
            "\n".join(
                [
                    f"{param_name}: {param.shape} <-> {self.grads[param_name].shape}"
                    for param_name, param in self.params.items()
                ]
            )
        )


@dataclass
class GraphPrinter:
    printed: set[str]

    def _print_graph(self, root_param, prefix="") -> str:
        str = ""
        str += f"{prefix}Name: {root_param.name}\n"
        str += f"{prefix}Shape: {root_param.shape}\n"
        str += f"{prefix}Trainable: {root_param.trainable}\n"
        str += f"{prefix}Propagate grads: {root_param.propagate_grads}\n"
        # print(str)
        if root_param.name in self.printed:
            return str
        self.printed.add(root_param.name)
        if root_param.parent:
            str += f"{prefix}Parent: {root_param.parent.func.__name__}\n"
            if root_param.parent.func in DIFFERENTIABLE_FUNCTIONS:
                str += f"{prefix}-->differentiable\n"
            else:
                str += f"{prefix}-->NOT differentiable\n"
            for parent_param in root_param.parent.inputs:
                if isinstance(parent_param, Param):
                    str += f"{prefix}\tParam:\n"
                    str += self._print_graph(parent_param, prefix + "\t  ")
                elif isinstance(parent_param, np.ndarray):
                    str += f"{prefix}\tParam:\n{prefix}\t  {parent_param.shape}\n"
                else:
                    str += f"{prefix}\tParam:\n{prefix}\t  {parent_param}\n"
        return str
