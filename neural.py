from abc import abstractmethod, abstractproperty
import numpy as np
import math
from dataclasses import dataclass
from computation import Param, ComputationGraph

@dataclass
class NamedOp:

    @abstractproperty
    def type_name(cls) -> str:
        pass

    @abstractmethod
    def forward(*args, **kwargs) -> np.ndarray:
        pass

    def __post_init__(self):
        self.name = self.name or f"{self.type_name}_{Param.random_name()}"

    def apply(self, *args, **kwargs) -> Param:
        res = self.forward(*args, **kwargs)
        parent = getattr(res, 'parent', None)
        return Param(res, name=f"{self.name}_out", parent=parent)

class Initialization:
    def initialize(shape: tuple[int], rng: np.random.Generator|None = None) -> np.array:
        pass

class HeInitialization:
    def initialize(shape: tuple[int], rng: np.random.Generator|None = None) -> np.array:
        rng = rng or np.random.default_rng()
        return rng.normal(0, math.sqrt(2/shape[0]), shape)

@dataclass
class LinearLayer(NamedOp):
    input_dim: int
    output_dim: int
    name: str|None = None
    W: Param|None = None
    b: Param|None = None
    initialization: Initialization = HeInitialization
    type_name: str = "Linear"

    def __post_init__(self):
        super().__post_init__()
        self.W = Param(self.initialization.initialize((self.output_dim, self.input_dim)), name=f"{self.name}_W", trainable=True)
        self.b = Param(np.zeros((self.output_dim)), name=f"{self.name}_b", trainable=True)

    def forward(self, inpt: np.array) -> Param:
        if len(inpt.shape) > 1:
            return (self.W @ inpt.reshape(list(inpt.shape) + [1])).squeeze(-1) + self.b
        else:
            # W @ inpt + b
            return np.einsum('ij,j->i', self.W, inpt) + self.b
        
@dataclass
class ReLU(NamedOp):
    name: str|None = None
    type_name: str = "ReLU"

    def forward(self, inpt: np.array) -> Param:
        return np.maximum(inpt, 0)

def relu(inpt: np.array, name: str | None = None) -> Param:
    return ReLU(name).apply(inpt)

@dataclass
class MeanSquaredLoss(NamedOp):
    name: str|None = None
    type_name: str = "MeanSquaredLoss"

    def forward(self, inpt: np.array, gold: np.array) -> np.array:
        return np.mean((inpt-gold)**2)

def mean_squared_loss(inpt: np.ndarray, gold: np.ndarray, name: str | None = None) -> Param:
    return MeanSquaredLoss(name).apply(inpt, gold)

class Optimizer:

    @abstractmethod
    def optimize_graph(self, graph: ComputationGraph) -> None:
        pass

    def optimize_param(self, param: Param) -> None:
        """
        Applies gradient descent to compuation graph resulting in the given parameter to minimize it.
        """
        self.optimize_graph(ComputationGraph.from_param(param))

    @classmethod
    def optimize(cls, graph: ComputationGraph, *args, **kwargs) -> None:
        return cls(*args, **kwargs).optimize_graph(graph)

    @classmethod
    def optimize(cls, param: Param, *args, **kwargs) -> None:
        return cls(*args, **kwargs).optimize_param(param)

@dataclass
class SimpleOptimizer(Optimizer):
    learning_rate: float = 2e-3

    def optimize_graph(self, graph: ComputationGraph) -> None:
        for param_name, param in graph.params.items():
            if param_name in graph.grads:
                param.grad_update(self.learning_rate * graph.grads[param_name])

