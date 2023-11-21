from abc import abstractmethod, abstractproperty
from dataclasses import dataclass
from computation import Param, ComputationGraph

import numpy as np

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
    gradient_clip: float = 1e4

    def optimize_graph(self, graph: ComputationGraph) -> None:
        for param_name, param in graph.params.items():
            if param_name in graph.grads:
                grad = graph.grads[param_name]
                grad = np.clip(grad, -self.gradient_clip, self.gradient_clip)
                param.grad_update(self.learning_rate * grad)

