from abc import abstractmethod, abstractproperty
import numpy as np
from dataclasses import dataclass
from computation import Param


@dataclass
class NamedOp:
    @abstractproperty
    def type_name(self) -> str:
        pass

    @abstractmethod
    def forward(self, *args, **kwargs) -> np.ndarray:
        pass

    def __post_init__(self):
        self.name = self.name or f"{self.type_name}_{Param.random_name()}"

    def apply(self, *args, **kwargs) -> Param:
        res = self.forward(*args, **kwargs)
        parent = getattr(res, "parent", None)
        return Param(res, name=f"{self.name}_out", parent=parent)

    def apply_fn(self, fn, *args, **kwargs) -> Param:
        res = fn(*args, **kwargs)
        parent = getattr(res, "parent", None)
        return Param(res, name=f"{self.name}_{fn}_out", parent=parent)
