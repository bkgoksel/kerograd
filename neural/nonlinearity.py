from dataclasses import dataclass

import numpy as np

from computation import Param
from .base import NamedOp

@dataclass
class ReLU(NamedOp):
    name: str|None = None
    type_name: str = "ReLU"

    def forward(self, inpt: np.array) -> Param:
        return np.maximum(inpt, 0)

def relu(inpt: np.array, name: str | None = None) -> Param:
    return ReLU(name).apply(inpt)

