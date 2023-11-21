from dataclasses import dataclass
import math

import numpy as np

from computation import Param
from .base import NamedOp
from .initialization import Initialization, HeInitialization

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
            return self.W @ inpt + self.b
