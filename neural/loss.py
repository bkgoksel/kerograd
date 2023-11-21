from dataclasses import dataclass

import numpy as np

from computation import Param
from .base import NamedOp

@dataclass
class MeanSquaredLoss(NamedOp):
    name: str|None = None
    type_name: str = "MeanSquaredLoss"

    def forward(self, inpt: np.array, gold: np.array) -> np.array:
        return np.mean((inpt-gold)**2)

def mean_squared_loss(inpt: np.ndarray, gold: np.ndarray, name: str | None = None) -> Param:
    return MeanSquaredLoss(name).apply(inpt, gold)
