from dataclasses import dataclass

import numpy as np

from .base import NamedOp
from .linear import LinearLayer
from .nonlinearity import ReLU
from .initialization import Initialization, HeInitialization


@dataclass
class FullyConnectedNet(NamedOp):
    input_dim: int
    output_dim: int
    layer_dim: int
    num_layers: int
    layers: list[LinearLayer] | None = None
    relu: ReLU | None = None
    name: str | None = None
    initialization: Initialization = HeInitialization
    type_name: str = "FullyConnectedNet"

    def __post_init__(self):
        super().__post_init__()
        self.layers = [
            LinearLayer(
                input_dim=self.input_dim if i == 0 else self.layer_dim,
                output_dim=self.output_dim
                if i == (self.num_layers - 1)
                else self.layer_dim,
                name=f"{self.name}_layer_{i}",
                initialization=self.initialization,
            )
            for i in range(self.num_layers)
        ]
        self.relu = ReLU(f"{self.name}_relu")

    def forward(self, x: np.ndarray) -> np.ndarray:
        intermediate_state = x
        for i, layer in enumerate(self.layers):
            intermediate_state = layer.apply(intermediate_state)
            if i < self.num_layers - 1:
                # only apply ReLU to non-final layers
                intermediate_state = self.relu.apply(intermediate_state)
        return intermediate_state
