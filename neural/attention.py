import numpy as np
import math

from dataclasses import dataclass
from computation import Param

from .base import NamedOp
from .ops import softmax
from .initialization import Initialization, HeInitialization


@dataclass
class AttentionHead(NamedOp):
    input_dim: int
    output_dim: int
    mask: bool = False
    name: str | None = None
    W_k: Param | None = None
    W_v: Param | None = None
    W_q: Param | None = None
    initialization: Initialization = HeInitialization
    type_name: str = "AttentionHead"

    def __post_init__(self):
        super().__post_init__()
        self.W_k = Param(
            self.initialization.initialize((self.input_dim, self.output_dim)),
            name=f"{self.name}_W_k",
            trainable=True,
        )
        self.W_q = Param(
            self.initialization.initialize((self.input_dim, self.output_dim)),
            name=f"{self.name}_W_q",
            trainable=True,
        )
        self.W_v = Param(
            self.initialization.initialize((self.input_dim, self.output_dim)),
            name=f"{self.name}_W_v",
            trainable=True,
        )

    def forward(
        self,
        q_in: np.array,  # [context_len, input_dim]
        k_in: np.array,  # [context_len, input_dim]
        v_in: np.array,  # [context_len, input_dim]
    ) -> np.array:
        Q = q_in @ self.W_q  # [context_len, output_dim]
        K = k_in @ self.W_k  # [context_len, output_dim]
        V = v_in @ self.W_v  # [context_len, output_dim]
        importance = Q @ K.transpose()  # [context_len, context_len]
        if self.mask:
            diagonal_mask = -1e8 * (1 - np.tril(np.ones(q_in.shape[0])))
            importance = importance + diagonal_mask
        importance_scaled = importance / math.sqrt(self.output_dim)
        importance_softmax = softmax(importance_scaled)
        return importance_softmax @ V  # [context_len, output_dim]


@dataclass
class MultiHeadAttention(NamedOp):
    input_dim: int
    output_dim: int
    num_heads: int
    mask: bool = False
    name: str | None = None
    W_o: Param | None = None
    heads: list[Param] | None = None
    initialization: Initialization = HeInitialization
    type_name: str = "AttentionHead"

    def __post_init__(self):
        super().__post_init__()
        self.heads = [
            AttentionHead(
                input_dim=self.input_dim,
                output_dim=self.output_dim // self.num_heads,
                mask=self.mask,
                name=f"{self.name}_head_{i}",
                initialization=self.initialization,
            )
            for i in range(self.num_heads)
        ]
        self.W_o = self.initialization.initialize((self.output_dim, self.output_dim))

    def forward(
        self,
        q: np.array,  # [context_len, input_dim]
        k: np.array,  # [context_len, input_dim]
        v: np.array,  # [context_len, input_dim]
    ) -> np.array:  # [context_len, output_dim]
        return (
            np.concatenate([head.apply(q, k, v) for head in self.heads], axis=-1) @ self.W_o
        )
