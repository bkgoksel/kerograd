import numpy as np

from dataclasses import dataclass

from computation import Param
from .base import NamedOp
from .initialization import Initialization, HeInitialization, RandomInitialization

@dataclass
class Embeddings(NamedOp):
    vocab_size: int
    dim: int
    name: str|None = None
    embeddings: Param|None = None
    initialization: Initialization = RandomInitialization
    type_name: str = "Embeddings"

    def __post_init__(self):
        super().__post_init__()
        self.embeddings = Param(self.initialization.initialize((self.vocab_size, self.dim)), name=f"{self.name}_embedding_matrix", trainable=True)

    def forward(self, sequence: np.array) -> np.array:
        return self.embeddings.take(sequence, axis=0)

    def to_embeddings(self, sequence: np.array) -> np.array:
        def _to_embeddings(sequence):
            embed_t = self.embeddings.transpose()
            return sequence @ embed_t # (c_l, emb_dim) @ (emb_dim, vocab_size) -> (c_l, vocab_size)
        return self.apply_fn(_to_embeddings, sequence)
