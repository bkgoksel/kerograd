from dataclasses import dataclass
import numpy as np

from .base import NamedOp
from .nets import FullyConnectedNet
from .attention import MultiHeadAttention
from .embedding import Embeddings
from .ops import softmax, layer_norm
from .positional_encoding import get_positional_encodings


@dataclass
class DecoderLayer(NamedOp):
    input_dim: int
    output_dim: int
    fc_num_layers: int
    fc_layer_dim: int
    num_heads: int
    attention: MultiHeadAttention | None = None
    fc_net: FullyConnectedNet | None = None
    name: str | None = None
    type_name: str = "DecoderLayer"

    def __post_init__(self):
        super().__post_init__()
        self.attention = MultiHeadAttention(
            input_dim=self.input_dim,
            output_dim=self.output_dim,
            num_heads=self.num_heads,
            mask=True,
            name=f"{self.name}_multihead_attention",
        )
        self.fc_net = FullyConnectedNet(
            input_dim=self.output_dim,
            output_dim=self.output_dim,
            layer_dim=self.fc_layer_dim,
            num_layers=self.fc_num_layers,
            name=f"{self.name}_fc_net",
        )

    def forward(self, input_state: np.ndarray) -> np.ndarray:
        hidden_state = input_state + self.attention.apply(
            q=input_state, k=input_state, v=input_state
        )
        #hidden_state = layer_norm(hidden_state)
        hidden_state = hidden_state + self.fc_net.apply(hidden_state)
        #hidden_state = layer_norm(hidden_state)
        return hidden_state


@dataclass
class Transformer(NamedOp):
    vocab: dict[str, int]
    model_dim: int
    num_heads: int
    num_decoder_layers: int
    reverse_vocab: dict[int, str] | None = None
    embedding_layer: Embeddings | None = None
    decoders: list[DecoderLayer] | None = None
    name: str | None = None
    type_name: str = "Transformer"

    def __post_init__(self):
        super().__post_init__()
        self.reverse_vocab = {v: k for k, v in self.vocab.items()}
        self.embedding_layer = Embeddings(vocab_size=len(self.vocab), dim=self.model_dim, name=f"{self.name}_embeddings")
        self.decoders = [
            DecoderLayer(
                input_dim=self.model_dim,
                output_dim=self.model_dim,
                fc_num_layers=2,
                fc_layer_dim=2 * self.model_dim,
                num_heads=self.num_heads,
                name=f"{self.name}_decoder_l_{i}",
            )
            for i in range(self.num_decoder_layers)
        ]

    def forward(self, input_tokens: list[str]) -> np.ndarray:
        input_tokens = np.array([self.vocab[tok] for tok in input_tokens])
        positional_encodings = get_positional_encodings(
            len(input_tokens), self.model_dim
        )
        hidden_state = self.embedding_layer.apply(input_tokens)
        hidden_state = hidden_state + positional_encodings
        for decoder_layer in self.decoders:
            hidden_state = decoder_layer.apply(hidden_state)
        logits = self.embedding_layer.to_embeddings(hidden_state)
        return logits

    def _get_prediction(self, logits: np.ndarray):
        token_probs = softmax(logits, axis=-1)
        prediction = np.argmax(token_probs, axis=-1)
        return [self.reverse_vocab[i] for i in prediction]

    def predict(self, input_tokens: list[str]) -> list[str]:
        logits = self.apply(input_tokens)
        return self._get_prediction(logits)
