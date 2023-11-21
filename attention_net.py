import string

import numpy as np

from computation import ComputationGraph
from neural import (
    MultiHeadAttention,
    FullyConnectedNet,
    Embeddings,
    mean_squared_loss,
    SimpleOptimizer,
    get_positional_encodings,
    layer_norm,
    softmax,
)

epochs = 1000

vocab = {"~": 0} | {t[1]: t[0] + 1 for t in enumerate(string.ascii_lowercase)}
reverse_vocab = ["~"] + list(string.ascii_lowercase)

model_dim = 128
embedding_layer = Embeddings(vocab_size=len(vocab), dim=model_dim)

input_string = "abcde"
input_tokens = np.array([vocab[char] for char in input_string])

positional_encodings = get_positional_encodings(len(input_string), model_dim)

num_heads = 4
attention = MultiHeadAttention(
    input_dim=model_dim,
    output_dim=model_dim,
    num_heads=num_heads,
    mask=True,
)

layer_fc = FullyConnectedNet(
    input_dim=model_dim, output_dim=model_dim, layer_dim=2*model_dim, num_layers=2
)

gold_output_tokens = input_string[1:] + "~"
gold_output_indices = np.array([vocab[char] for char in gold_output_tokens])
gold_output_one_hot = np.eye(len(vocab))[gold_output_indices]

def iteration(i):
    hidden_state = embedding_layer.apply(input_tokens)
    hidden_state = hidden_state + positional_encodings
    hidden_state = hidden_state + attention.apply(
        q=hidden_state, k=hidden_state, v=hidden_state
    )
    hidden_state = layer_norm(hidden_state) # [context_len, model_dim]

    hidden_state = hidden_state + layer_fc.apply(hidden_state)
    # TODO: Uncommenting below causes infinite recursion errors
    #hidden_state = layer_norm(hidden_state)

    hidden_state = embedding_layer.to_embeddings(hidden_state) # [context_len, vocab_size]
    output_token_probs = softmax(hidden_state, axis=-1) # [context_len, vocab_size] <- next token prediction for each index

    loss = mean_squared_loss(hidden_state, gold_output_one_hot)

    SimpleOptimizer.optimize(loss, learning_rate=1e-5)
    if not (i % 4):
        print(output_token_probs.shape) # [cl, vocab]
        prediction = np.argmax(output_token_probs, axis=1)
        prediction = " ".join(reverse_vocab[i] for i in prediction)
        print(f"Loss: {loss}, Prediction: {prediction}")

for i in range(epochs):
    iteration(i)
