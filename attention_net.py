import string

import numpy as np

from computation import ComputationGraph
from neural import (
    mean_squared_loss,
    SimpleOptimizer,
    Transformer,
)

epochs = 1000

vocab = {"~": 0} | {t[1]: t[0] + 1 for t in enumerate(string.ascii_lowercase)}
model_dim = 128
num_heads = 4
num_decoder_layers = 1
transformer = Transformer(vocab, model_dim, num_heads, num_decoder_layers)

input_tokens = ["a", "b", "c", "d"]
gold_output_tokens = input_tokens[1:] + ["~"]
gold_output_indices = np.array([vocab[char] for char in gold_output_tokens])
gold_output_one_hot = np.eye(len(vocab))[gold_output_indices]

def iteration(i):
    logits = transformer.apply(input_tokens)
    loss = mean_squared_loss(logits, gold_output_one_hot)
    SimpleOptimizer.optimize(loss, learning_rate=2e-4, gradient_clip=1e2)
    if not (i % 4):
        prediction = ' '.join(transformer._get_prediction(logits))
        print(f"Loss: {loss}, Prediction: {prediction}")

for i in range(epochs):
    iteration(i)
