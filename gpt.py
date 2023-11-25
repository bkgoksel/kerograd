import string
import sys

import numpy as np

from computation import ComputationGraph
from neural import (
    mean_squared_loss,
    SimpleOptimizer,
    Transformer,
)

# epochs to train on
epochs = 1000

# vocab is ascii lowercase alphabet and EOS (~) character
vocab = {"~": 0} | {t[1]: t[0] + 1 for t in enumerate(string.ascii_lowercase)}

# model parameters
model_dim = 16
num_heads = 2
num_decoder_layers = 2

# build the model
transformer = Transformer(vocab, model_dim, num_heads, num_decoder_layers)

# input data covers the entire alphabet, some overlap between segments
input_data = [["a","b","c","d"], ["c","d","e","f"], ["f","g","h"], ["g","h","j","k","l"],["k","l"],["l","m","n","o"],["m","n","o","p"],["m","n","o","p","q"],["q","r","s","t"],["s","t","u","v"],["v","w","x","y","z"]]
all_gold_output_tokens = [input_tokens[1:] + ["~"] for input_tokens in input_data]
all_gold_output_indices = [np.array([vocab[char] for char in gold_output_tokens]) for gold_output_tokens in all_gold_output_tokens]
all_gold_output_one_hot = [np.eye(len(vocab))[gold_output_indices] for gold_output_indices in all_gold_output_indices]

def iteration(i):
    total_loss = .0
    # no batching, iterate over each example one-by-one
    for input_tokens, gold_output_one_hot in zip(input_data, all_gold_output_one_hot):
        logits = transformer.apply(input_tokens)
        loss = mean_squared_loss(logits, gold_output_one_hot)
        SimpleOptimizer.optimize(loss, learning_rate=2e-4, gradient_clip=1e2)
        total_loss = total_loss + loss
    avg_loss = total_loss / len(input_data)
    if not (i % 4):
        prediction = ' '.join(transformer._get_prediction(logits))
        print(f"Loss: {avg_loss}, Prediction: {prediction}")

for i in range(epochs):
    iteration(i)
