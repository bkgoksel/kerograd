# Kerograd

My own little backpropagation library, built as a personal exercise.
Also has a GPT-style decoder-only Transformer built with it as an example.

* `computation.py` for computation graph implementation. This is where computation provenance is stored to make sure gradients are propagated during backprop.
* `derivation.py` for gradient computation code. This is where the partial derivatives for various functions are implemented.
* `scratch.py` for testing code that overfits a tiny MLP to a single example.
* `gpt.py` file that trains a GPT model.
* `neural`: implementation of higher-level neural net building blocks:
  * `attention.py`: Multi-head attention implementation
  * `base.py`: Base utilities for making sure all these neural ops construct debuggable computation graphs.
  * `embedding.py`: Basic embedding layer.
  * `initialization.py`: Weight initialization algorithms. He and uniform random initialization supported.
  * `linear.py`: Implementation of a fully connected linear layer.
  * `loss.py`: Loss computations.Acurrently only mean squared loss is implemented. 
  * `nets.py`: Multi-layer perceptron (MLP) implementation.
  * `nonlinearity.py`: Nonlinearities gfor neural nets. Only ReLU implemented for now.
  * `ops.py`: One-off operations: softmax etc.
  * `optimizer.py`: Optimizers. Only a simple optimizer is implemented.
  * `positional_encoding.py`: Sin/cos-based positional encoding generation as in the original transformer.
  * `transformer.py`: The actual transformer implementation.
