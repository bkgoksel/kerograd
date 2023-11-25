import sys
import numpy as np
from computation import ComputationGraph, Param
#from neural import FullyConnectedNet, mean_squared_loss, SimpleOptimizer, softmax, layer_norm
from neural import *

def test(x, y, epochs=1000, layer_d=64):
    input_d = x.shape[-1]
    output_d = y.shape[-1]

    FCNN = FullyConnectedNet(input_d, output_d, layer_dim=layer_d, num_layers=4)
    output = FCNN.apply(x)
    output = layer_norm(output)
    output = softmax(output)

    graph = ComputationGraph.from_param(mean_squared_loss(output, y, "loss"))
    graph.summary()
    sys.exit(0)

    for i in range(epochs):
        loss = mean_squared_loss(FCNN.apply(x), y, "mean_squared_loss")
        SimpleOptimizer.optimize(loss, learning_rate=2e-3)
        if not (i % 100):
            print(f"Loss: {loss}")

x = np.random.random((3))
y = np.array((0,1))

xb = np.random.random((4, 6))
yb = np.array(((0,1),(0,1),(1,0),(1,0)))

#test(x,y)
#test(xb, yb)

#single_head = AttentionHead(input_dim=3, output_dim=2, mask=True)
#attention = MultiHeadAttention(input_dim=6, output_dim=6, num_heads=2, mask=True)

out = attention.apply(xb, xb, xb)

#print(ComputationGraph.from_param(out))
ComputationGraph.from_param(out).summary()
