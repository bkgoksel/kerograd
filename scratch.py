import numpy as np
from computation import ComputationGraph
from neural import *        


epochs = 1000
input_d = 3
layer_d = 5
x = np.arange(input_d)+5
y =np.array((0,1))

l1 = LinearLayer(input_dim=input_d, output_dim=layer_d)
relu = ReLU()
l2 = LinearLayer(input_dim=layer_d, output_dim=2)
loss_op = MeanSquaredLoss()

optimizer = SimpleOptimizer()

for i in range(epochs):
    loss = loss_op.apply(l2.apply(relu.apply(l1.apply(x))), y)
    graph = ComputationGraph.from_param(loss)
    optimizer.optimize(graph)
    if not (i % 100):
        print(f"Loss: {loss}")
