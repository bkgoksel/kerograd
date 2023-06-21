import numpy as np
from computation import ComputationGraph
from neural import *        


epochs = 1000
input_d = 3
layer_d = 5
batch_size = 4
x = np.random.random((batch_size, input_d))
y = np.array(((0,1),(0,1),(1,0),(1,0)))

l1 = LinearLayer(input_dim=input_d, output_dim=layer_d, name="L1")
l2 = LinearLayer(input_dim=layer_d, output_dim=2, name="L2")

def neural_net(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    return mean_squared_loss(l2.apply(relu(l1.apply(x), "ReLU")), y, "MSE")

for i in range(epochs):
    loss = neural_net(x, y)
    SimpleOptimizer.optimize(loss, learning_rate=2e-3)
    if not (i % 100):
        print(f"Loss: {loss}")

print(ComputationGraph.from_param(neural_net(x, y), store_full_graph=True))