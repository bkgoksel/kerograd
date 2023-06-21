import numpy as np
from computation import ComputationGraph
from neural import *        


epochs = 1000
input_d = 3
layer_d = 5
x = np.random.random(input_d)
y = np.array((0,1))

l1 = LinearLayer(input_dim=input_d, output_dim=layer_d)
l2 = LinearLayer(input_dim=layer_d, output_dim=2)

def neural_net(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    return mean_squared_loss(l2.apply(relu(l1.apply(x))), y)

for i in range(epochs):
    loss = neural_net(x, y)
    SimpleOptimizer.optimize(loss, learning_rate=2e-3)
    if not (i % 100):
        print(f"Loss: {loss}")