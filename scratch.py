import numpy as np
from computation import ComputationGraph
from neural import *        

def test(x, y):
    epochs = 2000
    input_d = 3
    layer_d = 5

    l1 = LinearLayer(input_dim=input_d, output_dim=layer_d, name="L1")
    l2 = LinearLayer(input_dim=layer_d, output_dim=2, name="L2")

    def neural_net(x: np.ndarray) -> np.ndarray:
        return l2.apply(relu(l1.apply(x), "ReLU"))

    graph = ComputationGraph.from_param(mean_squared_loss(neural_net(x), y, "loss"), store_full_graph=True)
    #print(graph)
    graph.summary()
    return

    for i in range(epochs):
        loss = mean_squared_loss(neural_net(x), y, "mean_squared_loss")
        SimpleOptimizer.optimize(loss, learning_rate=2e-3)
        if not (i % 100):
            print(f"Loss: {loss}")

x = np.random.random((3))
y = np.array((0,1))

xb = np.random.random((4, 3))
yb = np.array(((0,1),(0,1),(1,0),(1,0)))

test(x,y)
test(xb, yb)