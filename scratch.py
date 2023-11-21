import numpy as np
from computation import ComputationGraph
from neural import FullyConnectedNet, mean_squared_loss, SimpleOptimizer

def test(x, y, epochs=1000, layer_d=64):
    input_d = x.shape[-1]
    output_d = y.shape[-1]

    FCNN = FullyConnectedNet(input_d, output_d, layer_dim=layer_d, num_layers=4)

    graph = ComputationGraph.from_param(mean_squared_loss(FCNN.apply(x), y, "loss"), store_full_graph=True)
    graph.summary()

    for i in range(epochs):
        loss = mean_squared_loss(FCNN.apply(x), y, "mean_squared_loss")
        SimpleOptimizer.optimize(loss, learning_rate=2e-3)
        if not (i % 100):
            print(f"Loss: {loss}")

x = np.random.random((3))
y = np.array((0,1))

xb = np.random.random((4, 3))
yb = np.array(((0,1),(0,1),(1,0),(1,0)))

test(x,y)
#test(xb, yb)
