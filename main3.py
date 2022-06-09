import numpy as np
import nnfs
from nnfs.datasets import spiral_data

nnfs.init()

X = [
    [1, 2, 3, 2.5],
    [2.0, 5.0, -1.0, 2.0],
    [-1.5, 2.7, 3.3, -0.8]
]

X, y = spiral_data(100, 3)

class DenseLayer:
    def __init__(self, nInputs, nNeurons):
        self.weights = np.random.randn(nInputs, nNeurons)
        self.biases = np.zeros((1, nNeurons))
    def forward(self, inputs):
        self.output = np.dot(inputs, self.weights) + self.biases

class ActivationReLU:
    def forward(self, inputs):
        self.output = np.maximum(0, inputs)

layer1 = DenseLayer(2, 5)
activation1 = ActivationReLU()

layer1.forward(X)
activation1.forward(layer1.output)

print(layer1.output)