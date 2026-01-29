import numpy as np

class Layer:
    def __init__(self, layer_size, output_size):
        self.activation = np.zeros((layer_size))
        self.bias = np.random.uniform(-1, 1, (output_size))
        self.weights = np.random.uniform(-1, 1, (output_size, layer_size))

    def feed_activation(self, activations):
        assert(activations.shape == self.activation.shape)
        self.activation = activations

    def forward(self):
        return self.weights @ self.activation + self.bias
