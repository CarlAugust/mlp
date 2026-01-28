import numpy as np

class layer:
     
    def __init__(self, layer_size, output_size):
        self.activation = np.zeros((layer_size))
        self.bias = np.random.uniform(-1, 1, (output_size))
        self.weights = np.random.uniform(-1, 1, (output_size, layer_size))


    def feed_activation(self, activations):
        assert(activations.shape == self.activation.shape)
        self.activation = activations


    
# a = np.array([3,2,3,2,4])
# w = np.array([
        # [3,2,1],
        # [1,1,1],
        # [4,2,3],
        # [1,2,3],
        # [1,5,3]
    # ])
# b = np.array([1,2,3])

# print(a @ w + b)

l = layer(4, 3)
l.feed_activation(np.array([1,2,3,4]))

print(l.weights @ l.activation + l.bias)