# This is a simple implementation of a neural network to solve the XOR problem using numpy.
# The XOR problem is a classic problem in machine learning and neural networks.

import numpy as np

# Instlize the weights
W1 = np.random.uniform(-1, 1, (2, 2))
W2 = np.random.uniform(-1, 1, (2, 1))

# Initialize the biases
b1 = np.zeros((1, 2))
b2 = np.zeros((1, 1))

# Define the sigmoid activation function
def sigmoid(x):
    return 1 / (1 + np.exp(-x))
# Define the derivative of the sigmoid function
def sigmoid_derivative(x):
    return x * (1 - x)

# training data
#input
X = np.array([[0, 0],
              [0, 1],
              [1, 0],
              [1, 1]])
#output
Y = np.array([[0],
              [1],
              [1],
              [0]])

