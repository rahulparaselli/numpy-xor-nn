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
    return sigmoid(x) * (1 - sigmoid(x))

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

# instlize the learning rate
learning_rate = 0.1
# number of epochs
epochs = 10000

# Training the neural network
for _ in range(epochs):
    # Forward propagation
    # Calculate the hidden layer output
    hidden_input = np.dot(X, W1) + b1
    hidden_output = sigmoid(hidden_input)

    # Calculate the final output
    output_input = np.dot(hidden_output, W2) + b2
    output = sigmoid(output_input)

    # Calculate the error
    error = Y - output

    # Backpropagation
    d_output = error * sigmoid_derivative(output)
    error_hidden = d_output.dot(W2.T)
    d_hidden = error_hidden * sigmoid_derivative(hidden_output)

    # Update the weights and biases
    W2 += hidden_output.T.dot(d_output) * learning_rate
    W1 += X.T.dot(d_hidden) * learning_rate
    b1 += np.sum(d_hidden, axis=0) * learning_rate
    b2 += np.sum(d_output, axis=0) * learning_rate

hidden = sigmoid(np.dot(X, W1) + b1)
output = sigmoid(np.dot(hidden, W2) + b2)
# Print the final output
print("Final output after training:")
print(output)

def predict(input_data):
    hidden = sigmoid(np.dot(input_data, W1) + b1)
    output = sigmoid(np.dot(hidden, W2) + b2)
    return np.round(output) , output


input_data = np.array([[1, 1]])
# Test the neural network
print("Predictions:")
pred , prob = predict(input_data)
print(f"Input: {input_data} -> Predicted: {pred}, Probability: {prob}")