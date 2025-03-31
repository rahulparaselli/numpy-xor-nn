# NumPy XOR Neural Network

A simple implementation of a neural network to solve the XOR problem using only NumPy.

## Project Overview

This project demonstrates how to build a basic neural network from scratch using NumPy to solve the XOR logical operation. The XOR (exclusive OR) problem is a classic example used in neural network education because it's not linearly separable, meaning it requires a hidden layer to solve.

### What is XOR?

XOR (exclusive OR) returns true only when exactly one of the inputs is true:

| Input A | Input B | Output |
|---------|---------|--------|
| 0       | 0       | 0      |
| 0       | 1       | 1      |
| 1       | 0       | 1      |
| 1       | 1       | 0      |

## Neural Network Architecture

```
Input Layer       Hidden Layer (2 neurons)      Output Layer
   (x1) ______________[h1]____________________  
             w11      /   \      w21           \ 
                    /      \                    [y]
   (x2) _________[h2]______\______w22_________/
             w12      \      /      w23
                       \____/
```

The network consists of:
- Input layer: 2 neurons (for the 2 binary inputs)
- Hidden layer: 2 neurons with sigmoid activation
- Output layer: 1 neuron with sigmoid activation

## Implementation

### Prerequisites

- Python 3.x
- NumPy library

```bash
pip install numpy
```

### Key Components

#### 1. Neural Network Setup

```python
import numpy as np

# Initialize the weights randomly
W1 = np.random.uniform(-1, 1, (2, 2))  # Weights between input and hidden layer
W2 = np.random.uniform(-1, 1, (2, 1))  # Weights between hidden and output layer

# Initialize the biases
b1 = np.zeros((1, 2))  # Biases for hidden layer
b2 = np.zeros((1, 1))  # Bias for output layer
```

#### 2. Activation Functions

```python
# Sigmoid activation function
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# Derivative of sigmoid for backpropagation
def sigmoid_derivative(x):
    return sigmoid(x) * (1 - sigmoid(x))
```

#### 3. Training Data

```python
# Input data - all possible XOR combinations
X = np.array([[0, 0],
              [0, 1],
              [1, 0],
              [1, 1]])

# Target outputs for XOR
Y = np.array([[0],
              [1],
              [1],
              [0]])
```

## Training Process

The training process uses backpropagation to update weights and biases:

1. **Forward Propagation**: Calculate outputs through the network
2. **Error Calculation**: Compare outputs with expected results
3. **Backpropagation**: Calculate gradients and update weights
4. **Repeat**: Perform steps 1-3 for multiple epochs

```python
# Training hyperparameters
learning_rate = 0.1
epochs = 10000

# Training loop
for _ in range(epochs):
    # Forward propagation
    hidden_input = np.dot(X, W1) + b1
    hidden_output = sigmoid(hidden_input)
    output_input = np.dot(hidden_output, W2) + b2
    output = sigmoid(output_input)

    # Calculate error
    error = Y - output

    # Backpropagation
    d_output = error * sigmoid_derivative(output)
    error_hidden = d_output.dot(W2.T)
    d_hidden = error_hidden * sigmoid_derivative(hidden_output)

    # Update weights and biases
    W2 += hidden_output.T.dot(d_output) * learning_rate
    W1 += X.T.dot(d_hidden) * learning_rate
    b1 += np.sum(d_hidden, axis=0) * learning_rate
    b2 += np.sum(d_output, axis=0) * learning_rate
```

## Making Predictions

After training, the network can make predictions on new inputs:

```python
def predict(input_data):
    hidden = sigmoid(np.dot(input_data, W1) + b1)
    output = sigmoid(np.dot(hidden, W2) + b2)
    return np.round(output), output  # Returns both binary prediction and probability

# Example prediction
input_data = np.array([[1, 1]])
pred, prob = predict(input_data)
print(f"Input: {input_data} -> Predicted: {pred}, Probability: {prob}")
```

## How It Works - The Backpropagation Algorithm

```
                  Forward Pass                 |             Backward Pass
                                               |
Input → Hidden Layer → Output Layer → Loss     |     Loss → Output Gradients → Hidden Gradients
                                               |
                                        Weight/Bias Updates
```

1. **Forward Pass**: Data flows through the network, with each layer applying weights, biases, and activation functions
2. **Loss Calculation**: Compare the network's output to the target
3. **Backward Pass**: Calculate how each weight contributes to the error
4. **Weight Update**: Adjust weights in the direction that reduces error

## Results

After training, the network will produce outputs very close to the expected XOR truth table. The final output should approximate:

```
Input [0,0] → Output ~0
Input [0,1] → Output ~1
Input [1,0] → Output ~1
Input [1,1] → Output ~0
```

## Why XOR Needs a Hidden Layer

XOR is not linearly separable, meaning no single line can separate the data points into the correct categories:

```
    1 │  1       0    │
      │               │
    0 │  O       1    │
      │               │
      └───────────────┘
        0       1
        
XOR Truth Table:
(0,0) → 0   (0,1) → 1
(1,0) → 1   (1,1) → 0
```

**Simple Explanation**: 
- Imagine you need to draw a single straight line on a piece of paper to separate all the 1s from all the 0s in XOR.
- It's impossible! Any straight line you draw will always have some misclassified points.
- A hidden layer allows the network to create a more complex boundary (like a curved line or multiple lines) that can correctly separate the XOR pattern.
- This is why simple perceptrons (networks without hidden layers) cannot solve XOR.
- The hidden layer essentially transforms the input space into a new representation where the problem becomes linearly separable.

## Extending This Project

You can extend this project by:
- Adding more hidden layers
- Using different activation functions
- Implementing mini-batch training
- Visualizing the decision boundary
- Adding regularization to prevent overfitting

## Credits

- **Code:** Rahul Paraselli
- **Documentation:** GitHub Copilot

## License

This project is open source and available for educational purposes.
