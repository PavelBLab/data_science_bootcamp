# Import Relevant Libraries
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


# Generate Random input data to train on
observations = 1000
xs = np.random.uniform(low=-10, high=10, size=(observations, 1))    # size => vector row x columns => rows = 1000 and columns = 1
xz = np.random.uniform(-10, 10, size=(observations, 1))
print(xs.shape, xz.shape)

input = np.column_stack((xs, xz))
print(input.shape)

noise = np.random.uniform(-1, 1, size=(observations, 1))
targets = 2 * xs -3 * xz + 5 + noise
print(targets.shape)


# Initialize variables
init_range = 0.1
weights = np.random.uniform(-init_range, init_range, size=(2, 1))
bias = np.random.uniform(-init_range, init_range, size=1)
print(weights, bias)

# Set a learning rate
learning_rate = 0.03

# Train the model
for i in range(1_000):
    outputs = np.dot(input, weights) + bias
    deltas = outputs - targets

    loss = np.sum(deltas ** 2) / 2 / observations

    # print(loss)

    deltas_scaled = deltas / observations
    weights = weights - learning_rate * np.dot(input.T, deltas_scaled)
    bias = bias - learning_rate * np.sum(deltas_scaled)
    print('weights', weights)
    print('bias', bias)

print(weights, bias)







