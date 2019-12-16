import numpy as np

# Scaler
s = 5

# Vector
v = np.array([5, -2, 4])
print(v)

# Matrices
m = np.array([[5, 12, 6], [-3, 0, 14]])
print(m)

# Data shape
print(v.shape)
print(m.shape)

# Respape
print(v.reshape(1, 3))
print(v.reshape(3, 1))