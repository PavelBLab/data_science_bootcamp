import numpy as np

m1 = np.array([[5, 12, 6], [-3, 0, 14]])
m2 = np.array([[7, 10, -88], [8, 5, -1]])

# Tensor is a collection of matrixes
t = np.array([m1, m2])
print(t)
print(t.shape)

# Addition/Substration/Multiplication/Division of Matrixes
print(m1 + m2)
print(m1 - m2)
print(m1 * m2)
print(m1 / m2)