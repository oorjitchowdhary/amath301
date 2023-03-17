import numpy as np
import scipy.linalg

A = np.array([[55, -10, -20], [-10, 30, -15], [-20, -15, 65]])
b = np.array([20, 10, 5])

x = scipy.linalg.solve(A, b)
print(x)

A = np.array([[0, 808, -1, -1, 832], [808, 0, 382, -1, 736], [-1, 382, 0, 270, -1], [-1, -1, 270, 0, 421], [832, 736, -1, 421, 0]])

E = np.array([[1, 1, 0], [1, 2, 808], [1, 3, -1], [1, 4, -1], [1, 5, 832], [2, 1, 808], [2, 2, 0], [2, 3, 382], [2, 4, -1], [2, 5, 736], [3, 1, -1], [3, 2, 382], [3, 3, 0], [3, 4, 270], [3, 5, -1], [4, 1, -1], [4, 2, -1], [4, 3, 270], [4, 4, 0], [4, 5, 421], [5, 1, 832], [5, 2, 736], [5, 3, -1], [5, 4, 421], [5, 5, 0]])