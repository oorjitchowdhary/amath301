import numpy as np
import matplotlib.pyplot as plt

## Problem 1
# Part a
x = 3.1
A1 = x

# Part b
y = -29
A2 = y

# Part c
z = 9 * np.e
A3 = z

# Part d
w = np.exp(4)
A4 = w

# Part e
A5 = np.sin(np.pi)


## Problem 2
x = np.linspace(0, 1, 5)
x = np.pi * x
A6 = np.cos(x)


## Problem 3
# Part a
u = np.linspace(3, 4, 6)
A7 = u

# Part b
v = np.arange(0, 4, 0.75)
A8 = v

# Part c
w = v + (2 * u)
A9 = w

# Part d
w = u ** 3
A10 = w

# Part e
x = np.tan(u) + np.exp(v)
A11 = x

# Part f
A12 = u[2]


## Problem 4
# Part a
z = np.arange(-6, 3 + 1/100, 1/100)
A13 = z
print(A13)

# Part b
temp = np.arange(0, len(z), 2)
A14 = z[temp]
print(A14)

# Part c
A15 = z[::3]
print(A15)

# Part d
A16 = z[-5:]
print(A16)