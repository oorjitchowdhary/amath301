import numpy as np
import matplotlib.pyplot as plt
import cv2
import os

# Problem 1
## Part a
R = lambda theta: np.array([[np.cos(theta), -np.sin(theta), 0], [np.sin(theta), np.cos(theta), 0], [0, 0, 1]])
v = lambda x, y, z: np.array([x, y, z])
w = lambda theta, x, y, z: R(theta) @ v(x, y, z)

A1 = R(np.pi/4)
print("A1:", A1)

## Part b
y = np.array([3, np.pi, 4])
x = np.linalg.solve(A1, y)
A2 = x.reshape(3, 1)
print("A2:", A2)

# Problem 2
## Part (a) - Think about how the matrix equation is setup!
x = lambda F1, F2, F3, F4, F5, F6, F7, F8, F9, F10, F11, F12, F13, F14, F15, F16, F17, F18, F19: np.array([F1, F2, F3, F4, F5, F6, F7, F8, F9, F10, F11, F12, F13, F14, F15, F16, F17, F18, F19])

## Part (b) - Write the matrix-vector equation
W8, W9, W10, W11 = 12000, 9200, 9000, 19200

s = 1 / np.sqrt(17) 
AA = np.zeros([19,19]) 
b = np.zeros([19, 1]) 

# Equation 1
j = 1-1 # because python indexing begins at 0, F1 is actually F[0]
AA[j, 1-1] = -s # Coefficient of F1
AA[j, 2-1] = 1  # Coefficient of F2
AA[j, 12-1] = s # Coefficient of F12

# Equation 2
j = 2-1  # because python indexing begins at 0, F[1] is actually F[0]
AA[j, 1-1] = -4 * s # Coefficient of F1
AA[j, 12-1] = -4 * s # Coefficient of F12

# Equation 3
j = 3-1 
AA[j, 2-1] = -1
AA[j, 3-1] = 1
AA[j, 13-1] = -s
AA[j, 14-1] = s

# Equation 4
j = 4-1 
AA[j, 13-1] = -4 * s
AA[j, 14-1] = -4 * s

# Equation 5
j = 5-1 
AA[j, 3-1] = -1
AA[j, 4-1] = 1
AA[j, 15-1] = -s
AA[j, 16-1] = s

# Equation 6
j = 6-1 
AA[j, 15-1] = -4 * s
AA[j, 16-1] = -4 * s

# Equation 7
j = 7-1 
AA[j, 4-1] = -1
AA[j, 5-1] = 1
AA[j, 17-1] = -s
AA[j, 18-1] = s

# Equation 8
j = 8-1 
AA[j, 17-1] = -4 * s
AA[j, 18-1] = -4 * s

# Equation 9
j = 9-1 
AA[j, 5-1] = -1
AA[j, 6-1] = s
AA[j, 19-1] = -s

# Equation 10
j = 10-1 
AA[j, 6-1] = -4 * s
AA[j, 19-1] = -4 * s

# Equation 11
j = 11-1 
AA[j, 6-1] = -s
AA[j, 7-1] = -1

# Equation 12
j = 12-1 
AA[j, 7-1] = 1
AA[j, 8-1] = -1
AA[j, 18-1] = -s
AA[j, 19-1] = s

# Equation 13
j = 13-1 
AA[j, 18-1] = 4 * s
AA[j, 19-1] = 4 * s
b[j] = W8 

# Equation 14
j = 14-1 
AA[j, 8-1] = 1
AA[j, 9-1] = -1
AA[j, 16-1] = -s
AA[j, 17-1] = s

# Equation 15
j = 15-1 
AA[j, 16-1] = 4 * s
AA[j, 17-1] = 4 * s
b[j] = W9

# Equation 16
j = 16-1 
AA[j, 9-1] = 1
AA[j, 10-1] = -1
AA[j, 14-1] = -s
AA[j, 15-1] = s

# Equation 17
j = 17-1 
AA[j, 14-1] = 4 * s
AA[j, 15-1] = 4 * s
b[j] = W10

# Equation 18
j = 18-1 
AA[j, 10-1] = 1
AA[j, 11-1] = -1
AA[j, 12-1] = -s
AA[j, 13-1] = s

# Equation 19
j = 19-1 
AA[j, 12-1] = 4 * s
AA[j, 13-1] = 4 * s
b[j] = W11

## Part c - save A
A3 = AA
print("A3:", A3)

## Part d - Find the forces
x = np.linalg.solve(AA, b)
A4 = x.reshape(19, 1)
print("A4:", A4)

## Part e - Find the largest force
# Make sure you have the order correct: we want the largest OF the absolute
# values!
A5 = np.amax(np.abs(A4))
print("A5:", A5)

## Part f - We now need to loop somehow. Once one of the forces exceeds 44000
# Newtons, the bridge collapses. It *can* be exactly 44000 Newtons without
# collapsing.
# You will need a break statement if the bridge collapses!

x = np.linalg.solve(AA, b).reshape(19, 1)
while np.amax(np.abs(x)) <= 44000:
    W10 += 5
    b[16] = W10
    x = np.linalg.solve(AA, b).reshape(19, 1)

A6 = W10
print("A6:", A6)

A7 = np.argmax(np.amax(np.abs(x))) + 1
print("A7:", A7)


# Problem 3
## Load in the image of beautiful Olive
A = cv2.imread(os.path.join(os.path.dirname(__file__), 'olive.jpg'), 0)

## Part a
A8 = A.shape[0] * A.shape[1]
print("A8:", A8)

## Part b
U, S, Vt = np.linalg.svd(A, full_matrices=False) # Vt = V transpose

A9 = U
print("A9:", A9)

A10 = S
print("A10:", A10)

A11 = Vt.T
print("A11:", A11)

## Part c
A12 = S[:15]
print("A12:", A12)

## Part d
total_energy = np.sum(S)
A13 = np.amax(S) / total_energy
print("A13:", A13)

## Part (d) - Calculate the proportion of energy in the rank-15 approximation
A14 = np.sum(A12) / total_energy
print("A14:", A14)

## Part (e) - Find the rank-r approximation that holds 75% of the total energy.
# Use np.where! Don't hard code this!
rank_r_approx = 0
r = 0

while rank_r_approx < 0.75:
    r += 1
    rank_r_approx = np.sum(S[:r]) / total_energy

A15 = r
print("A15:", A15)