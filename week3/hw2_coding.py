import numpy as np
import matplotlib.pyplot as plt

# Problem 1
sequence = []
for n in range(1, 33):
    p = n * (n + 1) * (2 * n + 1) / 6
    sequence.append(p)

A1 = np.array(sequence)

# Problem 2
## Part a
y1, y2, y3, y4 = 0, 0, 0, 0
term1, term2, term3, term4 = 0.1, 0.1, 0.25, 0.5

for k in range(100000):
    y1 += term1

for k in range(100000000):
    y2 += term2

for k in range(100000000):
    y3 += term3

for k in range(100000000):
    y4 += term4

A2, A3, A4, A5 = y1, y2, y3, y4

## Part b
x1 = np.abs(10000 - y1)
x2 = np.abs(y2 - 10000000)
x3 = np.abs(25000000 - y3)
x4 = np.abs(y4 - 50000000)

A6, A7, A8, A9 = x1, x2, x3, x4

# Problem 3
## Part a
Fibonacci = np.zeros(200)

## Part b
Fibonacci[0], Fibonacci[1] = 1, 1

## Part c
for k in range(200):
    if k <= 1:
        continue
    else:
        fibonacci_k = Fibonacci[k-1] + Fibonacci[k-2]
        if fibonacci_k < 1000000:
            Fibonacci[k] = fibonacci_k
        else:
            break

A10 = Fibonacci

## Part d
N, fibonacci_n = 0, 0
for k in range(200):
    if Fibonacci[k] > fibonacci_n:
        fibonacci_n = Fibonacci[k]
        N = k

A11 = N

## Part e
A12 = Fibonacci[:N+1]

# Problem 4
## Part a
x = np.linspace(-np.pi, np.pi, 100)
A13 = x

## Part b
Taylor = 0 * x

for k in range(4):
    Taylor += (-1) ** k * x ** (2*k) / np.math.factorial(2*k)

A14 = Taylor
