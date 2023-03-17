import numpy as np
import matplotlib.pyplot as plt
import scipy.integrate
import os

# Problem 1
M = np.genfromtxt(os.path.join(os.path.dirname(__file__), 'Plutonium.csv'), delimiter=',')

t = M[0, :]
P = M[1, :]

## Part a
h = t[1] - t[0]
A1 = h

## Part b
deriv_t0 = (P[0+int(h)] - P[0]) / h
A2 = deriv_t0

## Part c
deriv_t40 = (P[40] - P[40-int(h)]) / h
A3 = deriv_t40

## Part d
A4 = (-3*P[0] + 4*P[1] - P[2])/(2*h)

## Part e
A5 = (3 * P[40] - 4 * P[39] + P[38]) / (2 * h)

## Part f
dP = []
dP.append(A4)
for i in range(1, 40):
    dP.append((P[i+1] - P[i-1]) / (2*h))

dP.append(A5)
A6 = np.array(dP)

## Part g
decay_rate = []
for i in range(len(A6)):
    decay_rate.append(-1/P[i] * A6[i])

A7 = np.array(decay_rate)

## Part h
A8 = np.average(decay_rate)

## Part i
half_life = np.log(2) / A8
A9 = half_life

## Part j
second_deriv = (-2 * P[22] + P[21] + P[23]) / (h**2)
A10 = second_deriv

# Problem 2
mu = 85
sigma = 8.3
integrand = lambda x: np.exp(-(x-mu)**2/(2*sigma**2))/np.sqrt(2*np.pi*sigma**2)
left = 110
right = 130

## Part a - true value
Int, err = scipy.integrate.quad(integrand, left, right)
A11 = Int

## Part b - left hand rule approximation
LHR = []

power = -np.linspace(1, 16, 16)
h = 2 ** power

for dx in h:
    x = np.arange(left, right + dx, dx)
    y = integrand(x)
    S_approx = np.sum(y[:-1]) * dx
    LHR.append(S_approx)

A12 = np.array(LHR)

## Part c - right hand rule approximation
RHR = []

for dx in h:
    x = np.arange(left, right + dx, dx)
    y = integrand(x)
    S_approx = np.sum(y[1:]) * dx
    RHR.append(S_approx)

A13 = np.array(RHR)

## Part d - midpoint rule approximation
midpoint_approx = []

for dx in h:
    x = np.arange(left, right + dx, dx)
    y = integrand(x)
    MPR = 0
    for i in range(x.size - 1):
        MPR += integrand((x[i] + x[i+1])/2) * dx
    midpoint_approx.append(MPR)

A14 = np.array(midpoint_approx)

## Part e - trapezoid rule approximation
A15 = (A12 + A13) / 2

## Part f - Simpson's rule approximation
Simpson_approx = []

for dx in h:
    x = np.arange(left, right + dx, dx)
    y = integrand(x)
    simpson = (dx / 3)*(y[0] + 4*sum(y[1:-1:2]) + 2*sum(y[2:-2:2]) + y[-1])
    Simpson_approx.append(simpson)

A16 = np.array(Simpson_approx)
