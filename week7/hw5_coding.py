import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize
import os

# Problem 1
## Part (a)
M = np.genfromtxt(os.path.join(os.path.dirname(__file__), 'CO2_data.csv'), delimiter=',')

t = M[:, 0]
CO2 = M[:, 1]

A1 = t
A2 = CO2

print('A1 = ', A1)
print('A2 = ', A2)

## Part (b)
def sumSquaredError(a, b, r):
   y = lambda t: a + b*np.exp(r*t)
   error = sum((y(t) - CO2)**2)
   return error

A3 = sumSquaredError(300, 30, 0.03)
print('A3 = ', A3)

## Part (c)
adapter = lambda p: sumSquaredError(p[0], p[1], p[2])

guess = np.array([300, 30, 0.03])

optimal_params = scipy.optimize.fmin(adapter, guess)
A4 = optimal_params
print('A4 = ', A4)

## (d)
sumSquaredError = adapter(optimal_params)
A5 = sumSquaredError
print('A5 = ', A5)

## (e)
def maxError(a, b, r):
   y = lambda t: a + b*np.exp(r*t)
   error = np.amax(np.abs(y(t) - CO2))
   return error

A6 = maxError(300, 30, 0.03)
print('A6 = ', A6)

adapter = lambda p: maxError(p[0], p[1], p[2])
guess = np.array([300, 30, 0.03])
optimal_params = scipy.optimize.fmin(adapter, guess, maxiter=2000)
A7 = optimal_params
print('A7 = ', A7)

## (f)
def sumSquaredError(a, b, r, c, d, e):
    y = lambda t: a + b*np.exp(r*t) + c*np.sin(d * (t-e))
    error = sum((y(t) - CO2)**2)
    return error

A8 = sumSquaredError(300, 30, 0.03, -5, 4, 0)
print('A8 = ', A8)

## (g)
adapter = lambda p: sumSquaredError(p[0], p[1], p[2], p[3], p[4], p[5])
guess = np.append(A4, [-5, 4, 0])

optimal_params = scipy.optimize.fmin(adapter, guess, maxiter=2000)
A9 = optimal_params
print('A9 = ', A9)

## (h)
sumSquaredError = adapter(optimal_params)
A10 = sumSquaredError
print('A10 = ', A10)

# Problem 2
## Part (a)
M = np.genfromtxt(os.path.join(os.path.dirname(__file__), 'salmon_data.csv'), delimiter=',')

year = M[:,0] #Assign the 'year' array to the first column of the data
salmon = M[:,1] #Assign the 'salmon' array to the first column of the data

## (b) - Degree-1 polynomial
A11 = np.polyfit(year, salmon, 1)

## (c) - Degree-3 polynomial
A12 = np.polyfit(year, salmon, 3)

## (d) - Degree-5 polynomial
A13 = np.polyfit(year, salmon, 5)

## (e) - compare to exact number of salmon
exact =  752638 # The exact number of salmon

p1 = np.polyval(A11, 2022)
p3 = np.polyval(A12, 2022)
p5 = np.polyval(A13, 2022)

err1 = np.abs(p1 - exact) / exact
err2 = np.abs(p3 - exact) / exact
err3 = np.abs(p5 - exact) / exact

A14 = np.array([err1, err2, err3])
print('A14 = ', A14)
