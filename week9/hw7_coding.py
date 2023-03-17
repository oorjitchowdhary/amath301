import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize
import scipy.integrate
import time


# Problem 1
dydt = lambda t, y: 5e5 * (-y + np.sin(t))
tspan = np.linspace(0, 2 * np.pi, 100)
y0 = 0
dt = tspan[1] - tspan[0]

## Part (a)
g = lambda z: z - y0 - 0.5 * dt * (dydt(tspan[1], z) + dydt(tspan[0], y0))
A1 = g(3)
print("A1:", A1)

## Part (b)
y = np.zeros(len(tspan))
y[0] = y0

for k in range(len(y) - 1):
    g = lambda z: z - y[k] - 0.5 * dt * (dydt(tspan[k+1], z) + dydt(tspan[k], y[k]))
    y[k+1] = scipy.optimize.fsolve(g, y[k])

A2 = y
print("A2:", A2)


# Problem 2
## Part (a)
s = 77.27
w = 0.161
q = 1

y1_prime = lambda y1, y2, y3: s*(y2 - y1*y2 + y1 - q*y1**2)
y2_prime = lambda y1, y2, y3: 1/s* (-y2 - y1 * y2 + y3)
y3_prime = lambda y1, y2, y3: w * (y1 - y3)

y1_0 = 1
y2_0 = 2
y3_0 = 3

odefun = lambda t, y: np.array([y1_prime(y[0], y[1], y[2]), y2_prime(y[0], y[1], y[2]), y3_prime(y[0], y[1], y[2])])

A3 = odefun(1, np.array([2, 3, 4]))
print("A3:", A3)

## (b) Solve for 10 logarithmically spaced points, using RK45
y_rk45 = np.zeros([3, 10])
y0 = np.array([y1_0, y2_0, y3_0])

qspan = np.logspace(0, -5, 10)

for index, q in enumerate(qspan):
    y1_prime = lambda y1, y2, y3: s*(y2 - y1*y2 + y1 - q*y1**2)
    odefun = lambda t, y: np.array([y1_prime(y[0], y[1], y[2]), y2_prime(y[0], y[1], y[2]), y3_prime(y[0], y[1], y[2])])

    sol = scipy.integrate.solve_ivp(odefun, [0, 30], y0)
    y_rk45[:, index] = sol.y[:, -1]

A4 = y_rk45
print("A4:", A4)

## (c)  Solve for 10 logarithmically spaced points, using BDF
y_bdf = np.zeros([3, 10])
y0 = np.array([y1_0, y2_0, y3_0])

qspan = np.logspace(0, -5, 10)

for index, q in enumerate(qspan):
    y1_prime = lambda y1, y2, y3: s*(y2 - y1*y2 + y1 - q*y1**2)
    odefun = lambda t, y: np.array([y1_prime(y[0], y[1], y[2]), y2_prime(y[0], y[1], y[2]), y3_prime(y[0], y[1], y[2])])

    sol = scipy.integrate.solve_ivp(odefun, [0, 30], y0, method="BDF")
    y_bdf[:, index] = sol.y[:, -1]

A5 = y_bdf
print("A5:", A5)


# Problem 3
mu = 200
x0 = 2
y0 = 0

## Part (a)
dxdt = lambda x, y: y
dydt = lambda x, y: mu * (1 - x**2) * y - x

ode = lambda t, z: np.array([dxdt(z[0], z[1]), dydt(z[0], z[1])])

A6 = dxdt(2, 3)
print("A6:", A6)

A7 = dydt(2, 3)
print("A7:", A7)

## Part (b)
z0 = np.array([x0, y0])
sol = scipy.integrate.solve_ivp(ode, [0, 400], z0)
print(sol.y)
print(sol.y.shape)

A8 = sol.y[0, :]
print("A8:", A8)

## Part (c)
x0 = 2
y0 = 0
z0 = np.array([x0, y0])

sol_bdf = scipy.integrate.solve_ivp(ode, [0, 400], z0, method="BDF")
print(sol_bdf.y)
print(sol_bdf.y.shape)

A9 = sol_bdf.y[0, :]
print("A9:", A9)

## Part (d)
A10 = sol.y.shape[1] / sol_bdf.y.shape[1]
print("A10:", A10)

## Part (e)
dxdt = lambda x, y: y
dydt = lambda x, y: mu * y - x

A11 = dxdt(2, 3)
print("A11:", A11)

A12 = dydt(2, 3)
print("A12:", A12)

## Part f - linear system
A = np.array([[0, 1], [-1, mu]])

A13 = A
print("A13:", A13)

## Part g
dt = 0.01
tspan = np.arange(0, 400+dt, dt)

X = np.zeros([2, len(tspan)])
X[:, 0] = np.array([x0, y0])

for k in range(len(tspan) - 1):
    X[:, k+1] = X[:, k] + dt * A @ X[:, k]

A14 = X
print("A14:", A14)

## Part (i)
I = np.eye(2)
# To create C, we can just do subtraction and multiplication
C = I - dt * A

A15 = C
print("A15:", A15)

N = int(400/dt)
X = np.zeros([N+1, 2])
X[0, :] = np.array([x0, y0])

for k in range(N):
    X[k+1, :] = np.linalg.solve(C, X[k, :])

A16 = X[:, 0]
print("A16:", A16)

# A16 = np.linalg.solve(C, X)
# print("A16:", A16)
