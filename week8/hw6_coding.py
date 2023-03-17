import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize

# Problem 1
P = lambda p: p*(1-p)*(p-1/2)

## Part a - Solve the ODE with forward Euler
dt = 0.5
tspan = np.arange(0, 10+dt, dt)

p0 = 0.9
p = np.zeros(len(tspan))
p[0] = p0

for k in range(len(tspan)-1):
    p[k+1] = p[k] + dt*P(p[k])

A1 = p
print("A1:", A1)

## Part b - Solve using backward Euler
dt = 0.5
tspan = np.arange(0, 10+dt, dt)

p0 = 0.9
p = np.zeros(len(tspan))
p[0] = p0

for k in range(len(p)-1):
    g = lambda z: z - p[k] - dt*P(z)
    p[k+1] = scipy.optimize.fsolve(g, p[k])

A2 = p
print("A2:", A2)

## Part c - Solve using the midpoint method
dt = 0.5
tspan = np.arange(0, 10+dt, dt)

p0 = 0.9
p = np.zeros(len(tspan))
p[0] = p0

for k in range(len(p)-1):
    k1 = P(p[k])
    k2 = P(p[k] + 0.5*dt*k1)
    p[k+1] = p[k] + dt*k2

A3 = p
print("A3:", A3)


## Part d - Solve with RK4
dt = 0.5
tspan = np.arange(0, 10+dt, dt)

p0 = 0.9
p = np.zeros(len(tspan))
p[0] = p0

for k in range(len(p)-1):
    k1 = P(p[k])
    k2 = P(p[k] + 0.5*dt*k1)
    k3 = P(p[k] + 0.5*dt*k2)
    k4 = P(p[k] + dt*k3)
    p[k+1] = p[k] + dt/6*(k1 + 2*k2 + 2*k3 + k4)

A4 = p
print("A4:", A4)


# Problem 2
## Part (a)
a = 1/2
Rprime = lambda R, J: a*R + J
Jprime = lambda R, J: -R - a*J

ode = lambda t, x: np.array([Rprime(x[0], x[1]), Jprime(x[0], x[1])])

x0 = np.array([2, 1])

sol = scipy.integrate.solve_ivp(ode, [0, 20], x0)
print(sol)

t = sol.t

R = sol.y[0, :]
A5 = R
print("A5:", A5)

J = sol.y[1, :]
A6 = J
print("A6:", A6)

## (b) 
A7 = np.array([R[-1], J[-1]])

## (c) 
dt = 0.1
trange = np.arange(0, 20+dt, dt)

R = np.zeros(len(trange))
J = np.zeros(len(trange))

R[0] = 2
J[0] = 1

for k in range(len(trange)-1):
    R[k+1] = R[k] + dt*Rprime(R[k], J[k])
    J[k+1] = J[k] + dt*Jprime(R[k], J[k])

A8 = R
print("A8:", A8)

A9 = J
print("A9:", A9)

## (d) 
A10 = np.array([R[-1], J[-1]])
print("A10:", A10)

## (e) 
A11 = np.linalg.norm(A7 - A10)
print("A11:", A11)


# Problem 3
g = 9.8
L = 11
sigma = 0.12

## Part a
theta_prime = lambda theta, v: v
v_prime = lambda theta, v: -g/L*np.sin(theta) - sigma*v

x0 = np.array([-np.pi/8, -0.1])

## Part b
odefun = lambda t, p: np.array([theta_prime(p[0], p[1]), v_prime(p[0], p[1])])

A12 = odefun(1, np.array([2, 3]))
print("A12:", A12)

## Part c
sol = scipy.integrate.solve_ivp(odefun, [0, 50], x0)
print(sol)

A13 = sol.y
print("A13:", A13)