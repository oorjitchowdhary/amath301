import numpy as np
import matplotlib.pyplot as plt
import scipy.integrate
import time

### Define the Forward-Euler function
def forward_euler(odefun, tspan, y0):
    # Forward Euler method
    # Solves the differential equation y' = f(t,y) at the times
    # specified by the vector tspan and with initial condition y0.
    # - odefun is an anonymous function of the form odefun = lambda t,v: ...
    # - tspan is a 1D array
    # - y0 is a number

    dt = tspan[1] - tspan[0]  # calculate dt from t values
    y = np.zeros(len(tspan))  # Create array of same length as tspan
    y[0] = y0  # Set initial condition
    for k in range(len(y) - 1):
        y[k + 1] = y[k] + dt * odefun(tspan[k], y[k]) # Forward Euler step

    return tspan, y

## Test the Forward-Euler Function
dydt = lambda t, y: 0.3*y + t
tans, yans = forward_euler(dydt, np.arange(0, 1+0.1, 0.1), 1)
print(yans[-1])

#### Define the Backward-Euler function
def backward_euler(odefun, tspan, y0):
    # Backward Euler method
    # Solves the differential equation y' = f(t,y) at the times
    # specified by the vector tspan and with initial condition y0.
    # - odefun is an anonymous function of the form odefun = lambda t,v: ...
    # - tspan is a 1D array
    # - y0 is a number
    
    dt = tspan[1] - tspan[0]  # calculate dt from t values
    y = np.zeros(len(tspan))  # Create array of same length as tspan
    y[0] = y0  # Set initial condition
    
    for k in range(len(y) - 1):
        g = lambda z: z - y[k] - dt * odefun(tspan[k + 1], z)
        y[k + 1] = scipy.optimize.fsolve(g, y[k])  # Backward Euler step


    return tspan, y

## Test Backward-Euler here
dydt = lambda t, y: 0.3*y + t
tans, yans = backward_euler(dydt, np.arange(0, 1+0.1, 0.1), 1)
print(yans[-1])

#### Define the midpoint method
def midpoint(odefun, tspan, y0):
    # Midpoint method
    # Solves the differential equation y' = f(t,y) at the times
    # specified by the vector tspan and with initial condition y0.
    # - odefun is an anonymous function of the form odefun = lambda t,v: ...
    # - tspan is a 1D array
    # - y0 is a number

    # You fill in below
    dt = tspan[1] - tspan[0]
    y = np.zeros(len(tspan))
    y[0] = y0

    for k in range(len(y) - 1):
        k1 = odefun(tspan[k], y[k])
        k2 = odefun(tspan[k] + dt/2, y[k] + dt/2 * k1)
        y[k + 1] = y[k] + dt * k2

    return tspan, y

dydt = lambda t, y: 0.3*y + t
tans, yans = midpoint(dydt, np.arange(0, 1+0.1, 0.1), 1)
print(yans[-1])

### Solve the same ODE with scipy.integrate.solve_ivp
dydt = lambda t, y: 0.3*y + t
sol = scipy.integrate.solve_ivp(dydt, [0, 1], [1])
yans = sol.y[0, :]
print(yans[-1])

### Question 2 - Solve the ODE using forward Euler and calculate the error

######## System 1 ##########
#dydt = lambda t, y: y**3-y
#exact_sol = 3.451397662017099
#
## Solve with FE
#t0 = time.perf_counter()
#t_sol, y_sol = forward_euler(dydt, tspan, y0)
#time_FE = time.perf_counter() - t0
#error_FE =

## Do the same with BE

## Then do the same with the midpoint method

#t0 = time.perf_counter()
#sol = scipy.integrate.solve_ivp(dydt, np.array([tspan[0], tspan[-1]]), np.array([y0]))
#y_sol = sol.y[0, :]
#time_45 = time.perf_counter() - t0
#error_45 = np.abs(y_sol[-1] - exact_sol)

###########  System 2 ##########
# dydt = lambda t, y: 5e5*(-y + np.sin(t))
# exact_sol = -1e-6
# y0 = 0
# tspan = np.linspace(0, 2*np.pi, 10**2)

## FE

## BE

## Midpoint

## scipy.integrate.solve_ivp
