import numpy as np
import matplotlib.pyplot as plt
import scipy.integrate
import scipy.optimize

# Problem 1
## Part a
x = lambda t: 11/6 * (np.exp(-t/12) - np.exp(-t))
xprime = lambda t: 11*(-1/12*np.exp(-t/12) + np.exp(-t))/6

x0 = xprime(1.5)
A1 = x0
print("A1", A1)

xprime_roots = scipy.optimize.fsolve(xprime, x0)

t_max = xprime_roots[0]
A2 = t_max
print("A2", A2)

x_max = x(t_max)
A3 = x_max
print("A3", A3)

## Part b
t_max = scipy.optimize.fminbound(lambda t: -x(t), 0, 10)
x_max = x(t_max)
A4 = np.array([t_max, x_max])
print("A4", A4)

# Problem 2
## Part a
f_xy = lambda x, y: (x**2 + y - 11) ** 2 + (x + y**2 - 7) ** 2 # Himmelblau's function
f = lambda p: f_xy(p[0], p[1]) # Adapter function

A5 = f([3, 4])
print("A5", A5)

## Part b
argmin_f = scipy.optimize.fmin(f, np.array([-3, -2]))
A6 = argmin_f
print("A6", A6)

## Part c
gradf_xy = lambda x,y: np.array([4*x**3 - 42*x + 4*x*y + 2*y**2 - 14,
                                 4*y**3 - 26*y + 4*x*y + 2*x**2 - 22])
gradf = lambda p: gradf_xy(p[0], p[1])

A7 = gradf(argmin_f)
print("A7", A7)

norm_2 = np.linalg.norm(A7)
A8 = norm_2
print("A8", A8)

## Part d
p = [-3, -2] # Initial guess defined in part (e)
tol = 10**-7 # Tolerance
iterations = 0

for k in range(2000):
    grad = gradf(p)
    if np.linalg.norm(grad) < tol:
        iterations = k
        break

    phi = lambda t: p - t*grad
    f_phi = lambda t: f(phi(t))
    
    t_min = scipy.optimize.fminbound(f_phi, 0, 1)
    p = phi(t_min)


## Part e
# Done above!
A9 = p
print("A9", A9)

A10 = iterations
print("A10", A10)
