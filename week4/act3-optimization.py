import numpy as np

# Define function of two variables
f = lambda x, y: (x-2)**2 + (y+1)**2 + 5*np.sin(x)*np.sin(y) + 100

print(f(0, 0)) # Should be 105

# We want to use an "adapter function" to make this function have just one
# input. This is because that is the required format for python built-in
# functions.
fp = lambda p: f(p[0], p[1]) #p[0] represents x, p[1] represents y


# gradf function
gradf = lambda x, y: np.array([2*(x-2) + 5*np.cos(x)*np.sin(y), 2*(y+1) + 5*np.sin(x)*np.cos(y)])
print(gradf(1, 1))
print(gradf(6, 4))

phi = lambda t: np.array([1, 1]) - t*gradf(1, 1)
print("Phi:", phi(1))