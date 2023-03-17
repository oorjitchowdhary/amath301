import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize
import os

M = np.genfromtxt(os.path.join(os.path.dirname(__file__), 'fitdata.csv'), delimiter=',') # Read in data from csv file
X = M[:,0] # get x values from data
Y = M[:,1] # get y values from data

fig, ax = plt.subplots() # setup plots
ax.plot(X, Y, 'ko') # Plot the data
ax.set_ylim([-0.5, 1]) # Change the ylimit to match the data well

# Gaussian function
mu = 0
sigma2 = 1

y = lambda x: 1/(np.sqrt(2*np.pi*sigma2)) * np.exp(-(x-mu)**2/(2*sigma2))

# Error function
def sum_of_squares(mu, sigma2):
    y = lambda x: 1/(np.sqrt(2*np.pi*sigma2)) * np.exp(-(x-mu)**2/(2*sigma2))
    return sum((y(X) - Y)**2)

# Adapter function for error
sumSquaredError = lambda p: sum_of_squares(p[0], p[1])
print(sumSquaredError(np.array([0, 1])))

# Find the minimum
p = scipy.optimize.fmin(sumSquaredError, np.array([0, 1]))
print(p)

xs = np.linspace(-6,8) # The x values for plotting
ax.plot(xs, y(xs)) # Plot the model 'y'

for index, xdata in enumerate(X): # `xdata' will be the value in the `X' array. `index' will be the index of that x value.

    ydata = Y[index] # Get the y value for the matching x value from the data

    ax.plot( xdata, y(xdata), 'ko') # This plots the model's prediction of the y value at that x value.
    ax.plot( np.array([xdata, xdata]), np.array([ydata, y(xdata)]), 'r') # creates a red line between the true
                                                            # value and the model's prediction. This is the error.
plt.show()