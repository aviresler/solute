
import numpy as np
import matplotlib.pyplot as plt

delta = 1
x = np.linspace(-5,5,100)
y = np.zeros_like(x)
plt.plot(x, np.log(np.cosh(x)), color='blue', linewidth=2)
ind_below = np.nonzero( np.abs(x) < delta)
ind_above = np.nonzero( np.abs(x) >= delta)
y[ind_below] = 0.5*x[ind_below]*x[ind_below]
y[ind_above] = delta*(np.abs(x[ind_above]) - 0.5*delta)
plt.plot(x, y, color='red', linewidth=2)
plt.show()

#for m in range(number_of_samples):


