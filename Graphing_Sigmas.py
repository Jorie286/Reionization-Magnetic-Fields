import numpy as np
import math
import matplotlib.pyplot as plt
from scipy.linalg import solve_banded

# Calculate the value of sigma.
def get_sigmas(n,c):
    """
    Funtion to find the value of sigma_{l,m} for a certian number of sigmas. For this function, it is assumed that m=1 for all sigmas, only the value of l changes.
    The input for n must be a positive whole number for the function to work correctly, please note that it does not check for good input.
    
    Input arguments (2)
        required    integer values
                        n, the number of sigma_{l,m} parameters we want values for
                        c = (i*D_theta)/(k*v), a constant for which the value can be defined
    Returns
        the values of the first n sigma_{n,1} using a matrix to solve.

    Date of last revision: July 9, 2024
    """
    # Create a zero matrix and fill it with the diagonal part of the tridiagonal matrix
    ab = np.zeros((3,n), dtype = np.complex128)
    for l in range (1, n+1):
        ab[1,l-1] = -l*(l+1)*c # sigma_{l,m} coefficient
        
    for l in range (1, n):
        ab[0,l] = math.sqrt(((l+2)*(l))/(((2*l)+3)*((2*l)+1))) # sigma_{l+1,m} coefficient
        
    for l in range (2, n+1):
        ab[2,l-2] = math.sqrt(((l+1)*(l-1))/(((2*l)-1)*((2*l)+1))) # sigma_{l-1,m} coefficient
    
    # Create a zero matrix for the b vector of ab*x=b and fill it with the coefficients of each Y_l,m from the RHS of our equation.
    b = np.zeros((n,), dtype=np.complex128)
    b[0] = (-2*math.sqrt(math.pi))/math.sqrt(6)
    x = solve_banded((1, 1), ab, b) # Solve for the x vector
    return x

get_sigmas(20, 1j)

# Graph the values of sigma_{1,1} and sigma{2,1} in terms of various possible values for i D_theta / k v.
# We only want to look at these two sigmas because these are the only two that affect G_iso and G_ani.

#Create a list of 20 real numbers with steps of 0.1 for x. We cannot directly plot i D_theta / k v because Matlab will not plot imaginary numbers correctly. 
count = np.arange(0, 10, 0.1).tolist()

#Create a list of i D_theta / k v that are imaginary and have the same constants as count. This is to be used in get_sigmas.
g=[None]*len(count)
for i in range (0, len(count)):
    g[i] = count[i]*1j

#Calculate the values of sigma_{l,m} and separate each into two lists for graphing and analysis.
a = [None]*len(count)
r = [None]*len(count)
im = [None]*len(count)
for i in range (0, len(count)):
    a[i] = get_sigmas(20, g[i])
    r[i] = a[i].real
    im[i] = a[i].imag
    
#We can make an array that contains each r and im list for each value of count.
real = np.empty((len(count), len(a[0])))
imaginary = np.empty((len(count), len(a[0])))
for i in range (0, len(count)):
    real[i, :] = r[i]
    imaginary[i, :] = im[i]

# Graph the imaginary and real parts of the two sigmas sepearatly. 
fig, ax = plt.subplots(figsize=(14, 10))
#For this plot, we are only using the real part of sigma_{l,m}.
ax.plot(count, real[:, 0], label="real sigma_{1,1}")
ax.plot(count, real[:, 1], label="real sigma_{2,1}")
ax.set_xlabel("i D_theta / kv")
ax.set_ylabel("Value of sigma")
ax.legend()

fig, ax = plt.subplots(figsize=(14, 10))
#Here, we are only plotting the imaginary part of sigma_{l,m}.
ax.plot(count, imaginary[:, 0], label="imaginary sigma_{1,1}")
ax.plot(count, imaginary[:, 1], label="imaginary sigma_{2,1}")
ax.set_xlabel("i D_theta / kv")
ax.set_ylabel("Value of sigma")
ax.legend()
