import matplotlib.pyplot as plt
import numpy as np

# Get the data from Get_Giso_u to be used for graphing.
from Giso_u.Get_Giso_u import imaginary
# Get the data from Get_Gani to be used for graphing.
from Gani.Get_Gani import Gani_data
from Gani.Get_Gani import data

# get constants for calculating the electron number denstiy
G = const.G # gravitational constant
z = 7 # redshift
Omega_b = 0.046 # Fraction of the universe made of baryonic matter during reionization
H_o = 2.2618e-18 # Hubble constant

# Compute the electron number density during reionization.
def get_n_e(yH, yHe):
    """
    Funtion to find the electron number density (units of electrons m^-3). This function is used as a part of get_Giso_u. The inputs should be postive otherwise
    the ouptut will not make sense, please note that he function does not check for good inputs.
    
    Important note: all physical constants are in units of MKS for easy conversions.
    
    Input arguments (2)
        required    float or integer-like values
                        yH, neutral fraction of hydrogen
                        yHe, neutral fraction of helium
    Returns
        the number density of electrons under the given conditions
        
    Date of last revision: October 28, 2024
    """
    n_e = ((3*(1+z)**3*Omega_b*H_o**2)/(8*math.pi*G))*(4.5767e26*(1-yH)+3.6132e25*(1-yHe))
    return n_e

# calculate the values of the imaginary frequency for all slabs and wavenumbers
im_w_list = []
mu_0 = (4*np.pi)*(10**(-7)) # permiablity of free space (units of newtons ampere**-2)
for m in range(2000):
    n_e = get_n_e(data[m,2], data[m,3])
    k_sd = np.sqrt((mu_0*n_e*q_a**2)/m_b3)
    for n in range(8):
        im_w_list.append((k[n*10]/Giso_im_data[m*n])*(Gani_data[m*n]-(k[n*10]/k_sd)**2))


# make a plot of imaginary w against k
fig, ax = plt.subplots(figsize=(14,10))
ax.plot(k[:8], im_w_list[::2000])
ax.set_title("Imaginary w against k")
ax.set_xlabel("k")
ax.set_ylabel("Imaginary w")
fig.show()


# plot the wavenumbers agains the values that we calculated in Get_Giso and Get_Gani
# make a plot of Giso against k
fig, ax = plt.subplots(figsize=(14,10))
ax.plot(k[::10][:8], Giso_im_data[(8*1069):(8*1070)], label = "k slab 1070")
ax.set_title("Imaginary Giso against k")
ax.set_xlabel("k")
ax.set_ylabel("Giso Imaginary Value")
ax.set_xscale('log')
fig.show()


# make a plot of Gani against k
fig, ax = plt.subplots(figsize=(14,10))
ax.plot(k[::10][:8], Gani_data[(8*1069):(8*1070)], label = "k slab 1070")
ax.set_title("Gani against k")
ax.set_xlabel("k")
ax.set_ylabel("Gani Value")
ax.set_xscale('log')
fig.show()
