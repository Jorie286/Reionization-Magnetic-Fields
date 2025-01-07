import matplotlib.pyplot as plt
import numpy as np

# Get the data from Get_Giso_u to be used for graphing.
from Giso_u.Get_Giso_u import imaginary
# Get the data from Get_Gani to be used for graphing.
from Gani.Get_Gani import Gani_data
from Gani.Get_Gani import data
import calc_params

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
        
    Date of last revision: January 6, 2025
    """
    n_e = ((3*(1+calc_params.z)**3*calc_params.Omega_b*calc_params.H_o**2)/(8*math.pi*calc_params.G))*(calc_params.h*(1-yH)+calc_params.he*(1-yHe))
    return n_e
    
im_w_list = []
mu_0 = (4*np.pi)*(10**(-7)) # permiablity of free space (units of newtons ampere**-2)
for m in range(0, calc_params.NSLAB):
    n_e = get_n_e(data[m,2], data[m,3])
    k_sd = np.sqrt((mu_0*n_e*calc_params.q_a**2)/calc_params.m_e)
    for n in range(0, calc_params.num_k):
        im_w_list.append((calc_params.k[n*calc_params.k_step]/imaginary[m*n])*(Gani_data[m*n]-(calc_params.k[n*calc_params.k_step]/k_sd)**2))
print(im_w_list)
Im_w_arr = np.array(im_w_list).reshape(calc_params.NSLAB,calc_params.num_k) # reshape the list for easier graphing and analysis

# make a plot of imaginary w against k
fig, ax = plt.subplots(figsize=(14,10))
ax.plot(calc_params.k[::calc_params.k_step], Im_w_arr[0,:]) # plot the first row if values for imaginary omega.
ax.set_title("Imaginary w against k")
ax.set_xlabel("k")
ax.set_ylabel("Imaginary w")
fig.show()

# plot the wavenumbers agains the values that we calculated in Get_Giso and Get_Gani
# make a plot of Giso against k
fig, ax = plt.subplots(figsize=(14,10))
ax.plot(calc_params.k[::calc_params.k_step], imaginary[(calc_params.num_k*1069):(calc_params.num_k*1070)], label = "k slab 1070")
ax.set_title("Imaginary Giso against k")
ax.set_xlabel("k")
ax.set_ylabel("Giso Imaginary Value")
ax.set_xscale('log')
fig.show()


# make a plot of Gani against k
fig, ax = plt.subplots(figsize=(14,10))
ax.plot(calc_params.k[::calc_params.k_step], Gani_data[(calc_params.num_k*1069):(calc_params.num_k*1070)], label = "k slab 1070")
ax.set_title("Gani against k")
ax.set_xlabel("k")
ax.set_ylabel("Gani Value")
ax.set_xscale('log')
fig.show()
