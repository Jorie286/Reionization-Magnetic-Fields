import matplotlib.pyplot as plt
import numpy as np
import math

# Get the data from Get_Giso_u to be used for graphing.data = np.loadtxt(r'output.txt')
Gani_data = np.loadtxt(r'Gani.txt', dtype=np.complex128)
# get imaginary data from Get_Giso_u to be used for graphing
imaginary = np.loadtxt(r'Giso_u.txt')
data = np.loadtxt(r'output.txt')

import calc_params

# Compute the electron number density during reionization.
def get_n_e(yH, yHe):
    """
    Funtion to find the electron number density (units of electrons m^-3). This function is used as a part of get_Giso_u. The inputs should be postive otherwise
    the ouptut will not make sense, please note that he function does not check for good inputs.

    Important note: all physical constants are in units ov MKS for easy conversions.

    Input arguments (2)
        required    float or integer-like values
                        yH, neutral fraction of hydrogen
                        yHe, neutral fraction of helium
    Returns
        the number density of electrons under the given conditions

    Date of last revision: December 28, 2025
    """
    n_e = ((3*(1+calc_params.z)**3*calc_params.Omega_b*calc_params.H_o**2)/(8*math.pi*calc_params.G))*(calc_params.h*(1-yH)+calc_params.he*(1-yHe))
    return n_e
    
im_w_list = []
mu_0 = (4*np.pi)*(10**(-7)) # permiablity of free space (units of newtons ampere**-2)
for m in range(0, calc_params.NSLAB):
    n_e = get_n_e(data[m,2], data[m,3])
    k_sd = np.sqrt((mu_0*n_e*calc_params.q_a**2)/calc_params.m_e) # skin depth wavenumber
    for n in range(0, calc_params.num_k):
        im_w_list.append((calc_params.k[n*calc_params.k_step]/imaginary[m*calc_params.num_k+n])*(Gani_data[m*calc_params.num_k+n]-(calc_params.k[n*calc_params.k_step]/k_sd)**2))
print(im_w_list, len(im_w_list))
Im_w_arr = np.array(im_w_list).reshape(calc_params.NSLAB,calc_params.num_k) # reshape the list for easier graphing and analysis
f_w = open("Im_w.txt", "a")
for a in im_w_list: # write each value in Gani_data to a new line in the text file
    f_w.write(str(a))
    f_w.write("\n")
f_w.close()

# make a plot of imaginary w against k
fig, ax = plt.subplots(figsize=(14,10))
ax.plot(calc_params.k[::calc_params.k_step], Im_w_arr[0,:]) # plot the first row if values for imaginary omega.
ax.set_title("Imaginary w against k")
ax.set_xlabel("k")
ax.set_ylabel("Imaginary w")
fig.show()
fig.savefig("Im_w.pdf")

# plot the wavenumbers agains the values that we calculated in Get_Giso and Get_Gani
# make a plot of Giso against k
fig, ax = plt.subplots(figsize=(14,10))
for ind in range(0, calc_params.num_k):
    ax.plot(calc_params.k[ind::calc_params.k_step][:calc_params.num_k], Giso_im_arr[1069], label = "k slab 1069 (Source max) (k 2.1f%)" % ind)
ax.set_title("Imaginary Giso against k")
ax.set_xlabel("k")
ax.set_ylabel("Giso Imaginary Value")
ax.set_xscale('log')
ax.legend()
fig.show()
fig.savefig("k_Giso.pdf")

# make a plot of Gani against k
fig, ax = plt.subplots(figsize=(14,10))
ax.plot(calc_params.k[::calc_params.k_step][:calc_params.num_k], Gani_data[(calc_params.num_k*1068):(calc_params.num_k*1069)], label = "k slab 1069 (Source max)")
ax.plot(calc_params.k[::calc_params.k_step][:calc_params.num_k], Gani_data[(calc_params.num_k*1076):(calc_params.num_k*1077)], label = "k slab 1077 (Multipole max)")
ax.set_title("Gani against k")
ax.set_xlabel("k")
ax.set_ylabel("Gani Value")
ax.set_xscale('log')
ax.legend()
fig.show()
fig.savefig("k_Gani.pdf")
