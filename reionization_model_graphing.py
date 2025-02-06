# Using the Giso_list generated in the Get_Giso_u.py file, we can graph Giso_u against several variables.
import matplotlib.pyplot as plt
from matplotlib import cm
import matplotlib.colors as colors
import numpy as np
import math
import calc_params

# Get the data from the reionization front model
data = np.loadtxt(r'output.txt')
# get gani and imaginary giso_u data
Gani = np.loadtxt(r'Gani.txt', dtype=np.complex128)
imaginary = np.loadtxt(r'Giso_u.txt')
# get the S_{2,0} data from the file created by Get_Gani
S20 = np.loadtxt(r'S20.txt')
# get the a_{2,0} data file from the output of Get_Gani
a20 = np.loadtxt(r'a20.txt')

# get color maps for plotting
magma = cm.get_cmap('magma').resampled(101)
bwr = cm.get_cmap('bwr').resampled(101)
# create a symlog scale to be used when graphing colorbars
norm = colors.SymLogNorm(linthresh=1e-10, linscale=1.0, vmin=-1e-7, vmax=1e-7)
plt.rcParams['font.size'] = 25  # Change the matplotlib font size

# plot the neutral fractions of hydrogen and helium over the bins (in meters)
fig, ax = plt.subplots(figsize=(20,10))
ax.plot(data[:,0]*(calc_params.DNHI/calc_params.n_H), data[:,2], linewidth = 4, marker = "<", markersize = 20, markevery = 100, label="yH")
ax.plot(data[:,0]*(calc_params.DNHI/calc_params.n_H), data[:,3], linewidth = 4, marker = "*", markersize = 20, markevery = 100, label="yHe")
ax.set_xlabel("Distance (m)")
ax.set_ylabel("Neutral Fraction")
ax.set_yscale("logit", one_half = "0.5")
ax.set_rasterization_zorder(0)
ax.legend()
fig.savefig("Neutral_frac.pdf")

k_slab_list_1=[0, 10, 20, 40, 90] # list of slabs to plot in Gani plot



# Graph Gani.
fig, ax = plt.subplots(figsize=(20,10))
for k_index in k_slab_list_1:
    ax.plot(data[:,0]*(calc_params.DNHI/calc_params.n_H), Gani[k_index::calc_params.num_k][:calc_params.NSLAB], linewidth = 4, label = "k slab %2.1f" % k_index, color=magma(k_index*calc_params.k_step))
ax.set_xlabel("Distance (m)")
ax.set_ylabel("Giso/u")
ax.set_rasterization_zorder(0)
ax.legend()
fig.savefig("Gani_plt_1.pdf")

k_slab_list_2=[0, 10, 20, 40, 60, 80, 90] # list of slabs to plot in Giso plot

# Graph imaginary Giso.
fig, ax = plt.subplots(figsize=(20,10))
for k_index in k_slab_list_2:
    ax.plot(data[:,0]*(calc_params.DNHI*calc_params.n_H), imaginary[k_index::calc_params.num_k][:calc_params.NSLAB], linewidth = 4, label = "k slab %2.1f" % k_index, color=magma(k_index*calc_params.k_step))
ax.set_xlabel("Distance (m)")
ax.set_ylabel("Giso/u")
ax.set_rasterization_zorder(0)
ax.legend()
fig.savefig("Giso_plt_1.pdf")



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

# compute and plot the imaginary growth rate (Im w) throughout the model
im_w_list = []
mu_0 = (4*np.pi)*(10**(-7)) # permiablity of free space (units of newtons ampere**-2)
for m in range(0, calc_params.NSLAB):
    n_e = get_n_e(data[m,2], data[m,3])
    k_sd = np.sqrt((mu_0*n_e*calc_params.q_a**2)/calc_params.m_e)
    for n in range(0, calc_params.num_k):
        im_w_list.append((calc_params.k[n*calc_params.k_step]/imaginary[m*calc_params.num_k+n])*(Gani[m*calc_params.num_k+n]-(calc_params.k[n*calc_params.k_step]/k_sd)**2))
Im_w_arr = np.array(im_w_list).reshape((calc_params.NSLAB,calc_params.num_k)) # reshape the list for easier graphing and analysis

# make a 2D heatmap of Im w
fig, ax = plt.subplots(1, 2, width_ratios = np.array([3, 1]), figsize=(20,10), sharey=True)
# set the graphing tick labels so that they show the maximum and minimum values of the k and model distance
left=data[0,0]*(calc_params.DNHI/calc_params.n_H)
right=data[-1,0]*(calc_params.DNHI/calc_params.n_H)
bottom=calc_params.k[0]
top=calc_params.k[-1]
extent = [left, right, bottom, top]
im = ax[0].imshow(Im_w_arr.real.T, aspect='auto', cmap="bwr", norm=norm, extent=extent)

# pick out slabs that we want to plot individualy
plot_list=[500, 1080, 1800] # list of slabs we want to plot in the Im w heatmap subplot
for slab in plot_list:
    ax[0].axvline(x=data[slab,0]*(calc_params.DNHI/calc_params.n_H), linewidth = 4, color = magma(slab/2000), linestyle="--")
cbar = plt.colorbar(im, pad = 0.13, location = "left")
cbar.set_label("Im w", labelpad=-110, y=-0.1, rotation=0)
# change the xticks to represent distance instead of slab number
ax[0].set_xlabel("Distance (m)")
ax[0].set_ylabel("Wavenumber")

# plot Im w for the above chosen slabs
for slab in plot_list:
    ax[1].plot(Im_w_arr[slab, :].real, calc_params.k[::calc_params.k_step][:calc_params.num_k], linewidth = 4, color = magma(slab/2000))
ax[1].set_xlabel("Im w")
fig.subplots_adjust(wspace=0.05)
fig.savefig('Im_w_2D.pdf')



# plot the source term over the slabs indicated in the Im w plot
fig, ax = plt.subplots(figsize=(20,10))
for slab in plot_list:
    ax.plot(calc_params.velocity, (4*np.pi*calc_params.velocity**3*S20[(calc_params.Nv*calc_params.num_k*(slab-1)):(calc_params.Nv*calc_params.num_k*slab)][calc_params.Nv*(calc_params.num_k-1):calc_params.Nv*calc_params.num_k]), color = magma(slab/2000), linewidth = 4, label = "Slab %5.0f" % slab)
ax.set_xlabel("Velocity")
ax.set_ylabel("Source Term Value")
fig.savefig('S20_plt.pdf')



# plot the multipole moment over the slabs indicated in the Im w plot
fig, ax = plt.subplots(figsize=(20,10))
for slab in plot_list:
    ax.plot(calc_params.velocity, (4*np.pi*calc_params.velocity**3*a20[(calc_params.Nv*calc_params.num_k*(slab-1)):(calc_params.Nv*calc_params.num_k*slab)][calc_params.Nv*(calc_params.num_k-1):calc_params.Nv*calc_params.num_k]), color = magma(slab/2000), linewidth = 4, label = "Slab %5.0f" % slab)
ax.set_xlabel("Velocity")
ax.set_ylabel("Multipole Moment")
fig.savefig('a20_plt.pdf')
