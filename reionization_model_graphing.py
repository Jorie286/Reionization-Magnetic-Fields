# Using the Giso_list generated in the Get_Giso_u.py file, we can graph Giso_u against several variables.
import matplotlib.pyplot as plt
from matplotlib import cm
import matplotlib.colors as colors
from scipy.linalg import solve_banded
import numpy as np
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
# get the sigma_11, sigma_21, and D_theta data
d_theta = np.loadtxt(r'D_theta_vel.txt', dtype=np.complex128)
sigmas_11 = np.loadtxt(r'sigmas_vel_11.txt', dtype=np.complex128)
sigmas_21 = np.loadtxt(r'sigmas_vel_21.txt', dtype=np.complex128)

# get color maps for plotting
magma = cm.get_cmap('magma').resampled(101)
bwr = cm.get_cmap('bwr').resampled(101)
# create a symlog scale to be used when graphing colorbars
plt.rcParams['font.size'] = 25  # Change the matplotlib font size

# plot the neutral fractions of hydrogen and helium over the bins (in meters)
fig, ax = plt.subplots(figsize=(20,10))
ax.plot(data[:,0]*(calc_params.DNHI/calc_params.n_H), data[:,2], linewidth = 4, marker = "<", markersize = 20, markevery = 100, label="yH")
ax.plot(data[:,0]*(calc_params.DNHI/calc_params.n_H), data[:,3], linewidth = 4, marker = "*", markersize = 20, markevery = 100, label="yHe")
ax.grid(True, linestyle="--")
ax.set_ylim(1e-4, 1-1e-4)
ax.set_xlim(1.5e21, max(data[:,0]*(calc_params.DNHI/calc_params.n_H)))
ax.set_xlabel("Distance ($m$)")
ax.set_ylabel("Neutral Fraction")
ax.set_yscale("logit", one_half = "0.5")
ax.set_rasterization_zorder(0)
ax.legend()
fig.savefig("Neutral_frac.pdf")

k_slab_list_1=[0, 10, 20, 40, 90] # list of slabs to plot in Gani plot



# Graph Gani.
fig, ax = plt.subplots(figsize=(20,10))
for k_index in k_slab_list_1:
    ax.plot(data[:,0]*(calc_params.DNHI/calc_params.n_H), Gani[k_index::calc_params.num_k][:calc_params.NSLAB], linewidth = 4, label = "k = %2.1e $m^{-1}$" % calc_params.k[k_index], color=magma(k_index*calc_params.k_step))
ax.hlines(max(Gani), min(data[:,0]*(calc_params.DNHI/calc_params.n_H)), max(data[:,0]*(calc_params.DNHI/calc_params.n_H)), linestyles = "--", color="r", label="Max value of Gani = %3.3f" % max(Gani)) # mark the largest value of Gani
ax.set_ylim(1e-14, max(Gani)+10) # set limits on the plot to remove ambiguous portions
ax.set_yscale("log")
ax.set_xlabel("Distance (m)")
ax.set_ylabel("Gani")
ax.grid(True, linestyle="--") # add grid to make values easier to read
ax.legend()
fig.savefig("Gani_plt_1.pdf")

k_slab_list_2=[0, 10, 20, 40, 60, 80, 90] # list of slabs to plot in Giso plot

# Graph imaginary Giso.
fig, ax = plt.subplots(figsize=(20,10))
for k_index in k_slab_list_2:
    ax.plot(data[:,0]*(calc_params.DNHI*calc_params.n_H), imaginary[k_index::calc_params.num_k][:calc_params.NSLAB], linewidth = 4, label = "k = %2.1e $m^{-1}$" % calc_params.k[k_index], color=magma(k_index*calc_params.k_step))
ax.set_yscale("log")
ax.set_xlabel("Distance ($m$)")
ax.set_ylabel("Giso/u ($s \\ m^{-1}$)")
ax.grid(True, linestyle="--") # add grid to make values easier to read
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
    n_e = ((3*(1+calc_params.z)**3*calc_params.Omega_b*calc_params.H_o**2)/(8*np.pi*calc_params.G))*(calc_params.h*(1-yH)+calc_params.he*(1-yHe))
    return n_e

# compute and plot the imaginary growth rate (Im w) throughout the model
im_w_list = []
x_e_list = []
mu_0 = (4*np.pi)*(10**(-7)) # permiablity of free space (units of newtons ampere**-2)
for m in range(0, calc_params.NSLAB):
    n_e = get_n_e(data[m,2], data[m,3])
    k_sd = np.sqrt((mu_0*n_e*calc_params.q_a**2)/calc_params.m_e)
    for n in range(0, calc_params.num_k):
        im_w_list.append((calc_params.k[n*calc_params.k_step]/imaginary[m*calc_params.num_k+n])*(Gani[m*calc_params.num_k+n]-(calc_params.k[n*calc_params.k_step]/k_sd)**2))
Im_w_arr = np.array(im_w_list).reshape((calc_params.NSLAB,calc_params.num_k)) # reshape the list for easier graphing and analysis
np.savetxt('Im_w_arr.txt', Im_w_arr, delimiter=',') # save the array of Im_w results for future reference

# make a 2D heatmap of Im w
norm = colors.SymLogNorm(linthresh=1e-10, linscale=1.0, vmin=-np.min(Im_w_arr).real, vmax=np.min(Im_w_arr).real)
fig, ax = plt.subplots(1, 2, width_ratios = np.array([3, 1]), figsize=(20,10), sharey=True)
# set the graphing tick labels so that they show the maximum and minimum values of the k and model distance
left=data[0,0]*(calc_params.DNHI/calc_params.n_H)
right=data[-1,0]*(calc_params.DNHI/calc_params.n_H)
bottom=calc_params.k[0]
top=calc_params.k[-1]
extent = [left, right, bottom, top]
im = ax[0].imshow(Im_w_arr.real.T, aspect='auto', cmap="bwr", norm=norm, extent=extent)
ax[0].set_xscale("linear")
ax[0].set_yscale("linear")

# pick out slabs that we want to plot individualy
plot_list=[700, 1100, 1500, 1900] # list of slabs we want to plot in the Im w heatmap subplot
for slab in plot_list:
    ax[0].axvline(x=data[slab,0]*(calc_params.DNHI/calc_params.n_H), linewidth = 4, color = magma(slab/2000), linestyle="--")
cbar = plt.colorbar(im, pad = 0.15, location = "left")
cbar.set_label("Im w ($s^{-1}$)", labelpad=-110, y=-0.1, rotation=0)
# change the xticks to represent distance instead of slab number
ax[0].set_xlabel("Distance ($m$)")
ax[0].set_ylabel("Wavenumber ($m^{-1}$)")

# create a secondary axis that shows the fraction of free electrons in the system as the reionization front moves through
x_e = [] # calculate the fraction of free electrons at each slab
for slab in range(calc_params.NSLAB):
    x_e.append(((1-data[slab, 2])*(1-calc_params.f_He))+((1-data[slab,3])*(calc_params.f_He))) # round all the numbers to 1 decimal
    # x_e = (1-y_Hi)*fH + (1 - y_Hei)fHe ???
#print(x_e)
# add labels for the fraction of ionized electrons
ax[0].text(1.5e21, 0.72e-8, '$\\chi_{e}$=%1.1e' % x_e[700], fontsize=25, color='k', rotation=90)
ax[0].text(2.5e21, 0.75e-8, '$\\chi_{e}$=%1.1e' % x_e[1100], fontsize=25, color='k', rotation=90)
ax[0].text(3.5e21, 0.75e-8, '$\\chi_{e}$=%1.1e' % x_e[1500], fontsize=25, color='k', rotation=90)
ax[0].text(4.5e21, 0.75e-8, '$\\chi_{e}$=%1.1e' % x_e[1900], fontsize=25, color='k', rotation=90)

# plot Im w for the above chosen slabs
for slab in plot_list:
    ax[1].plot(Im_w_arr[slab, :].real, calc_params.k[::calc_params.k_step][:calc_params.num_k], linewidth = 4, color = magma(slab/2000), label="Slab %5.0f" % slab)
ax[1].set_xlabel("Im w ($s^{-1}$)")
ax[1].legend()
#ax[1].set_xlim(np.min(Im_w_arr)-1e-5, np.max(Im_w_arr)+1e-2)
ax[1].set_xscale("log")
fig.subplots_adjust(wspace=0.05)
fig.savefig('Im_w_2D.pdf')



# plot the source term over the slabs indicated in the Im w plot
k_slabs=[1, 50, 100]
fig, ax = plt.subplots(figsize=(20,10))
for slab in plot_list:
    ax.plot(calc_params.velocity, (4*np.pi*calc_params.velocity**3*S20[(calc_params.Nv*calc_params.num_k*(slab-1)):(calc_params.Nv*calc_params.num_k*slab)][calc_params.Nv*(calc_params.num_k-1):calc_params.Nv*calc_params.num_k]), color = magma(slab/2000), linewidth = 4, label = "Slab %5.0f" % slab)
ax.legend()
ax.set_xlabel("Velocity ($m s^{-1}$)")
ax.set_ylabel("Source Term Value ($m^{-3} s^{-1}$)")
fig.savefig('S20_plt.pdf')



# plot the multipole moment over the slabs indicated in the Im w plot
fig, ax = plt.subplots(figsize=(20,10))
for slab in plot_list:
    ax.plot(calc_params.velocity, (4*np.pi*calc_params.velocity**3*a20[(calc_params.Nv*calc_params.num_k*(slab-1)):(calc_params.Nv*calc_params.num_k*slab)][calc_params.Nv*(calc_params.num_k-1):calc_params.Nv*calc_params.num_k]), color = magma(slab/2000), linewidth = 4, label = "Slab %5.0f" % slab)
ax.legend()
ax.set_xlabel("Velocity ($m s^{-1}$)")
ax.set_ylabel("Multipole Moment ($m^{-3}$)")
fig.savefig('a20_plt.pdf')
