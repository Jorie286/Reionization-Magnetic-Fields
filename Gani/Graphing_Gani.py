# import necessary plotting packages
import matplotlib.pyplot as plt
import numpy as np

# Get the data from Get_Gani to be used for graphing.
from Get_Gani import Gani_data
from Get_Gani import data
# get imaginary data from Get_Giso_u to be used for graphing
from Giso_u.Get_Giso_u import imaginary

import calc_params

# iterate through all the values of a20 and find the maximum absolute value of each slab
n = 0
v = 1
max_a = 0
max_slab = 0
max_val = 0
a_list = []
a_max = []
for j in a20:
    if n==calc_params.Nv*8*v: # check to make sure we are not at the end of a slab
        max_a = np.max(np.abs(a_list))
        # get the slab and value of a_20 at its maximum
        if max_a>max_val:
            max_slab = v
            max_val = max_a
        v+=1 # increase v so the loop checks for the end of the next slab
        a_max.append(max_a) # append the maximum value of the previous slab to the max list
        a_list = [] # clear the list for the next iteration
        
    # add the value of a20 for the velocity bin we are on
    a_list.append(j)
    n+=1
    if n==calc_params.NSLAB*8*calc_params.Nv:
        a_max.append(np.max(np.abs(a_list)))
print("The maximum absolute value of the multipole moment is", max_val)
print("The slab number (j) where the absolute value of the multipole moment is at a maximum is", max_slab)

# iterate through all the values of S20 and find the maximum absolute value of each slab
n = 0
v = 1
max_S = 0
max_slab_S = 0
max_val_S = 0
S_list = []
S_max = []
for j in S20:
    if n==calc_params.Nv*8*v: # check to make sure we are not at the end of a slab
        max_S = np.max(np.abs(S_list))
        # get the slab and value of S_20 at its maximum
        if max_S>max_val_S:
            max_slab_S = v
            max_val_S = max_S
        v+=1 # increase v so the loop checks for the end of the next slab
        S_max.append(max_S) # append the maximum value of the previous slab to the max list
        S_list = [] # clear the list for the next iteration
        
    # add the value of a20 for the velocity bin we are on
    S_list.append(j)
    n+=1
    if n==calc_params.NSLAB*8*calc_params.Nv:
        S_max.append(np.max(np.abs(S_list)))
print("The maximum absolute value of the source term is", max_val_S)
print("The slab number (j) where the absolute value of the source term is at a maximum is", max_slab_S)


# plot the source term values at various bin numbers, we want to find where it reaches its maximum
# get the S_{2,0} data from the file created by Get_Gani
S20 = np.loadtxt(r'S20.txt')
# create a plot and indexing list for convenient graphing
fig, ax = plt.subplots(figsize=(14,10))
# plot source term over various slab numbers
# Note: we are only plotting the last k slab
ax.plot(calc_params.velocity, (4*np.pi*np.array(calc_params.velocity)**3*S20[(calc_params.Nv*calc_params.num_k*999):(calc_params.Nv*calc_params.num_k*1000)][calc_params.Nv*(calc_params.num_k-1):calc_params.Nv*calc_params.num_k]), label = "Slab 1000")
ax.plot(calc_params.velocity, (4*np.pi*np.array(calc_params.velocity)**3*S20[(calc_params.Nv*calc_params.num_k*1029):(calc_params.Nv*calc_params.num_k*1030)][calc_params.Nv*(calc_params.num_k-1):calc_params.Nv*calc_params.num_k]), label = "Slab 1030")
ax.plot(calc_params.velocity, (4*np.pi*np.array(calc_params.velocity)**3*S20[(calc_params.Nv*calc_params.num_k*1049):(calc_params.Nv*calc_params.num_k*1050)][calc_params.Nv*(calc_params.num_k-1):calc_params.Nv*calc_params.num_k]), label = "Slab 1050")
ax.plot(calc_params.velocity, (4*np.pi*np.array(calc_params.velocity)**3*S20[(calc_params.Nv*calc_params.num_k*1069):(calc_params.Nv*calc_params.num_k*1070)][calc_params.Nv*(calc_params.num_k-1):calc_params.Nv*calc_params.num_k]), label = "Slab 1070")
ax.plot(calc_params.velocity, (4*np.pi*np.array(calc_params.velocity)**3*S20[(calc_params.Nv*calc_params.num_k*1089):(calc_params.Nv*calc_params.num_k*1090)][calc_params.Nv*(calc_params.num_k-1):calc_params.Nv*calc_params.num_k]), label = "Slab 1090")
ax.plot(calc_params.velocity, (4*np.pi*np.array(calc_params.velocity)**3*S20[(calc_params.Nv*calc_params.num_k*1099):(calc_params.Nv*calc_params.num_k*1100)][calc_params.Nv*(calc_params.num_k-1):calc_params.Nv*calc_params.num_k]), label = "Slab 1100")
# set up plot labels
ax.set_xlabel("Velocity [m/s]")
ax.set_ylabel("Source Term Value")
ax.set_title("Change in Source Term for a Selection of Slab Numbers")
ax.legend()
fig.show()


# plot the multipole moment values of a_{2,0} over varous slab numbers like we plotted S_{2,0} above
# get the a_{2,0} data file from the output of Get_Gani
a20 = np.loadtxt(r'a20.txt')
# create a plot and indexing list for convenient graphing
fig, ax = plt.subplots(figsize=(14,10))
# plot multipole moment over various slab numbers
# Note: we are only plotting the last k slab
ax.plot(calc_params.velocity, (4*np.pi*calc_params.velocity**3*a20[(calc_params.Nv*calc_params.num_k*999):(calc_params.Nv*calc_params.num_k*1000)][calc_params.Nv*(calc_params.num_k-1):calc_params.Nv*calc_params.num_k]), label = "Slab 1000")
ax.plot(calc_params.velocity, (4*np.pi*calc_params.velocity**3*a20[(calc_params.Nv*calc_params.num_k*1029):(calc_params.Nv*calc_params.num_k*1030)][calc_params.Nv*(calc_params.num_k-1):calc_params.Nv*calc_params.num_k]), label = "Slab 1030")
ax.plot(calc_params.velocity, (4*np.pi*calc_params.velocity**3*a20[(calc_params.Nv*calc_params.num_k*1049):(calc_params.Nv*calc_params.num_k*1050)][calc_params.Nv*(calc_params.num_k-1):calc_params.Nv*calc_params.num_k]), label = "Slab 1050")
ax.plot(calc_params.velocity, (4*np.pi*calc_params.velocity**3*a20[(calc_params.Nv*calc_params.num_k*1069):(calc_params.Nv*calc_params.num_k*1070)][calc_params.Nv*(calc_params.num_k-1):calc_params.Nv*calc_params.num_k]), label = "Slab 1070")
ax.plot(calc_params.velocity, (4*np.pi*calc_params.velocity**3*a20[(calc_params.Nv*calc_params.num_k*1089):(calc_params.Nv*calc_params.num_k*1090)][calc_params.Nv*(calc_params.num_k-1):calc_params.Nv*calc_params.num_k]), label = "Slab 1090")
ax.plot(calc_params.velocity, (4*np.pi*calc_params.velocity**3*a20[(calc_params.Nv*calc_params.num_k*1099):(calc_params.Nv*calc_params.num_k*1100)][calc_params.Nv*(calc_params.num_k-1):calc_params.Nv*calc_params.num_k]), label = "Slab 1100")
ax.plot(calc_params.velocity, (4*np.pi*calc_params.velocity**3*a20[(calc_params.Nv*calc_params.num_k*1199):(calc_params.Nv*calc_params.num_k*1200)][calc_params.Nv*(calc_params.num_k-1):calc_params.Nv*calc_params.num_k]), label = "Slab 1200")
# set up the plot labels
ax.set_xlabel("Velocity [m/s]")
ax.set_ylabel("Multipole Moment Value")
ax.set_title("Change in Multipole Moment for a Selection of Slab Numbers")
ax.legend()
fig.show()


# plot the values of Gani against the neutal fractions
# set up a pair of subplots that are ligned up so that we can compare neutral fractions of hydrogen and helium to change in Gani
fig, axs = plt.subplots(2, figsize=(14,10))
# create a list of values that correspond to the bins in the data
j_list = []
for i in range(0,calc_params.NSLAB):
    j_list.append(i)
# plot Gani over bins (j)
axs[0].plot(j_list, Gani_data[::calc_params.k_step], label="Gani")
# plot the hydrogen and helium neutral fraction over the bins (j) in a separate subplot from Gani
axs[1].plot(j_list, data[:,2], label="yH")
axs[1].plot(j_list, data[:,3], label="yHe")
# plot a vertical line that roughly corresponds to the bin number with the maximum value for the source term
axs[0].axvline(x=max_slab_S, color = 'r', linestyle = "--", label = "Source Term Max")
axs[1].axvline(x=max_slab_S, color = 'r', linestyle = "--", label = "Source Term Max")
axs[0].axvline(x=max_slab, color = 'b', linestyle = "--", label = "a_20 Max")
axs[1].axvline(x=max_slab, color = 'b', linestyle = "--", label = "a_20 Max")
# set up plot labels
axs[0].legend()
axs[1].legend()
axs[1].set_xlabel("Bin")
axs[0].set_ylabel("Gani Value")
axs[1].set_ylabel("Neutral Fraction")
axs[0].set_title("Gani Compared to the Neutral Fraction for all Slab Numbers")
fig.show()


# make a plot of a_{2,0} at its largest (absolute) value
# find the velocity bin number of the following value
v_n = ((3*calc_params.k_B*data[1077,5])/calc_params.m_e)**0.5
for v in range(len(calc_params.velocity)):
    if v_n <= calc_params.velocity[v]:
        v_bin = v
        break
        
fig, ax = plt.subplots(figsize=(14,10))
# plot multipole moment at its maximum
# Note: we are only plotting the last k bin
ax.plot(calc_params.velocity, (4*np.pi*calc_params.velocity**3*a20[(calc_params.Nv*calc_params.num_k*1076):(calc_params.Nv*calc_params.num_k*1077)][calc_params.Nv*(calc_params.num_k-1):calc_params.Nv*calc_params.num_k]), label = "Slab 1077")
ax.axvline(x=calc_params.velocity[v_bin], color = 'r', linestyle = "--")
# add the equation for this and arrow for this value on the plot
equation = r'$\sqrt{\frac{3 k_B T_e}{m_e}}$'
ax.text(calc_params.velocity[v_bin+1], -0.005, equation, fontsize=20)
# set up the plot labels
ax.set_xlabel("Velocity [m/s]")
ax.set_ylabel("Multipole Moment Value")
ax.set_title("Change in Multipole Moment at Maximum")
ax.legend()
fig.show()


# make a plot of S_{2,0} at its largest (absolute) value
fig, ax = plt.subplots(figsize=(14,10))
# plot multipole moment at its maximum
ax.plot(calc_params.velocity, (4*np.pi*np.array(calc_params.velocity)**3*S20[(calc_params.Nv*calc_params.num_k*1068):(calc_params.Nv*calc_params.num_k*1069)][calc_params.Nv*(calc_params.num_k-1):calc_params.Nv*calc_params.num_k]), label = "Slab 1069")
# set up the plot labels
ax.set_xlabel("Velocity [m/s]")
ax.set_ylabel("Source Term Value")
ax.set_title("Change in the Source Term at Maximum")
ax.legend()
fig.show()

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
