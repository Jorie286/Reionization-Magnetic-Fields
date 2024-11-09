# import necessary plotting packages
import matplotlib.pyplot as plt
import numpy as np

# Get the data from Get_Gani to be used for graphing.
from Get_Gani import Gani_data
from Get_Gani import data

# plot the source term values at various bin numbers, we want to find where it reaches its maximum
# get the S_{2,0} data from the file created by Get_Gani
S20 = np.loadtxt(r'S20test.txt')
# create a plot and indexing list for convenient graphing
fig, ax = plt.subplots(figsize=(14,10))
v_list = []
for i in range(0,71):
    v_list.append(i)
# plot source term over various slab numbers (the numbers plotted below correspond to where S_{2,0} reaches a max)
ax.plot(v_list, S20[(71*999):(71*1000)], label = "Slab 1000")
ax.plot(v_list, S20[(71*1029):(71*1030)], label = "Slab 1030")
ax.plot(v_list, S20[(71*1049):(71*1050)], label = "Slab 1050")
ax.plot(v_list, S20[(71*1069):(71*1070)], label = "Slab 1070")
ax.plot(v_list, S20[(71*1089):(71*1090)], label = "Slab 1090")
ax.plot(v_list, S20[(71*1099):(71*1100)], label = "Slab 1100")
# set up plot labels
ax.set_xlabel("Velocity bin")
ax.set_ylabel("Source Term Value")
ax.set_title("Change in Source Term for a Selection of Slab Numbers")
ax.legend()
fig.show()

# plot the multipole moment values of a_{2,0} over varous slab numbers like we plotted S_{2,0} above
# get the a_{2,0} data file from the output of Get_Gani
a20 = np.loadtxt(r'a20test.txt')
# create a plot and indexing list for convenient graphing
fig, ax = plt.subplots(figsize=(14,10))
v_list = []
for i in range(0,71):
    v_list.append(i)
# plot multipole moment over various slab numbers
ax.plot(v_list, a20[(71*999):(71*1000)], label = "Slab 1000")
ax.plot(v_list, a20[(71*1029):(71*1030)], label = "Slab 1030")
ax.plot(v_list, a20[(71*1049):(71*1050)], label = "Slab 1050")
ax.plot(v_list, a20[(71*1069):(71*1070)], label = "Slab 1070")
ax.plot(v_list, a20[(71*1089):(71*1090)], label = "Slab 1090")
ax.plot(v_list, a20[(71*1099):(71*1100)], label = "Slab 1100")
ax.plot(v_list, a20[(71*1199):(71*1200)], label = "Slab 1200")
# set up the plot labels
ax.set_xlabel("Velocity bin")
ax.set_ylabel("Multipole Moment Value")
ax.set_title("Change in Multipole Moment for a Selection of Slab Numbers")
ax.legend()
fig.show()

# iterate through all the values of a20 and find the maximum absolute value of each slab
n = 0
v = 1
max_a = 0
max_slab = 0
max_val = 0
a_list = []
a_max = []
for j in a20:
    if n==71*v: # check to make sure we are not at the end of a slab
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
    if n==2000*71:
        a_max.append(np.max(np.abs(a_list)))
print("The maximum absolute value of the multipole moment is", max_val)
print("The slab number where the absolute value of the multipole moment is at a maximum is", max_slab)

# iterate through all the values of S20 and find the maximum absolute value of each slab
n = 0
v = 1
max_S = 0
max_slab_S = 0
max_val_S = 0
S_list = []
S_max = []
for j in S20:
    if n==71*v: # check to make sure we are not at the end of a slab
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
    if n==2000*71:
        S_max.append(np.max(np.abs(S_list)))
print("The maximum absolute value of the source term is", max_val_S)
print("The slab number where the absolute value of the source term is at a maximum is", max_slab_S)
        
# plot the values of Gani against the neutal fractions
# set up a pair of subplots that are ligned up so that we can compare neutral fractions of hydrogen and helium to change in Gani
fig, axs = plt.subplots(2, figsize=(14,10))
# create a list of values that correspond to the bins in the data
j_list = []
for i in range(0,2000):
    j_list.append(i)
# plot Gani over bins (j)
axs[0].plot(j_list, Gani_data, label="Gani")
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
v_n = ((3*k_B*data[1078,5])/m_e)**0.5
for v in range(len(velocity)):
    if v_n <= velocity[v]:
        v_bin = v
        break
        
fig, ax = plt.subplots(figsize=(14,10))
v_list = []
for i in range(0,71):
    v_list.append(i)
# plot multipole moment at its maximum
ax.plot(v_list, a20[(71*1077):(71*1078)], label = "Slab 1078")
ax.axvline(x=v_bin, color = 'r', linestyle = "--")
# add the equation for this and arrow for this value on the plot
equation = r'$\sqrt{\frac{3 k_B T_e}{m_e}}$'
ax.text(v_bin+1, -0.5e-18, equation, fontsize = 20)
# set up the plot labels
ax.set_xlabel("Velocity bin")
ax.set_ylabel("Multipole Moment Value")
ax.set_title("Change in Multipole Moment at Maximum")
ax.legend()
fig.show()

# make a plot of S_{2,0} at its largest (absolute) value
fig, ax = plt.subplots(figsize=(14,10))
v_list = []
for i in range(0,71):
    v_list.append(i)
# plot source term at its maximum
ax.plot(v_list, S20[(71*1068):(71*1069)], label = "Slab 1069")
# set up the plot labels
ax.set_xlabel("Velocity bin")
ax.set_ylabel("Source Term Value")
ax.set_title("Change in the Source Term at Maximum")
ax.legend()
fig.show()
