# import necessary plotting packages
import matplotlib.pyplot as plt
import numpy as np

# Get the data from Get_Gani to be used for graphing.
from Get_Gani import Gani_data
from Get_Gani import data

# plot the values of Gani and Giso_u against the neutal fractions
# set up a pair of subplots that are ligned up so that we can compare neutral fractions of hydrogen and helium to change in Gani and Giso_u
fig, axs = plt.subplots(2, figsize=(14,10))
# create a list of values that correspond to the bins in the data
j_list = []
for i in range(0,2000):
    j_list.append(i)
# plot Gani and Giso_u over bins (j)
axs[0].plot(j_list, Gani_data, label="Gani")
# plot the hydrogen and helium neutral fraction over the bins (j) in a separate subplot from Gani
axs[1].plot(j_list, data[:,2], label="yH")
axs[1].plot(j_list, data[:,3], label="yHe")
# plot a vertical line that roughly corresponds to the bin number with the maximum value for the source term
axs[0].axvline(x=1070, color = 'r', linestyle = "--", label = "Source Term Max")
axs[1].axvline(x=1070, color = 'r', linestyle = "--", label = "Source Term Max")
# set up plot labels
axs[0].legend()
axs[1].legend()
axs[1].set_xlabel("Bin")
axs[0].set_ylabel("Gani Value")
axs[1].set_ylabel("Neutral Fraction")
axs[0].set_title("Gani Compared to the Neutral Fraction for all Slab Numbers")
fig.show()

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
# set up the plot labels
ax.set_xlabel("Velocity bin")
ax.set_ylabel("Multipole Moment Value")
ax.set_title("Change in Multipole Moment for a Selection of Slab Numbers")
ax.legend()
fig.show()
