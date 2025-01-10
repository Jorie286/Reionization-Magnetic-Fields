# Using the Giso_list generated in the Get_Giso_u.py file, we can graph Giso_u against several variables.
import matplotlib.pyplot as plt
import numpy as np

# Get the data from Get_Giso_u to be used for graphing.
imaginary = np.loadtext(r'Giso_u.txt')

import calc_params

# Set up a 2x3 grid of subplots so all plots can be viewed together.
fig, axs = plt.subplots(nrows = 2, ncols = 3, figsize=(14,10))

# Create lists to iterate through while creating the plots. This eliminates many lines of uncessary code.
positions = [(0,0), (0,1), (0,2), (1,0), (1,1), (1,2)]
# in this plot, we will use the slab numbers to calcuate the amount of time that has passed (timestep comes from output.txt)
data_list = [data[:,0]*calc_params.Timestep, data[:,2], data[:,3], data[:,5], data[:,7], data[:,13]] 
labels = ["Time", "yH", "yHe", "T e", "THII", "THeII"] 
# Graph real and imaginary portions of Giso_list seperatly for each graph.
# Plot Giso_u over time.
for g in range(0,6):
    axs[positions[0+g]].plot(data_list[0+g], imaginary[::calc_params.k_step], color = "r", label = "Giso imaginary") # Note: the timestep for data is 12000 (years???).
    axs[positions[0+g]].set_xlabel(labels[0+g])
    axs[positions[0+g]].set_ylabel("Giso/u")
    axs[positions[0+g]].legend()
