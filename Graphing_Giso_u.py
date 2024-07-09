# Using the Giso_list generated in the Get_Giso_u.py file, we can graph Giso_u against several variables.
import matplotlib.pyplot as plt
import numpy as np

# Get the data from Get_Giso_u to be used for graphing.
from Get_Giso_u import Giso_list
from Get_Giso_u import data

# Set up a 2x3 grid of subplots so all plots can be viewed together.
fig, axs = plt.subplots(nrows = 2, ncols = 3, figsize=(14,10))

# Plot Giso_u over time.
axs[0,0].scatter(data[:,0]*12000, Giso_list, marker = ".") # Note: the timestep for data is 12000 (years???).
axs[0,0].set_xlabel("Time")
axs[0,0].set_ylabel("Giso/u")

# Plot Giso_u against yH (neutral H fraction).
axs[0,1].scatter(data[:,2], Giso_list, marker = ".")
axs[0,1].set_xlabel("yH")
axs[0,1].set_ylabel("Giso/u")

# Plot Giso_u against yHe (neutral He fraction).
axs[0,2].scatter(data[:,3], Giso_list, marker = ".")
axs[0,2].set_xlabel("yHe")
axs[0,2].set_ylabel("Giso/u")

# Plot Giso_u against Te.
axs[1,0].scatter(data[:,5], Giso_list, marker = ".")
axs[1,0].set_xlabel("T e")
axs[1,0].set_ylabel("Giso/u")

# Plot Giso_u against THII.
axs[1,1].scatter(data[:,7], Giso_list, marker = ".")
axs[1,1].set_xlabel("THII")
axs[1,1].set_ylabel("Giso/u")

#Plot Giso_u against THeII.
axs[1,2].scatter(data[:,13], Giso_list, marker = ".")
axs[1,2].set_xlabel("THeII")
axs[1,2].set_ylabel("Giso/u")
