# Using the Giso_list generated in the Get_Giso_u.py file, we can graph Giso_u against several variables.
import matplotlib.pyplot as plt
import numpy as np

# Get the data from Get_Giso_u to be used for graphing.
from Get_Giso_u import real
from Get_Giso_u import imaginary
from Get_Giso_u import data

# Set up a 2x3 grid of subplots so all plots can be viewed together.
fig, axs = plt.subplots(nrows = 2, ncols = 3, figsize=(14,10))

# Graph real and imaginary portions of Giso_list seperatly for each graph.
# Plot Giso_u over time.
axs[0,0].scatter(data[:,0]*12000, real, marker = ".", color = "c", label = "Giso real") # Note: the timestep for data is 12000 (years???).
axs[0,0].scatter(data[:,0]*12000, imaginary, marker = ".", color = "r", label = "Giso imaginary")
axs[0,0].set_xlabel("Time")
axs[0,0].set_ylabel("Giso/u")
axs[0,0].legend()

# Plot Giso_u against yH (neutral H fraction).
axs[0,1].scatter(data[:,2], real, marker = ".", color = "c", label = "Giso real")
axs[0,1].scatter(data[:,2], imaginary, marker = ".", color = "r", label = "Giso imaginary")
axs[0,1].set_xlabel("yH")
axs[0,1].set_ylabel("Giso/u")
axs[0,1].legend()

# Plot Giso_u against yHe (neutral He fraction).
axs[0,2].scatter(data[:,3], real, marker = ".", color = "c", label = "Giso real")
axs[0,2].scatter(data[:,3], imaginary, marker = ".", color = "r", label = "Giso imaginary")
axs[0,2].set_xlabel("yHe")
axs[0,2].set_ylabel("Giso/u")
axs[0,2].legend()

# Plot Giso_u against Te.
axs[1,0].scatter(data[:,5], real, marker = ".", color = "c", label = "Giso real")
axs[1,0].scatter(data[:,5], imaginary, marker = ".", color = "r", label = "Giso imaginary")
axs[1,0].set_xlabel("T e")
axs[1,0].set_ylabel("Giso/u")
axs[1,0].legend()

# Plot Giso_u against THII.
axs[1,1].scatter(data[:,7], real, marker = ".", color = "c", label = "Giso real")
axs[1,1].scatter(data[:,7], imaginary, marker = ".", color = "r", label = "Giso imaginary")
axs[1,1].set_xlabel("THII")
axs[1,1].set_ylabel("Giso/u")
axs[1,1].legend()

# Plot Giso_u against THeII.
axs[1,2].scatter(data[:,13], real, marker = ".", color = "c", label = "Giso real")
axs[1,2].scatter(data[:,13], imaginary, marker = ".", color = "r", label = "Giso imaginary")
axs[1,2].set_xlabel("THeII")
axs[1,2].set_ylabel("Giso/u")
axs[1,2].legend()
