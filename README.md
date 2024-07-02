# Reionization-Magnetic-Fields
This repository contains code for solving parts of the magnetic fields at reionization problem.

The Graphing_Sigmas Python code solves for various values of sigma to be used in solving for the anisotropic and isotropic parts of the distribution.
It then graphs the sigmas in terms various potential values of i D_theta / k v to see how chaning i D_theta / k v affects the value of sigma.

The output.txt file contains the output from the reionization code by frankelzeng.

The Get_Giso_u Python code uses the output.txt file to solve for Giso/u after finding the value of D_theta for a specific number of species. This is done by using a Reimann sum over values of velocity in place of using an integral.
