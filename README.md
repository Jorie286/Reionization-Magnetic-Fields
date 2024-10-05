# Reionization-Magnetic-Fields
This repository contains code for solving parts of the magnetic fields at reionization problem.

The Graphing_Sigmas Python code solves for various values of sigma to be used in solving for the anisotropic and isotropic parts of the distribution.
It then graphs the sigmas in terms various potential values of $\frac{i D_theta}{k v}$ to see how chaning $\frac{i D_theta}{k v}$ affects the value of sigma.

The output.txt file contains the output from the reionization code by frankelzeng.

The Get_Giso_u Python code uses the output.txt file to solve for $\frac{Giso}{u}$ after finding the value of D_theta for a specific number of species. This is done by using a Reimann sum over values of velocity in place of using an integral.

The Graphing_Giso_u Python code uses the data from Get_Giso_u to graph the Giso_u data in terms of variables in the output.txt file. These variables include time, H neutral fraction, He neutral fraction, electron temperature, ionized H temperature, and ionized He temperature.

The Get_Gani Python code uses the data from the output.txt file to solve for Gani. This uses a very similar method to Get_Giso_u in order to compute the values of Gani. Again, a Reimann sum was used to substitute for the integral over v. The funtion for get_alm has been added but there are still some bugs in the code that need to be worked out.
