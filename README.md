# Reionization-Magnetic-Fields
This repository contains code for solving parts of the magnetic fields at reionization problem.

The Graphing_Sigmas Python code solves for various values of sigma to be used in solving for the anisotropic and isotropic parts of the distribution.
It then graphs the sigmas in terms various potential values of $\frac{i\ D_{\theta}}{k v}$ to see how chaning $\frac{i\ D_{\theta}}{k v}$ affects the value of sigma.

The Reionization-Temp-Modified.c file contains a modified version of the reionization front temperature evolution model by frankelzeng. The output.txt file contains the main output from this modified temperature evolution model. __Note:__ the modified reionization code returns other files which are necessary to run Get_Gani.py but are not included in this repository because they are large and do not display well.

The Get_Giso_u Python code uses the output.txt file to solve for $\frac{G^{iso}}{u}$ after finding the value of $D_{\theta}$ for a specific number of species. This is done by using a Reimann sum over values of velocity in place of using an integral.

The Graphing_Giso_u Python code uses the data from Get_Giso_u to graph the $G^{iso}_u$ data in terms of variables in the output.txt file. These variables include time, H neutral fraction, He neutral fraction, electron temperature, ionized H temperature, and ionized He temperature.

The Get_Gani Python code uses the data from the output.txt file to solve for $G^{ani}$. This uses a very similar method to Get_Giso_u in order to compute the values of $G^{ani}$. Again, a Reimann sum was used to substitute for the integral over v.

The Graphing_Gani code uses the data from Get_Gani to graph the $G^{ani}$ data and the neutral fraction of hydrogen and helium in terms of the slab number ($j$). It also graphs other aspects of the $G^{ani}$ expression including the source term $S_{2,0}$ and the multipole moment $a_{2,0}$ over velocity slabs.

The graph_against_k code graphs the Gani, Giso_u against the wave number (k). This code also calulates the values of the imaginary frequcny (Im w) and plots it against the wave number. Note: this code may contain some errors so it should be double checked at some time.
