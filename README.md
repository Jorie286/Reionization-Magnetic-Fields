# Reionization-Magnetic-Fields
This repository contains code for solving parts of the magnetic fields at reionization problem. The following is a list the main files in this repository as well as their purpose.

The Reionization-Temp-Modified.c file contains a modified version of the reionization front temperature evolution model by frankelzeng [see repository for more informaiton.](https://github.com/frankelzeng/reionization.git) The output.txt file contains the main output from this modified temperature evolution model. __Note:__ the modified reionization code returns other files which are necessary to run Get_Gani.py but are not included in this repository because they are large.

The calc_params.py script contains all the necessary variables to run the computational and graphing code for both $G^{ani}$ and $G^{iso}_u$. To adjust the parameters that you want to use when calculating $G^{iso}_u$ or $G^{ani}$ the variables in calc_params.py can be changed and these changes should then be reflected throughout the computational and graphing code.

The get_sigmas.py script calculates the values of $\frac{D_{theta}}{kv}$ and $\sigma_{l,m}$ for all wavenumber slabs and slabs of the reionization front model. These are stored in a text file for use in the reionization_model_graphing.py file.

The reionization_model_graphing.py file uses the outputs from Get_Giso_u.py, Get_Gani.py, get_sigmas.py, and Reionization-Temp-Modified.c to create plots illustrating relationships between various parameters. These include imaginary growth rate against wavenumber, value of $G^{iso}_u$ and $G^{ani}$ against the reionization front model slabs, and the value of the source term and multipole moment against the reionization front model slabs.

The Graphing_Sigmas.py python code solves for various values of sigma to be used in solving for the anisotropic and isotropic parts of the distribution.
It then graphs the sigmas in terms various potential values of $\frac{i\ D_{\theta}}{k v}$ to see how chaning $\frac{i\ D_{\theta}}{k v}$ affects the value of sigma.

The graph_against_k.py code graphs Gani and Giso_u against the wave number (k). This code also calulates the values of the imaginary frequcny (Im $\omega$) and plots it against the wave number. Note: this code may contain some errors so it should be double checked at some time.

__In Giso_u:__

The Get_Giso_u Python code uses the output.txt file to solve for $\frac{G^{iso}}{u}$ after finding the value of $D_{\theta}$ for a specific number of species. This is done by using a Reimann sum over values of velocity in place of using an integral.

The Graphing_Giso_u Python code uses the data from Get_Giso_u to graph the $G^{iso}_u$ data in terms of variables in the output.txt file. These variables include time, H neutral fraction, He neutral fraction, electron temperature, ionized H temperature, and ionized He temperature.

__In Gani:__

The Get_Gani Python code uses the data from the output.txt file to solve for $G^{ani}$. This uses a very similar method to Get_Giso_u in order to compute the values of $G^{ani}$. Again, a Reimann sum was used to substitute for the integral over v.

The Graphing_Gani code uses the data from Get_Gani to graph the $G^{ani}$ data and the neutral fraction of hydrogen and helium in terms of the slab number ($j$). It also graphs other aspects of the $G^{ani}$ expression including the source term $S_{2,0}$ and the multipole moment $a_{2,0}$ over velocity slabs.

# __To reproduce the magnetic field calculation:__

- Use the Reionization-Temp-Modified.c file to create an output file with the reionization front parameters. An example of the model output is given in the repository for the given verion of the Reionization-Temp-Modified.c file. We use source temperature of $5 \times 10^4$ K and an ionization speed of $5 \times 10^8$ cm/s.

- With the reionization output calc_params files in the same directory, run the Get_Gani.py and Get_Giso_u.py files. They will output the $G^{ani}$ and $\frac{G^{iso}}{u}$ calculated data as text files.

- The ouptut files from the reionization front model and calculations for $G^{ani}$ and $\frac{G^{iso}}{u}$ can be used in the plotting routines to create plots which illustrate the evolution of the reionization front, the instability, and potential growth of a magnetic field.
