# Using the Giso_list generated in the Get_Giso_u.py file, we can graph Giso_u against several variables.
import matplotlib.pyplot as plt
from scipy.linalg import solve_banded
import numpy as np
import calc_params

# get the sigma_11, sigma_21, and D_theta data
d_theta = np.loadtxt(r'D_theta_vel.txt', dtype=np.complex128)
sigmas_11 = np.loadtxt(r'sigmas_vel_11.txt', dtype=np.complex128)
sigmas_21 = np.loadtxt(r'sigmas_vel_21.txt', dtype=np.complex128)

# plot D_theta/kv against sigma_11 to see the relationship between the two parameters
fig, ax = plt.subplots(figsize=(20, 10))
# plot slab 1500, wavenumber 50 across all velocities
ax.plot(np.imag(d_theta[1500*calc_params.num_k*calc_params.Nv:(1500*calc_params.num_k*calc_params.Nv+calc_params.Nv)]),-sigmas_11[1+1500*calc_params.num_k*calc_params.Nv*2:((1+2*1500*calc_params.num_k*calc_params.Nv)+(2*calc_params.Nv)):2], label="$-\\mathcal{i} \\ \\sigma_{1,1}$ calculated")
ax.hlines(np.sqrt((3*(np.pi**3))/8), min(np.imag(d_theta[1500*calc_params.num_k*calc_params.Nv:(1500*calc_params.num_k*calc_params.Nv+calc_params.Nv)])), max(np.imag(d_theta[1500*calc_params.num_k*calc_params.Nv:(1500*calc_params.num_k*calc_params.Nv+calc_params.Nv)])), linestyles = "--", color="r", label="$-\\mathcal{i} \\ \\sigma_{1,1}=\\sqrt{\\frac{3\\pi^3}{8}}$")
ax.plot(np.imag(d_theta[1500*calc_params.num_k*calc_params.Nv:(1500*calc_params.num_k*calc_params.Nv+calc_params.Nv)]), np.sqrt(np.pi/6)/np.imag(d_theta[1500*calc_params.num_k*calc_params.Nv:(1500*calc_params.num_k*calc_params.Nv+calc_params.Nv)]), ls = "--", c="r", label = "$-\\mathcal{i} \\ \\sigma_{1,1}=\\sqrt{\\frac{\\pi}{6}} \\frac{k v}{\\mathcal{i} \\ D_{\\theta}}$")
ax.set_xlabel("$\\frac{\\mathcal{i} \\ D_{\\theta}}{kv}$ ($s^{-1}$)")
ax.set_ylabel("$-\\mathcal{i} \\ \\sigma_{1,1}$ ($m s^{-1}$)")
ax.set_xscale("log")
ax.set_yscale("log")
ax.legend()
fig.savefig("sigma_11_plt.pdf")

# plot D_theta/kv against sigma_21 to see the relationship between the two parameters
fig, ax = plt.subplots(figsize=(20, 10))
# plot slab 1500, wavenumber 50 across all velocities
ax.plot(np.imag(d_theta[1500*calc_params.num_k*calc_params.Nv:(1500*calc_params.num_k*calc_params.Nv+calc_params.Nv)]), -sigmas_21[1500*calc_params.num_k*calc_params.Nv*2:((2*1500*calc_params.num_k*calc_params.Nv)+(2*calc_params.Nv)):2], label="$-\\sigma_{2,1}$ calculated")
ax.hlines(np.sqrt((10*np.pi)/3), min(np.imag(d_theta[1500*calc_params.num_k*calc_params.Nv:(1500*calc_params.num_k*calc_params.Nv+calc_params.Nv)])), max(np.imag(d_theta[1500*calc_params.num_k*calc_params.Nv:(1500*calc_params.num_k*calc_params.Nv+calc_params.Nv)])), linestyles = "--", color="r", label="$-\\sigma_{2,1}=- \\sqrt{\\frac{10\\pi}{3}}$")
ax.set_xlabel("$\\frac{\\mathcal{i} \\ D_{\\theta}}{kv}$")
ax.set_ylabel("$-\\sigma_{2,1}$")
ax.set_xscale("log")
ax.set_yscale("log")
ax.legend()
fig.savefig("sigma_21_plt.pdf")

def get_sigmas(n, c): # m=1, n=number sigma parameters to be solved for, c=iD_theta/kv
    """
    Funtion to find the value of sigma_{l,m} for a certian number of sigmas. For this function, it is assumed that m=1 for all sigmas, only the value of l changes. The input for n must be a positive whole number for the function to work correctly, please note that it does not check for good input. We add
    a check within the function to prevent it from using D_theta/kv values that will cause unrealistic values of sigmas.

    Important note: all physical constants are in units of MKS for easy conversions.

    Input arguments (2)
        required    float or integer-like values
                        n, the number of sigma_{l,m} parameters we want values for
                        c = (i*D_theta)/(k*v), a constant for which a value can be defined
    Returns
        the values of the first n sigma_{n,1}

    Date of last revision: February 19, 2025
    """
    # Create a zero matrix and fill it with the diagonal part of the tridiagonal matrix
    ab = np.zeros((3,n), dtype = np.complex128)
    for l in range (1, n+1):
        ab[1,l-1] = -l*(l+1)*c # sigma_{l,m} coefficient

    for l in range (1, n):
        ab[0,l] = np.sqrt(((l+2)*l)/((2*l+3)*(2*l+1))) # sigma_{l+1,m} coefficient

    for l in range (2, n+1):
        ab[2,l-2] = np.sqrt(((l+1)*(l-1))/((2*l-1)*(2*l+1))) # sigma_{l-1,m} coefficient

    # Create a zero matrix for the b vector of ab*x=b and fill it with the coefficients of each Y_l,m from the RHS of our equation.
    b = np.zeros((n,), dtype=np.complex128)
    b[0] = (-2*np.sqrt(np.pi))/np.sqrt(6)
    x = solve_banded((1, 1), ab, b) # Solve for the x vector

    if abs(c) <= 1e-3: # compare the absolute value of (i*D_theta)/kv to our cut-off value to prevent unwanted behavior at low values of D_theta/kv
        x[0]=-1j*np.sqrt((3*(np.pi**3))/8)

    return x

d_theta_log = np.logspace(-10, 5, num=71)
# plot D_theta/kv against sigma_21 to see the relationship between the two parameters
sigmas_test=[]
for theta in d_theta_log:
    sigmas_test.append(get_sigmas(calc_params.n_sigmas, 1j*theta)[1])
fig, ax = plt.subplots(figsize=(20, 10))
# plot slab 1500, wavenumber 50 across all velocities
ax.plot(d_theta_log, -np.array(sigmas_test), label="$-\\sigma_{2,1}$ calculated")
ax.hlines(np.sqrt((10*np.pi)/3), min(d_theta_log), max(d_theta_log), linestyles = "--", color="r", label="$-\\sigma_{2,1}=- \\sqrt{\\frac{10\\pi}{3}}$")
ax.set_xlabel("$\\frac{\\mathcal{i} \\ D_{\\theta}}{kv}$")
ax.set_ylabel("$-\\sigma_{2,1}$")
ax.set_xscale("log")
ax.set_yscale("log")
ax.legend()
fig.savefig("sigma_21_test_plt.pdf")
