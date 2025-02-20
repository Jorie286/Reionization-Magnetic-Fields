# import necessary packages and data
import numpy as np
import calc_params
from scipy.linalg import solve_banded
import math

# Get the data from the reionization front model
data = np.loadtxt(r'output.txt')

def get_n_e(yH, yHe):
    """
    Funtion to find the electron number density (units of electrons m^-3). This function is used as a part of get_Giso_u. The inputs should be postive otherwise
    the ouptut will not make sense, please note that he function does not check for good inputs.

    Important note: all physical constants are in units ov MKS for easy conversions.

    Input arguments (2)
        required    float or integer-like values
                        yH, neutral fraction of hydrogen
                        yHe, neutral fraction of helium
    Returns
        the number density of electrons under the given conditions

    Date of last revision: December 28, 2025
    """
    n_e = ((3*(1+calc_params.z)**3*calc_params.Omega_b*calc_params.H_o**2)/(8*math.pi*calc_params.G))*(calc_params.h*(1-yH)+calc_params.he*(1-yHe))
    return n_e

def get_D_theta(T, Te, THII, THeII, yH, yHe, i):
    """
    Function to get the value of D_theta (the angular diffusion coefficient) for certian conditions. This function can be used to iterate over a series of slabs in a
    distribution for which we know the velocity in that specific slab, i is used to indicate the slab number being considered. Please note that the inputs should be
    postive otherwise the ouptut will not make sense, the function does not check for good inputs.

    Important note: all physical constants are in units ov MKS for easy conversions.

    Input arguments (7)
        required    float or integer-like values
                        T = 5e4 Kelvin, the temperature of the reionization front
                        Te, temperature of electrons in the reionization front
                        THII, temperature of ionized hydrogen in the reionization front
                        THeII, temperature of ionized helium in the reionization front
                        yH, neutral fraction of hydrogen
                        yHe, neutral fraction of helium
                        i, the slab number of the iteration over velocities
    Returns
        the value of D_theta for the specific conditions entered into the function

    Date of last revision: December 28, 2025
    """
    n_e = get_n_e(yH, yHe) # electron density function (also the number density of m_e)
    n_b1 = ((3*(1+calc_params.z)**3*calc_params.Omega_b*calc_params.H_o**2)/(8*math.pi*calc_params.G))*calc_params.h*(1-yH) # number density of ionized H
    n_b2 = ((3*(1+calc_params.z)**3*calc_params.Omega_b*calc_params.H_o**2)/(8*math.pi*calc_params.G))*calc_params.he*(1-yHe) # number density of ionized He
    n_b3 = n_e
    # Calculate the columb logarithm.
    lamda_c = ((3/2)*math.log((calc_params.k_B*calc_params.T)/calc_params.R_y))-((1/2)*math.log(64*math.pi*calc_params.a_o**3*n_e))

    # Calculate the velocity dispersion (one for each of the species)
    sigma_b1 = math.sqrt((calc_params.k_B*THII)/(calc_params.m_b1))
    sigma_b2 = math.sqrt((calc_params.k_B*THeII)/(calc_params.m_b2))
    sigma_b3 = math.sqrt((calc_params.k_B*Te)/(calc_params.m_e))

    numbers = [n_b1, n_b2, n_b3, sigma_b1, sigma_b2, sigma_b3] # List of coefficients to be used in calculating D_theta.
    D_final = 0
    for n in range(0,3): # Iterate through numbers and calculate D_theta for each of the species. Returns the sum over all species.
        D_one = (calc_params.q_a**2*calc_params.q_b**2*numbers[0+n]*lamda_c)/(8*math.pi*calc_params.epsilon_o**2*calc_params.m_e**2*calc_params.velocity[i]**3)
        D_two = (1-(numbers[3+n]**2/calc_params.velocity[i]**2))*math.erf(calc_params.velocity[i]/(math.sqrt(2)*numbers[3+n]))+math.sqrt(2/math.pi)*(numbers[3+n]/calc_params.velocity[i])*math.exp(-calc_params.velocity[i]**2/(2*numbers[3+n]**2))
        D_final = D_one*D_two+D_final
    return D_final

def get_sigmas(n, c): # m=1, n=number sigma parameters to be solved for, c=iD_theta/kv
    """
    Funtion to find the value of sigma_{l,m} for a certian number of sigmas. For this function, it is assumed that m=1 for all sigmas, only the value of l changes.
    This funciton is used as part of the function get_Giso_u. The input for n must be a positive whole number for the function to work correctly, please note that
    it does not check for good input.

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
        ab[0,l] = math.sqrt(((l+2)*l)/((2*l+3)*(2*l+1))) # sigma_{l+1,m} coefficient

    for l in range (2, n+1):
        ab[2,l-2] = math.sqrt(((l+1)*(l-1))/((2*l-1)*(2*l+1))) # sigma_{l-1,m} coefficient

    # Create a zero matrix for the b vector of ab*x=b and fill it with the coefficients of each Y_l,m from the RHS of our equation.
    b = np.zeros((n,), dtype=np.complex128)
    b[0] = (-2*math.sqrt(math.pi))/math.sqrt(6)
    x = solve_banded((1, 1), ab, b) # Solve for the x vector

    if abs(c) <= 1e-3: # compare the absolute value of (i*D_theta)/kv to our cut-off value to prevent unwanted behavior at low values of D_theta/kv
        x[0]=1j*np.sqrt((3*(np.pi**3))/8)

    return x

# make empty lists to temporarily store the data
d_theta_vel = []
sigmas_vel_11 = []
sigmas_vel_21 = []
# calculate and store the sigma and d_theta/kv values in text files
for j in range(0, calc_params.NSLAB): # Iterate through all the rows of data
    for ik in range(0,calc_params.num_k):
        for i in range(0, calc_params.Nv): # Compute the average of the velocities for a row of data
            d_theta_vel.append(1j*get_D_theta(calc_params.T, data[j,5], data[j,7], data[j,13], data[j,2], data[j,3], i)/(calc_params.k[ik*calc_params.k_step]*calc_params.velocity[i]))
            sigmas_vel_11.append(get_sigmas(calc_params.n_sigmas, (d_theta_vel[i]))[0]) # calculate all sigmas for a slab and add them to an array
            sigmas_vel_21.append(get_sigmas(calc_params.n_sigmas, (d_theta_vel[i]))[1])

            # write velocity data to a file
            f = open("D_theta_vel.txt", "a")
            f.write(str(d_theta_vel[i]))
            f.write("\n")
            f.close()

            f = open("sigmas_vel_11.txt", "a")
            f.write(str(np.real(sigmas_vel_11[i])))
            f.write("\n")
            f.write(str(np.imag(sigmas_vel_11[i])))
            f.write("\n")
            f.close()

            f = open("sigmas_vel_21.txt", "a")
            f.write(str(np.real(sigmas_vel_21[i])))
            f.write("\n")
            f.write(str(np.imag(sigmas_vel_21[i])))
            f.write("\n")
            f.close()

        d_theta_vel=[] # clear list
        sigmas_vel_11 = [] # clear list
        sigmas_vel_21 = [] # clear list
