import math
import numpy as np
import scipy.constants as const
from scipy.linalg import solve_banded

# Open and load the reionization temperatures output into Python
data = np.loadtxt(r'output.txt')
# get the variables and parameters for the calculation
import calc_params

# Compute Giso/u (the coefficient of proportionality) for a specific value of velocity using get_sigmas and get_D_theta.
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
        
    Date of last revision: October 28, 2024
    """
    n_e = ((3*(1+calc_params.z)**3*calc_params.Omega_b*calc_params.H_o**2)/(8*math.pi*calc_params.G))*(calc_params.h*(1-yH)+calc_params.he*(1-yHe))
    return n_e

def get_Giso_u(Te, THII, THeII, yH, yHe, nHtot, k, i):
    """
    Function to get the value of Giso_u for certian conditions. This function can be used to iterate over a series of slabs in a distribution for which we know
    the velocity in that specific slab, i is used to indicate the slab number being considered. The inputs should be postive otherwise the ouptut will not make
    sense, please note that the function does not check for good inputs.

    Important note: all physical constants are in units ov MKS for easy conversions.
    
    Input arguments (7)
        required    float or integer-like values 
                        Te, temperature of electrons in the reionization front
                        THII, temperature of ionized hydrogen in the reionization front
                        THeII, temperature of ionized helium in the reionization front
                        yH, neutral fraction of hydrogen
                        yHe, neutral fraction of helium
                        nHtot, total number of hydrogen atoms in the distribution
                        k, wavenumbers
                        i, the slab number of the iteration
    Returns
        the value of Giso_u for the specific conditions entered into the function
        
    Date of last revision: October 28, 2024
    """
    n_e = get_n_e(yH, yHe) # electron density function
    sigma_e = math.sqrt((calc_params.k_B*Te)/(calc_params.m_e))
    
    Giso_const = -(1/calc_params.n_e)*((4*math.sqrt(math.pi))/math.sqrt(6))
    Giso = (calc_params.velocity[i]**2)*(get_sigmas(calc_params.n_sigmas, (1j*get_D_theta(calc_params.T, Te, THII, THeII, yH, yHe, i))/(calc_params.k*calc_params.velocity[i]))[0])*((calc_params.n_e*calc_params.velocity[i])/((2*math.pi)**(3/2)*sigma_e**5))*math.exp(-(calc_params.velocity[i]**2)/(2*sigma_e**2))
    Giso_u = Giso_const*Giso
    return np.array(Giso_u)

# Computing D_theta
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
        
    Date of last revision: October 28, 2024
    """
    n_e = get_n_e(yH, yHe) # electron density function (also the number density of m_e)
    n_b1 = ((3*(1+calc_params.z)**3*calc_params.Omega_b*calc_params.H_o**2)/(8*math.pi*calc_params.G))*calc_params.h*(1-yH) # number density of ionized H
    n_b2 = ((3*(1+calc_params.z)**3*calc_params.Omega_b*calc_params.H_o**2)/(8*math.pi*calc_params.G))*calc_params.he*(1-yHe) # number density of ionized He
    n_b3 = calc_params.n_e
    # Calculate the columb logarithm.
    lamda_c = ((3/2)*math.log((calc_params.k_B*calc_params.T)/calc_params.R_y))-((1/2)*math.log(64*math.pi*calc_params.a_o**3*calc_params.n_e))
    
    # Calculate the velocity dispersion (one for each of the species)
    sigma_b1 = math.sqrt((calc_params.k_B*THII)/(calc_params.m_b1))
    sigma_b2 = math.sqrt((calc_params.k_B*THeII)/(calc_params.m_b2))
    sigma_b3 = math.sqrt((calc_params.k_B*Te)/(calc_params.m_e))
    
    numbers = [n_b1, n_b2, n_b3, sigma_b1, sigma_b2, sigma_b3] # List of coefficients to be used in calculating D_theta.
    D_final = 0
    for n in range(0,3): # Iterate through numbers and calculate D_theta for each of the species. Returns the sum over all species.
        D_one = (calc_params.q_a**2*calc_params.q_b**2*numbers[0+n]*lamda_c)/(8*math.pi*calc_params.epsilon_o**2*calc_params.m_a**2*calc_params.velocity[i]**3)
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
        
    Date of last revision: October 28, 2024
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
    return x

# Computes Giso_u as a sum over the velocities for a row in output.txt
Giso_final = 0
Giso_list = []
for j in range(0, len(data[:,0])): #Iterate through all the rows of data and compute Giso_final (sum over velocities) for each.
    for ik in range(0,calc_params.k_step):
        for i in range(0, calc_params.Nv): # Compute the Reimann sum of velocities for a row of data.
            Giso_compute = get_Giso_u(data[j,5], data[j,7], data[j,13], data[j,2], data[j,3], NHtot, calc_params.k[ik*10], i)
            Giso_final = Giso_final + Giso_compute # Compute the Reimann sum in place of the integral.
            Giso_compute = 0 # Reset Giso_compute so it does not interfere with the following iteration
        Giso_list.append(Giso_final) #Add the computed value of Giso_u to the list of all Giso_u computed for each row of data.
        Giso_final = 0 # Clear Giso_final so it does not interfere with the next iteration.
# Create lists of the correct length to fill with the computed values.
real = [None]*len(Giso_list)
imaginary = [None]*len(Giso_list)
# Add the computed values to the lists.
for m in range(0, len(Giso_list)):
    real[m] = Giso_list[m].real
    imaginary[m] = Giso_list[m].imag
# Print out the results.
print(Giso_list)
print(imaginary)
print(real)
