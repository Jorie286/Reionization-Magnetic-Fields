import math
import numpy as np
import scipy.constants as const
from scipy.linalg import solve_banded

# Open and load the reionization temperatures output into Python
data = np.loadtxt(r'output.txt')

# Create a distribution of velocities in log space.
velocity = np.logspace(1,8,71)

# Compute Giso/u (the coefficient of proportionality) for a specific value of velocity using get_sigmas and get_D_theta.
def get_n_e(yH, yHe):
    """
    Funtion to find the value of the number density of electrons for .
    
    Input arguments (2)
        required    integer values
                        yH,
                        yHe,
    Returns
        the number density of electrons for 
        
    Date of last revision: July 12, 2024
    """
    z = 7
    Omega_b = 0.046 # Fraction of the universe made of baryonic matter
    H_o = 2.2618e-18 # Hubble constant
    G = const.G # gravitational constant
    n_e = ((3*(1+z)**3*Omega_b*H_o**2)/(8*math.pi*G))*(4.5767e26*(1-yH)+3.6132e25*(1-yHe))
    return n_e

def get_Giso_u(Te, THII, THeII, yH, yHe, nHtot, k, i):
    k_B = const.k # Boltzmann constant
    m_e = const.m_e # mass electron
    n_e = get_n_e(yH, yHe) # electron density function
    sigma_e = math.sqrt((k_B**2*Te)/(m_e**2))
    
    Giso_const = (1/n_e)*((4*math.sqrt(math.pi))/math.sqrt(6))
    Giso = velocity[i]**2*get_sigmas(20, (1j*get_D_theta(5e4, Te, THII, THeII, yH, yHe, i))/(k*velocity[i]))[0]*((n_e*velocity[i])/((2*math.pi)**(3/2)*sigma_e**5))*math.exp(-(velocity[i]**2)/(2*sigma_e**2))
    Giso_u = Giso_const*Giso
    return Giso_u

# Computing D_theta
def get_D_theta(T, Te, THII, THeII, yH, yHe, i):
    """
    Function to get the value of D_theta (the angular diffusion coefficient) for certian conditions.

    Input arguments (7)
        required    float or integer-like values
                        T, the temperature
                        Te, 
                        THII,
                        THEII,
                        yH,
                        yHe,
                        i,
    """
    k_B = const.k # Boltzmann constant
    R_y = const.Rydberg*const.h # Rydberg constant (unit of energy)
    a_o = 5.29177210903e-11 # Bohr radius
    m_a = const.m_e # mass of an electron
    m_b1 = const.m_p # mass of HII
    m_b2 = 2*const.m_p+2*const.m_n+const.m_e # mass of HeII (ionized once so it still has one electron)
    m_b3 = const.m_e # mass of an electron
    q_a = -const.eV # charge of an electron (also the charge of m_b3)
    q_b = const.eV # charge of HII and HeII
    epsilon_o = const.epsilon_0 # vacuum permittivity
    Omega_b = 0.046 # Fraction of the universe made of baryonic matter during reionization
    H_o = 2.2618e-18 # Hubble constant
    G = const.G # gravitational constant
    z = 7
    n_e = get_n_e(yH, yHe) # electron density function (also the number density of m_b3)
    n_b1 = ((3*(1+z)**3*Omega_b*H_o**2)/(8*math.pi*G))*4.5767e26*(1-yH) # number density of ionized H
    n_b2 = ((3*(1+z)**3*Omega_b*H_o**2)/(8*math.pi*G))*3.6132e25*(1-yHe) # number density of ionized He
    n_b3 = n_e
    # Calculate the columb logarithm.
    lamda_c = ((3/2)*math.log((k_B*T)/R_y))-((1/2)*math.log(64*math.pi*a_o**3*n_e))
    
    # Calculate the velocity dispersion (one for each of the species)
    sigma_b1 = math.sqrt((k_B**2*THII)/(m_b1**2))
    sigma_b2 = math.sqrt((k_B**2*THeII)/(m_b2**2))
    sigma_b3 = math.sqrt((k_B**2*Te)/(m_b3**2))
    
    numbers = [n_b1, n_b2, n_b3, sigma_b1, sigma_b2, sigma_b3] # List of coefficients to be used in calculating D_theta.
    D_final = 0
    for n in range(0,3): # Iterate through numbers and calculate D_theta for each of the species. Returns the sum over all species.
        D_one = (q_a**2*q_b**2*numbers[0+n]*lamda_c)/(8*math.pi*epsilon_o**2*m_a**2*velocity[i]**3)
        D_two = (1-(numbers[3+n]**2/velocity[i]**2))*math.erf(velocity[i]/(math.sqrt(2)*numbers[3+n]))+math.sqrt(2/math.pi)*(numbers[3+n]/velocity[i])*math.exp(-velocity[i]**2/(2*numbers[3+n]**2))
        D_final = D_one*D_two+D_final
    return D_final

def get_sigmas(n, c):
    """
    Funtion to find the value of sigma_{l,m} for a certian number of sigmas. For this function,
    it is assumed that m=1 for all sigmas, only the value of l changes.
    
    Input arguments (2)
        required    integer values
                        n, the number of sigma_{l,m} parameters we want values for
                        c = (i*D_theta)/(k*v), a constant for which the value can be defined
    Returns
        the values of the first n sigma_{n,1} using a matrix to solve.
        
    Date of last revision: July 9, 2024
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
    for i in range(0, 71): # Compute the Reimann sum of velocities for a row of data.
        Giso_compute = get_Giso_u(data[j,5], data[j,7], data[j,13], data[j,2], data[j,3], 200, 1e-12, i)
        Giso_final = Giso_final + Giso_compute # Compute the Reimann sum in place of the integral.
        Giso_compute = 0 # Reset Giso_compute so it does not interfere with the following iteration
    Giso_list.append(Giso_final) #Add the computed value of Giso_u to the list of all Giso_u computed for each row of data.
    Giso_final = 0 # Clear Giso_final so it does not interfere with the next iteration.
real = [None]*len(Giso_list)
imaginary = [None]*len(Giso_list)
for m in range(0, len(Giso_list)):
    real[m] = Giso_list[m].real
    imaginary[m] = Giso_list[m].imag
print(Giso_list)
print(imaginary)
print(real)
