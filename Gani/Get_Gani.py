import math
import time
import numpy as np
import scipy.constants as const
from scipy.linalg import solve_banded
from scipy.misc import derivative
import sys

computation_start_time=time.time() # get the time the code cell started computations

# Open and load the reionization temperatures output into Python
data = np.loadtxt(r'output.txt')
tauHdat = np.loadtxt(r'tauH.txt')
tauHedat = np.loadtxt(r'tauHe.txt')
fracflux = np.loadtxt(r'fracflux.txt')

# define the maximum velocity and number of velocities that we want to use for the compuation
vmax = 1.0e8
Nv = 71

# Create a distribution of velocities in linear space.
velocity = np.linspace(vmax/Nv, vmax, Nv)
# make a linear velocity distribution including "half-steps" for get_alm
velocity_half = np.linspace(vmax/Nv, vmax, Nv*2)

# Define necessary constants for all computations
# IMPORTANT NOTE: the scipy.conststants module gives the constants in mks units.
k_B = const.k # Boltzmann constant
R_y = const.Rydberg*const.h*const.c # Rydberg constant (unit of energy)
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
z = 7 # redshift
N_NU = 128 # number of frequency bins
DNHI = 2.5e16
f_He = 0.079 # He abundance

# Ionization energy of hydrogen and helium (in Joules)
I_H = 2.18e-18
I_He = 3.4e-19
m_e = const.m_e # mass of an electron

T=5e4 # reionization front temperature (Kelvin)

# Compute the range of energies for H and He across all frequency bins
# choose the highest and lowest frequencies we want to consider
freq_max = 1e3 # hertz (radio waves)
freq_min = 1e19 # hertz (gamma rays)

# compute the energies that correspond to these frequencies
E_max = const.h*freq_max
E_min = const.h*freq_min

# make a list of energies that we are considering with the same length as the number of frequency bins
E_list = np.linspace(E_min, E_max, N_NU)

# Compute the electron number density during reionization.
def get_n_e(yH, yHe):
    """
    Funtion to find the electron number density (units of electrons m^-3). This function is used as a part of get_Giso_u. The inputs should be postive otherwise
    the ouptut will not make sense, please note that he function does not check for good inputs.
    
    Important note: all physical constants are in units of MKS for easy conversions.
    
    Input arguments (2)
        required    float or integer-like values
                        yH, neutral fraction of hydrogen
                        yHe, neutral fraction of helium
    Returns
        the number density of electrons under the given conditions
        
    Date of last revision: October 28, 2024
    """
    n_e = ((3*(1+z)**3*Omega_b*H_o**2)/(8*math.pi*G))*(4.5767e26*(1-yH)+3.6132e25*(1-yHe))
    return n_e

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

# Computing D_theta
def get_D_theta(Te, THII, THeII, yH, yHe, velocity):
    """
    Function to get the value of D_theta (the angular diffusion coefficient) for certian conditions. This function can be used to iterate over a series of slabs in a
    distribution for which we know the velocity in that specific slab, i is used to indicate the slab number being considered. Please note that the inputs should be
    postive otherwise the ouptut will not make sense, the function does not check for good inputs.
    
    Important note: all physical constants are in units of MKS for easy conversions.

    Input arguments (6)
        required    float or integer-like values
                        Te, temperature of electrons in the reionization front
                        THII, temperature of ionized hydrogen in the reionization front
                        THEII, temperature of ionized helium in the reionization front
                        yH, neutral fraction of hydrogen
                        yHe, neutral fraction of helium
                        velocity, given speed of electrons from linearly distributed list
    Returns
        the value of D_theta for the specific conditions entered into the function
        
    Date of last revision: October 28, 2024
    """
    n_e = get_n_e(yH, yHe) # electron density function (also the number density of m_b3)
    n_b1 = ((3*(1+z)**3*Omega_b*H_o**2)/(8*math.pi*G))*4.5767e26*(1-yH) # number density of ionized H
    n_b2 = ((3*(1+z)**3*Omega_b*H_o**2)/(8*math.pi*G))*3.6132e25*(1-yHe) # number density of ionized He
    n_b3 = n_e
    
    # Calculate the columb logarithm.
    lambda_c = ((3/2)*math.log((k_B*T)/R_y))-((1/2)*math.log(64*math.pi*a_o**3*n_e))
    
    # Calculate the velocity dispersion (one for each of the species)
    sigma_b1 = math.sqrt((k_B*THII)/(m_b1))
    sigma_b2 = math.sqrt((k_B*THeII)/(m_b2))
    sigma_b3 = math.sqrt((k_B*Te)/(m_b3))
    
    numbers = [n_b1, n_b2, n_b3, sigma_b1, sigma_b2, sigma_b3] # List of coefficients to be used in calculating D_theta.
    # set up variables for computing D_theta over species
    D_final = 0
    D_one = 0
    D_two = 0
    for n in range(0,3): # Iterate through numbers and calculate D_theta for each of the species. Returns the sum over all species.
        D_one = (q_a**2*q_b**2*numbers[0+n]*lambda_c)/(8*math.pi*epsilon_o**2*m_a**2*velocity**3)
        D_two = (1-(numbers[3+n]**2/velocity**2))*math.erf(velocity/(math.sqrt(2)*numbers[3+n]))+math.sqrt(2/math.pi)*(numbers[3+n]/velocity)*math.exp(-(velocity**2)/(2*numbers[3+n]**2))
        D_final = D_one*D_two+D_final
        # reset D_one and D_two for the next iteration.
        D_one = 0
        D_two = 0
    return D_final

def get_A_a(Te, THII, THeII, yH, yHe, velocity):
    """
    Function to get the fraciton of the ionizing photons that are absorbed in slab j (a row of the data) and are in a photon energy bin. This function can be used to 
    iterate over a series of slabs in a distribution for which we know the velocity in that specific slab, i is used to indicate the slab number being considered. 
    Please note that the inputs should be postive otherwise the ouptut will not make sense, the function does not check for good inputs.

    Important note: all physical constants are in units of MKS for easy conversions.
    
    Input arguments (6)
        required    float or integer-like values
                        Te, temperature of electrons in the reionization front
                        THII, temperature of ionized hydrogen in the reionization front
                        THeII, temperature of ionized helium in the reionization front
                        yH, neutral fraction of hydrogen
                        yHe, neutral fraction of helium
                        velocity, given speed of electrons from linearly distributed list
    Returns
        the value of A_a for the specific conditions entered into the function
        
    Date of last revision: October 28, 2024
    """
    n_e = get_n_e(yH, yHe) # electron density function (also the number density of m_b3)
    n_b1 = ((3*(1+z)**3*Omega_b*H_o**2)/(8*math.pi*G))*4.5767e26*(1-yH) # number density of ionized H
    n_b2 = ((3*(1+z)**3*Omega_b*H_o**2)/(8*math.pi*G))*3.6132e25*(1-yHe) # number density of ionized He
    n_b3 = n_e

    # Calculate the columb logarithm.
    lambda_c = ((3/2)*math.log((k_B*T)/R_y))-((1/2)*math.log(64*math.pi*a_o**3*n_e))

    # Calculate the velocity dispersion (one for each of the species)
    sigma_b1 = math.sqrt((k_B*THII)/(m_b1))
    sigma_b2 = math.sqrt((k_B*THeII)/(m_b2))
    sigma_b3 = math.sqrt((k_B*Te)/(m_b3))
    
    A_numbers = [n_b1, n_b2, n_b3, sigma_b1, sigma_b2, sigma_b3, m_b1, m_b2, m_b3] # List of coefficients to be used in calculating D_theta.
    # set up variables for computing over species
    A_final = 0
    A_one = 0
    A_two = 0
    A_final_neg = 0
    for a in range(0,3): # Iterate through numbers and calculate A_a for each of the species. Returns the sum over all species.
        A_one = (q_a**2*q_b**2*A_numbers[0+a]*((m_a/A_numbers[6+a])+1)*lambda_c)/(4*math.pi*epsilon_o**2*m_a**2*velocity**2)
        A_two = math.erf(velocity/(math.sqrt(2)*A_numbers[3+a])) - math.sqrt(2/math.pi)*(velocity/A_numbers[3+a])*math.exp(-(velocity**2)/(2*A_numbers[3+a]**2))
        A_final = A_final + (A_one*A_two)
        # reset for next iteration
        A_one = 0
        A_two = 0
    A_final_neg = -A_final # The result for A_a(v) is addative inverse of its sum over species.
    return A_final_neg

def get_D_a(Te, THII, THeII, yH, yHe, velocity):
    """
    Function to get the value for the along the path diffusion coefficient. This function can be used to iterate over a series of slabs in a distribution for which we 
    know the velocity in that specific slab, i is used to indicate the slab number being considered. The inputs should be postive otherwise the ouptut will not make
    sense, please note that the function does not check for good inputs.

    Important note: all physical constants are in units of MKS for easy conversions.
    
    Input argument (6)
        required    float or integer-like values
                        Te, temperature of electrons in the reionization front
                        THII, temperature of ionized hydrogen in the reionization front
                        THEII, temperature of ionized helium in the reionization front
                        yH, the neutral fraction of hydrogen
                        yHe, the neutral fraction of helium
                        velocity, given speed of electrons from linearly distributed list
    Returns
        the value of the along the path diffusion coefficient for the values entered into the function

    Date of last revision: October 28, 2024
    """
    n_e = get_n_e(yH, yHe) # electron density function (also the number density of m_b3)
    n_b1 = ((3*(1+z)**3*Omega_b*H_o**2)/(8*math.pi*G))*4.5767e26*(1-yH) # number density of ionized H
    n_b2 = ((3*(1+z)**3*Omega_b*H_o**2)/(8*math.pi*G))*3.6132e25*(1-yHe) # number density of ionized He
    n_b3 = n_e
    
    # Calculate the columb logarithm.
    lambda_c = ((3/2)*math.log((k_B*T)/R_y))-((1/2)*math.log(64*math.pi*a_o**3*n_e))
    
    # Calculate the velocity dispersion (one for each of the species)
    sigma_b1 = math.sqrt((k_B*THII)/(m_b1))
    sigma_b2 = math.sqrt((k_B*THeII)/(m_b2))
    sigma_b3 = math.sqrt((k_B*Te)/(m_b3))
    
    Da_numbers = [n_b1, n_b2, n_b3, sigma_b1, sigma_b2, sigma_b3, m_b1, m_b2, m_b3] # List of coefficients to be used in calculating D_theta.
    Da_final = 0
    for d in range(0,3): # Iterate through numbers and calculate A_a for each of the species. Returns the sum over all species.
        Da_one = (q_a**2*q_b**2*Da_numbers[0+d]*((m_a/Da_numbers[6+d])+1)*Da_numbers[3+d]**2*lambda_c)/(4*math.pi*epsilon_o**2*m_a*Da_numbers[6+d]*velocity**3)
        Da_two = math.erf(velocity/(math.sqrt(2)*Da_numbers[3+d])) - math.sqrt(2/math.pi)*(velocity/Da_numbers[3+d])*math.exp(-(velocity**2)/(2*Da_numbers[3+d]**2))
        Da_final = Da_final + Da_one*Da_two
    return Da_final

# Source term
def get_Slm(yH, tauHdat, tauHedat, fracflux, k, j, velocity):
    """
    Function to get the value for the source term, S_{2,0}. This is the only nonzero term in the source equation. (????) This function can be used to iterate over a
    series of slabs in a distribution for which we know the velocity in that specific slab, i is used to indicate the slab number being considered. The inputs should 
    be postive otherwise the ouptut will not make sense, please note that the function does not check for good inputs.

    Important note: all physical constants are in units of MKS for easy conversions.

    Input argument (6)
        required    float or integer-like values
                        yH, neutral fraction of hydrogen
                        tauHdat, hydrogen optical depths
                        tauHedat, helium optical depths
                        fracflux, flux fraction in a photon bin
                        k = 1e-12, wave number
                        j, the bin number (time step)
                        velocity, given speed of electrons from linearly distributed list
    Returns
        the value of the source term for the specific conditions entered into the function

    Date of last revision: October 28, 2024
    """
    E_lambda_H = I_H + (1/2)*m_e*velocity**2 # Energy of H for a photon energy bin, lambda
    E_lambda_He = I_He + (1/2)*m_e*velocity**2 # Energy of He for a photon energy bin, lambda
    delta_E_H = E_lambda_H*math.log(4)/N_NU # Energy bin width for H
    delta_E_He = E_lambda_He*math.log(4)/N_NU # Energy bin width for He
    n_H = ((3*(1+z)**3*Omega_b*H_o**2)/(8*math.pi*G))*4.5767e26*(1-yH) # number density of ionized H
    F = (500000000*n_H*(1+f_He))/(1-500000000/const.c) # incident flux
    
    # Get the sum for each element in the tauHdat and tauHedat data
    tautot = tauHdat + tauHedat
    tautot=np.reshape(tautot, (128,2000)) # make sure that the array is the correct shape
    
    # Determine which energy bin the energy of H or He is in for the given velocity
    for r in range(len(E_list)):
        if E_lambda_H <= E_list[r]:
            r_H = r
            break
            
    for r in range(len(E_list)):
        if E_lambda_He <= E_list[r]:
            r_He = r
            break
    
    # Get S_{2,0} for H and He
    A_j_H = fracflux[r_H]*math.exp(-np.sum(tautot[r_H,:j]))*((-np.expm1(-tautot[r_H,j]))/DNHI)
    Slm_H = -((8*math.pi)/3)*n_H*F*A_j_H*(tauHdat[r_H,j]/(tauHdat[r_H,j]+tauHedat[r_He,j]))*(m_e/(velocity*delta_E_H))*(1/3)*(math.sqrt((16*math.pi)/5))
        
    A_j_He = fracflux[r_He]*math.exp(-np.sum(tautot[r_He,:j]))*((-np.expm1(-tautot[r_He,j]))/DNHI)
    Slm_He = -((8*math.pi)/3)*n_H*F*A_j_He*(tauHedat[r_H,j]/(tauHdat[r_He,j]+tauHedat[r_He,j]))*(m_e/(velocity*delta_E_He))*(1/3)*(math.sqrt((16*math.pi)/5))
    
    Slm_tot = Slm_H + Slm_He # Sum over the species in the source term (H and He)
    return Slm_tot

def get_alm(Te, THII, THeII, yH, yHe, tauHdat, tauHedat, fracflux, k, j):
    """
    Function to get the value of a_{l,m} for values of (l, m). Uses matrix algebra to solve. However, since the only nonzero value of a_{l,m} is for l=2, m=0, 
    this is the only one that is computed. (?????????) The inputs should be postive otherwise the ouptut will not make sense, please note that the function
    does not check for good inputs.

    Important note: all physical constants are in units of MKS for easy conversions.

    Input argument (11)
        required    integer values
                        Te, temperature of electrons in the reionization front
                        THII, temperature of ionized hydrogen in the reionization front
                        THeII, temperature of ionized helium in the reionization front
                        yH, neutral fraction of hydrogen
                        yHe, neutral fraction of helium
                        tauHdat, hydrogen optical depths
                        tauHedat, helium optical depths
                        fracflux, flux fraction in a photon bin
                        k = 1e-12, wave number
                        j, the bin number (time step)
    Returns
        the value of a_{l,m} (the multipole moment) for the given l and m

    Date of last revision: October 28, 2024
    """
    # define all the variables for calculating the matricies
    D_theta_vals=np.array([])
    A_v_vals_plus=np.array([])
    A_v_vals=np.array([])
    D_para_vals_plus=np.array([])
    D_para_vals_1=np.array([])
    D_para_vals_2=np.array([])
    D_para_vals_minus=np.array([])
    Slm_vals=np.array([])
    plus_1 = 0
    minus_1 = 0
    start_time=0
    end_time=0

    # loop over velocities and calulate the values of each component of the matricies to be appended to their arrays    
    for i in range(len(velocity)):
        D_theta_vals = np.append(D_theta_vals, (6*get_D_theta(data[j,5], data[j,7], data[j,13], data[j,2], data[j,3], velocity[i])*velocity[i]**2))
        # make the vector for the source terms
        Slm_vals = np.append(Slm_vals, (get_Slm(data[j,2], tauHdat, tauHedat, fracflux, 1e-12, j, velocity[i])*(velocity[i]**2)))

        # create indeicies to check if (i*2)+/-1 will be out of range for velocity[i]
        # since we are using velocity_half which runs a half step above or below velocity, we need to change the indexing of these values to account for it.
        plus_1 = (i*2)+1
        minus_1 = (i*2)-1

        # ensure that the i+/-1 indicies will not be out of range by checking their values
        # note, velocity_half has twice the number of values as velocity so each step in velocity_half is a "half step" in velocity
        if i>=len(velocity)-1:
            A_v_vals = np.append(A_v_vals, ((get_A_a(data[j,5], data[j,7], data[j,13], data[j,2], data[j,3], velocity[i])*(velocity[i]**2))/(-velocity[i])))
            D_para_vals_1 = np.append(D_para_vals_1, ((get_D_a(data[j,5], data[j,7], data[j,13], data[j,2], data[j,3], velocity_half[plus_1])*(velocity_half[plus_1]**2))/(-velocity[i])))
            D_para_vals_2 = np.append(D_para_vals_2, ((get_D_a(data[j,5], data[j,7], data[j,13], data[j,2], data[j,3], velocity_half[minus_1])*(velocity_half[minus_1]**2))/(-velocity[i])))
        else:
            A_v_vals_plus = np.append(A_v_vals_plus, ((get_A_a(data[j,5], data[j,7], data[j,13], data[j,2], data[j,3], velocity[i+1])*(velocity[i+1]**2))/(velocity[i+1]-velocity[i])))
            A_v_vals = np.append(A_v_vals, ((get_A_a(data[j,5], data[j,7], data[j,13], data[j,2], data[j,3], velocity[i])*(velocity[i]**2))/(velocity[i+1]-velocity[i])))
            D_para_vals_plus = np.append(D_para_vals_plus, ((get_D_a(data[j,5], data[j,7], data[j,13], data[j,2], data[j,3], velocity_half[plus_1+1])*(velocity_half[plus_1+1]**2))/(velocity[i+1]-velocity[i])))
            D_para_vals_1 = np.append(D_para_vals_1, ((get_D_a(data[j,5], data[j,7], data[j,13], data[j,2], data[j,3], velocity_half[plus_1])*(velocity_half[plus_1]**2))/(velocity[i+1]-velocity[i])))
        
            D_para_vals_minus = np.append(D_para_vals_minus, ((get_D_a(data[j,5], data[j,7], data[j,13], data[j,2], data[j,3], velocity_half[minus_1])*(velocity_half[minus_1]**2))/(velocity[i+1]-velocity[i])))
            D_para_vals_2 = np.append(D_para_vals_2, ((get_D_a(data[j,5], data[j,7], data[j,13], data[j,2], data[j,3], velocity_half[minus_1])*(velocity_half[minus_1]**2))/(velocity[i+1]-velocity[i])))
        plus_1 = 0
        minus_1 = 0
        
    # diagonalize the matricies to make a tri-diagonal matrix    
    D_theta_matrix = np.diag(D_theta_vals)
    A_v_matrix = np.diag(A_v_vals)
    D_para_matrix_1 = np.diag(D_para_vals_1)
    D_para_matrix_2 = np.diag(D_para_vals_2)
    D_para_matrix_minus = np.diag(D_para_vals_minus, k=-1)
    A_v_matrix_plus = np.diag(A_v_vals_plus, k=1)
    D_para_matrix_plus = np.diag(D_para_vals_plus, k=1)

    # add the matricies together to get the complete matrix
    matrix = D_theta_matrix - A_v_matrix + D_para_matrix_1 + D_para_matrix_2 - D_para_matrix_minus + A_v_matrix_plus - D_para_matrix_plus
    
    # now that we know what the matrix is and what the vector Slm is, we can solve the equation Slm = matrix*a20
    a20 = np.linalg.solve(matrix, Slm_vals)
    return a20

def compute_for_slab_timestep(Te, THII, THeII, yH, yHe, tauHdat, tauHedat, fracflux, k, j):
    """
    Calls function to get the values of a_{l,m} for each velocity bin.

    Important note: all physical constants are in units of MKS for easy conversions.
    
    Input argument (10)
        required    integer values
                        Te, temperature of electrons in the reionization front
                        THII, temperature of ionized hydrogen in the reionization front
                        THeII, temperature of ionized helium in the reionization front
                        yH, neutral fraction of hydrogen
                        yHe, neutral fraction of helium
                        tauHdat, hydrogen optical depths
                        tauHedat, helium optical depths
                        fracflux, flux fraction in a photon bin
                        k = 1e-12, wave number
                        j, the bin number (time step)
    Returns
        the value of a_{l,m} (the multipole moment) for the given l and m

    Date of last revision: October 28, 2024
    """
    start_time=time.time() # get the time the function started computing
    print("Starting slab", j, "computation.")
    alm = get_alm(Te, THII, THeII, yH, yHe, tauHdat, tauHedat, fracflux, k, j)

    end_time=time.time() # get the time the funtion finished computing
    
    # print out the total time spent on this funciton
    print("Time for slab", j, "alm computation was", end_time-start_time, "seconds.")
    return alm

# Compute Gani for a specific value of sigma and D_theta.
def get_Gani(Te, THII, THeII, yH, yHe, nHtot, tauHdat, tauHedat, fracflux, alm, i, k, j):
    """
    Function to get the value of Gani for certian conditions. This function can be used to iterate over a series of slabs in a distribution for which we know
    the velocity in that specific slab, i is used to indicate the slab number being considered. The inputs should be postive otherwise the ouptut will not make
    sense, please note that the function does not check for good inputs.
    
    Important note: all physical constants are in units of MKS for easy conversions.
    
    Input arguments (12)
        required    float or integer-like values 
                        Te, temperature of electrons in the reionization front
                        THII, temperature of ionized hydrogen in the reionization front
                        THeII, temperature of ionized helium in the reionization front
                        yH, neutral fraction of hydrogen
                        yHe, neutral fraction of helium
                        nHtot = 200, total number of hydrogen atoms in the distribution??
                        tauHdat, hydrogen optical depths
                        tauHedat, helium optical depths
                        fracflux, flux fraction in a photon bin
                        alm, solution of a_{2,0} for all velocities for a slab number
                        i, the slab number of the iteration over velocities
                        k = 1e-12, wave number
                        j, the bin number (time step)
    Returns
        the value of Gani for the specific conditions entered into the function
        
    Date of last revision: October 28, 2024
    """
    n_e = get_n_e(yH, yHe) # electron density function
    # this needs to updated to reflect the solution for a_{l,m}
    Gani = (1/n_e)*velocity[i]**2*get_sigmas(20, (1j*get_D_theta(Te, THII, THeII, yH, yHe, velocity[i]))/(k*velocity[i]))[1]*(math.sqrt(6)*alm)
    return Gani

# Computes Gani as a sum over the velocities for a row in output.txt
Gani_final = 0
Gani_data = []
for j in range(0, len(data[:,0])): #Iterate through all the rows of data and compute Gani_final (sum over velocities) for each.
    alm = compute_for_slab_timestep(data[j,5], data[j,7], data[j,13], data[j,2], data[j,3], tauHdat, tauHedat, fracflux, 1e-12, j)
    print("Values for a_{2,0}:\n", alm)
    for i in range(0, 71): # Compute the Reimann sum of velocities for a row of data.
        Gani_compute = get_Gani(data[j,5], data[j,7], data[j,13], data[j,2], data[j,3], 200, tauHdat, tauHedat, fracflux, alm[i], i, 1e-12, j)
        Gani_final = Gani_final + Gani_compute # Compute the Reimann sum in place of the integral.
        Gani_compute = 0 # Reset Gani_compute so it does not interfere with the following iteration
    Gani_data.append(Gani_final) #Add the computed value of Gani to the list of all Gani computed for each row of data.
    Gani_final = 0 # Clear Gani_final so it does not interfere with the next iteration.
    
computation_end_time=time.time() # get the time the code cell finished
# return the total time it spent calculating the values
print("Time for computation to complete:", computation_end_time-computation_start_time, "seconds")
print(Gani_data)
