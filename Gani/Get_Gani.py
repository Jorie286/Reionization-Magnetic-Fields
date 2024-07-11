# A draft of what Get_Gani might look like.

import math
import numpy as np
import scipy.constants as const
from scipy.linalg import solve_banded

# Import the data from the Get_Giso_u file to be used again here.
from Get_Giso_u.py import data
from Get_Giso_u.py import velocity

# Compute the electron number density during reionization.
def get_n_e(yH, yHe):
    z = 7
    Omega_b = 0.046 # Fraction of the universe made of baryonic matter
    H_o = 2.2618e-18 # Hubble constant
    G = const.G # gravitational constant
    n_e = ((3*(1+z)**3*Omega_b*H_o**2)/(8*math.pi*G))*(4.5767e26*(1-yH)+3.6132e25*(1-yHe))
    return n_e

def get_sigmas(n, c): # m=1, n=number sigma parameters to be solved for, c=iD_theta/kv
    
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
def get_D_theta(T, Te, THII, THeII, yH, yHe, i):
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

# Compute the Chandrasekhar dynamical friction formula.
def get_A_a(T, THII, THeII, yH, yHe, i):
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
    A_numbers = [n_b1, n_b2, n_b3, sigma_b1, sigma_b2, sigma_b3, m_b1, m_b2, m_b3] # List of coefficients to be used in calculating D_theta.
    A_final = 0
    for a in range(0,3): # Iterate through numbers and calculate A_a for each of the species. Returns the sum over all species.
        A_one = (q_a**2*q_b**2*A_numbers[0+a]*(m_a/(A_numbers[6+a]+1))*lamda_c)/(4*math.pi*epsilon_o**2*m_a**2*velocity[i]**2)
        A_two = math.erf(velocity[i]/(math.sqrt(2)*A_numbers[3+a])) - math.sqrt(2/math.pi)*(velocity[i]/A_numbers[3+a])*math.exp(-velocity[i]**2/(2*A_numbers[3+a]**2))
        A_final = A_final + A_one*A_two
    return -A_final # The result for A_a(v) is addative inverse of its sum over species.

# Compute the diffusion coefficient for a.
def get_D_a(T, THII, THeII, yH, yHe, i):
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
    
    Da_numbers = [n_b1, n_b2, n_b3, sigma_b1, sigma_b2, sigma_b3, m_b1, m_b2, m_b3] # List of coefficients to be used in calculating D_theta.
    Da_final = 0
    for d in range(0,3): # Iterate through numbers and calculate A_a for each of the species. Returns the sum over all species.
        Da_one = (q_a**2*q_b**2*Da_numbers[0+d]*(m_a/(Da_numbers[6+d]+1))*Da_numbers[3+d]*lamda_c)/(4*math.pi*epsilon_o**2*m_a*Da_numbers[6+d]*velocity[i]**3)
        Da_two = math.erf(velocity[i]/(math.sqrt(2)*Da_numbers[3+a])) - math.sqrt(2/math.pi)*(velocity[i]/Da_numbers[3+a])*math.exp(-velocity[i]**2/(2*Da_numbers[3+a]**2))
        Da_final = Da_final + Da_one*Da_two
    return -Da_final # The result for D_a(v) is the addative inverse of its sum over species.

def get_alm(l, m): # In the equation for Gani l=2, m=0,2,-2.
    # Generate a list of values over which to calculate alm for both theta and phi.
    theta = np.arange(0, 2*math.pi, 0.001)
    phi = np.arange(0,math.pi/2, 0.001)
    alm = 0
    alm_r = 0
    alm_im = 0
    alm_compute=0
    
    # Use an if loop to check for which value of m to use and calculate alm.
    # Use a Reimann sum over theta and phi for all m to calulate alm.
    if m == 0:
        for p in range(0,50):
            a = theta[p]*phi[p] # Need to determine the function for a(theta, phi) to solve for alm!!!
            alm_compute = a*(math.sqrt(5)/math.sqrt(16*math.pi))*(3*math.cos(theta[p])**2-1)
            alm = alm_compute + alm
            alm_compute = 0 # Reset alm_compute so it does not interfere with the next iteration.
        return alm
            
    # For m = +/-2 we need to consider the both the imaginary and real portions for calculation. They can be added back together and returned as an imaginary number.
    elif m == 2:
        for p in range(0,50):
            a = theta[p]*phi[p] # Need to determine the function for a(theta, phi) to solve for alm!!!
            alm_compute_r = a*(math.sqrt(15)/math.sqrt(32*math.pi))*math.sin(theta[p])**2*math.exp((-2*1j*phi[p]).real)
            alm_compute_im = a*(math.sqrt(15)/math.sqrt(32*math.pi))*math.sin(theta[p])**2*math.exp((-2*1j*phi[p]).imag)
            alm_compute = alm_compute + (alm_compute_r + 1j*alm_compute_im)
            alm_compute_r = 0 # Reset alm_compute_r and alm_compute_im so it does not interfere with the next iteration.
            alm_compute_im = 0 
        return alm_compute
    else:
        for p in range(0,50):
            a = theta[p]*phi[p] # Need to determine the function for a(theta, phi) to solve for alm!!!
            alm_compute_r = a*(math.sqrt(15)/math.sqrt(32*math.pi))*math.sin(theta[p])**2*math.exp((2*1j*phi[p]).real)
            alm_compute_im = a*(math.sqrt(15)/math.sqrt(32*math.pi))*math.sin(theta[p])**2*math.exp((2*1j*phi[p]).imag)
            alm_compute = alm_compute + (alm_compute_r + 1j*alm_compute_im)
            alm_compute_r = 0 # Reset alm_compute_r and alm_compute_im so it does not interfere with the next iteration.
            alm_compute_im = 0 
        return alm_compute

# Compute Gani for a specific value of sigma and D_theta.
def get_Gani(Te, THII, THeII, yH, yHe, nHtot, k, i):
    n_e = get_n_e(yH, yHe) # electron density function
    Gani = (1/n_e)*velocity[i]**2*get_sigmas(20, (1j*get_D_theta(5e4, Te, THII, THeII, yH, yHe, i))/(k*velocity[i]))[1]*(math.sqrt(6)*get_alm(2,0) - get_alm(2,2) - get_alm(2,-2))
    return Gani

# Computes Gani as a sum over the velocities for a row in output.txt
Gani_final = 0
Gani_data = []
for j in range(0, len(data[:,0])): #Iterate through all the rows of data and compute Gani_final (sum over velocities) for each.
    for i in range(0, 71): # Compute the Reimann sum of velocities for a row of data.
        Gani_compute = get_Gani(data[j,5], data[j,7], data[j,13], data[j,2], data[j,3], 200, 1e-12, i)
        Gani_final = Gani_final + Gani_compute # Compute the Reimann sum in place of the integral.
        Gani_compute = 0 # Reset Gani_compute so it does not interfere with the following iteration
    Gani_data.append(Gani_final) #Add the computed value of Gani to the list of all Gani computed for each row of data.
    Gani_final = 0 # Clear Gani_final so it does not interfere with the next iteration.
print(Gani_data)
