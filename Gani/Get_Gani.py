import math
import numpy as np
import scipy.constants as const
from scipy.linalg import solve_banded
from scipy.misc import derivative

# A draft of what Get_Gani might look like (I am not sure how the get_alm function will look and what variables it will have in it).

# Open and load the reionization temperatures output into Python
data = np.loadtxt(r'output.txt')
tauHdat = np.loadtxt(r'tauH.txt')
tauHedat = np.loadtxt(r'tauHe.txt')
fracflux = np.loadtxt(r'fracflux.txt')

# Create a distribution of velocities in log space.
velocity = np.logspace(1,8,71)

# Compute the electron number density during reionization.
def get_n_e(yH, yHe):
    """
    Funtion to find the electron number density (units of electrons m^-3). This function is used as a part of get_Giso_u. The inputs should be postive otherwise
    the ouptut will not make sense, please note that he function does not check for good inputs.
    
    Input arguments (2)
        required    float or integer-like values
                        yH, neutral fraction of hydrogen
                        yHe, neutral fraction of helium
    Returns
        the number density of electrons under the given conditions
        
    Date of last revision: July 12, 2024
    """
    z = 7
    Omega_b = 0.046 # Fraction of the universe made of baryonic matter
    H_o = 2.2618e-18 # Hubble constant
    G = const.G # gravitational constant
    n_e = ((3*(1+z)**3*Omega_b*H_o**2)/(8*math.pi*G))*(4.5767e26*(1-yH)+3.6132e25*(1-yHe))
    return n_e

def get_sigmas(n, c): # m=1, n=number sigma parameters to be solved for, c=iD_theta/kv
    """
    Funtion to find the value of sigma_{l,m} for a certian number of sigmas. For this function, it is assumed that m=1 for all sigmas, only the value of l changes.
    This funciton is used as part of the function get_Giso_u. The input for n must be a positive whole number for the function to work correctly, please note that
    it does not check for good input.
    
    Input arguments (2)
        required    float or integer-like values
                        n, the number of sigma_{l,m} parameters we want values for
                        c = (i*D_theta)/(k*v), a constant for which a value can be defined
    Returns
        the values of the first n sigma_{n,1}
        
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

# Computing D_theta
def get_D_theta(T, Te, THII, THeII, yH, yHe, i):
    """
    Function to get the value of D_theta (the angular diffusion coefficient) for certian conditions. This function can be used to iterate over a series of slabs in a
    distribution for which we know the velocity in that specific slab, i is used to indicate the slab number being considered. Please note that the inputs should be
    postive otherwise the ouptut will not make sense, the function does not check for good inputs.

    Input arguments (7)
        required    float or integer-like values
                        T = 5e4 Kelvin, the temperature of the reionization front???
                        Te, temperature of electrons in the reionization front
                        THII, temperature of ionized hydrogen in the reionization front
                        THEII, temperature of ionized helium in the reionization front
                        yH, neutral fraction of hydrogen
                        yHe, neutral fraction of helium
                        i, the slab number of the iteration over velocities
    Returns
        the value of D_theta for the specific conditions entered into the function
        
    Date of last revision: July 12, 2024
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

def get_A_a(T, THII, THeII, yH, yHe, i):
    """
    Function to get the fraciton of the ionizing photons that are absorbed in slab j (a row of the data) and are in a photon energy bin. This function can be used to 
    iterate over a series of slabs in a distribution for which we know the velocity in that specific slab, i is used to indicate the slab number being considered. 
    Please note that the inputs should be postive otherwise the ouptut will not make sense, the function does not check for good inputs.

    Input arguments (6)
        required    float or integer-like values
                        T = 5e4 Kelvin, the temperature of the reionization front???
                        THII, temperature of ionized hydrogen in the reionization front
                        THeII, temperature of ionized helium in the reionization front
                        yH, neutral fraction of hydrogen
                        yHe, neutral fraction of helium
                        i, the slab number of the iteration over velocities
    Returns
        the value of A_a for the specific conditions entered into the function
        
    Date of last revision: July 11, 2024
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
    A_numbers = [n_b1, n_b2, n_b3, sigma_b1, sigma_b2, sigma_b3, m_b1, m_b2, m_b3] # List of coefficients to be used in calculating D_theta.
    A_final = 0
    for a in range(0,3): # Iterate through numbers and calculate A_a for each of the species. Returns the sum over all species.
        A_one = (q_a**2*q_b**2*A_numbers[0+a]*(m_a/(A_numbers[6+a]+1))*lamda_c)/(4*math.pi*epsilon_o**2*m_a**2*velocity[i]**2)
        A_two = math.erf(velocity[i]/(math.sqrt(2)*A_numbers[3+a])) - math.sqrt(2/math.pi)*(velocity[i]/A_numbers[3+a])*math.exp(-velocity[i]**2/(2*A_numbers[3+a]**2))
        A_final = A_final + A_one*A_two
    return -A_final # The result for A_a(v) is addative inverse of its sum over species.

def get_DA_a(T, THII, THeII, yH, yHe, i):
    """
    Function to get the derivative fraciton of the ionizing photons that are absorbed in slab j (a row of the data) and are in a photon energy bin. This function can
    be used to iterate over a series of slabs in a distribution for which we know the velocity in that specific slab, i is used to indicate the slab number being 
    considered. Please note that the inputs should be postive otherwise the ouptut will not make sense, the function does not check for good inputs.

    Input arguments (6)
        required    float or integer-like values
                        T = 5e4 Kelvin, the temperature of the reionization front???
                        THII, temperature of ionized hydrogen in the reionization front
                        THeII, temperature of ionized helium in the reionization front
                        yH, neutral fraction of hydrogen
                        yHe, neutral fraction of helium
                        i, the slab number of the iteration over velocities
    Returns
        the value of the derivative of A_a for the specific conditions entered into the function
        
    Date of last revision: September 23, 2024
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
    A_numbers = [n_b1, n_b2, n_b3, sigma_b1, sigma_b2, sigma_b3, m_b1, m_b2, m_b3] # List of coefficients to be used in calculating D_theta.
    DA_final = 0
    for a in range(0,3): # Iterate through numbers and calculate DA_a for each of the species. Returns the sum over all species.
        A_one = (q_a**2*q_b**2*A_numbers[0+a]*(m_a/(A_numbers[6+a]+1))*lamda_c)/(4*math.pi*epsilon_o**2*m_a**2*velocity[i]**2)
        A_two = math.exp(-velocity[i]**2/(2*A_numbers[3+a]**2)*((2/(math.sqrt(2*math.pi)*A_numbers[3+a]))-(math.sqrt(2/math.pi)*(1/A_numbers[3+a]))+(math.sqrt(2/math.pi)*((velocity[i]**2)/(A_numbers[3+a]**3))))
        DA_final = DA_final + A_one*A_two
    # The result for A_a(v) is addative inverse of its sum over species.
    return -DA_final
               
def get_D_a(T, THII, THeII, yH, yHe, i):
    """
    Function to get the value for the along the path diffusion coefficient. This function can be used to iterate over a series of slabs in a distribution for which we 
    know the velocity in that specific slab, i is used to indicate the slab number being considered. The inputs should be postive otherwise the ouptut will not make
    sense, please note that the function does not check for good inputs.

    Input argument (6)
        required    float or integer-like values
                        T = 5e4 Kelvin, the temperature of the reionization front???
                        THII, temperature of ionized hydrogen in the reionization front
                        THEII, temperature of ionized helium in the reionization front
                        yH, the neutral fraction of hydrogen
                        yHe, the neutral fraction of helium
                        i, the slab number of the iteration over velocities
    Returns
        the value of the along the path diffusion coefficient for the values entered into the function

    Date of last revision: September 23, 2024
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
    
    Da_numbers = [n_b1, n_b2, n_b3, sigma_b1, sigma_b2, sigma_b3, m_b1, m_b2, m_b3] # List of coefficients to be used in calculating D_theta.
    Da_final = 0
    for d in range(0,3): # Iterate through numbers and calculate A_a for each of the species. Returns the sum over all species.
        Da_one = (q_a**2*q_b**2*Da_numbers[0+d]*(m_a/(Da_numbers[6+d]+1))*Da_numbers[3+d]**2*lamda_c)/(4*math.pi*epsilon_o**2*m_a*Da_numbers[6+d]*velocity[i]**3)
        Da_two = math.erf(velocity[i]/(math.sqrt(2)*Da_numbers[3+a])) - math.sqrt(2/math.pi)*(velocity[i]/Da_numbers[3+a])*math.exp(-velocity[i]**2/(2*Da_numbers[3+a]**2))
        Da_final = Da_final + Da_one*Da_two
    # The result for DA_a(v) is the addative inverse of its sum over species.
    return -Da_final

# Source term
def get_Slm(yH, tauH, tauHe, fracflux, i, k):
    """
    Function to get the value for the source term, S_{2,0}. This is the only nonzero term in the source equation. (????) This function can be used to iterate over a
    series of slabs in a distribution for which we know the velocity in that specific slab, i is used to indicate the slab number being considered. The inputs should 
    be postive otherwise the ouptut will not make sense, please note that the function does not check for good inputs.

    Input argument (6)
        required    float or integer-like values
                        yH, neutral fraction of hydrogen
                        tauH, hydrogen optical depth
                        tauHe, helium optical depth
                        fracflux, flux fraction in a photon bin
                        i, the slab number of the iteration over velocities
                        k = 1e-12, ???????
    Returns
        the value of the source term for the specific conditions entered into the function

    Date of last revision: September 23, 2024
    """
    N_NU = 128 # number of frequency bins
    DNHI = 2.5e16
    f_He = 0.079 # He abundance
    # Ionization energy of hydrogen and helium (in Joules)
    I_H = 2.18e-18
    I_He = 3.4e-19
    m_e = const.m_e # mass of an electron
    H_o = 2.2618e-18 # Hubble constant
    Omega_b = 0.046 # Fraction of the universe made of baryonic matter during reionization
    G = const.G # gravitational constant
    E_lamda_H = I_H + (1/2)*m_e*velocity[i]**2 # Energy of H for a photon energy bin, lambda
    E_lamda_He = I_He + (1/2)*m_e*velocity[i]**2 # Energy of He for a photon energy bin, lambda
    delta_E_H = E_lamda_H*math.log(4)/N_NU # Energy bin width for H
    delta_E_He = E_lamda_He*math.log(4)/N_NU # Energy bin width for He
    n_H = ((3*(1+z)**3*Omega_b*H_o**2)/(8*math.pi*G))*4.5767e26*(1-yH) # number density of ionized H
    F = (500000000*n_H*(1+f_He))/(1-500000000/const.c) # incident flux
    tautot = tauHdat + tauHedat
    
    A_j = fracflux[:, None]*math.exp(-np.cumsum(tautot, axis=1))*((1-math.exp(-tautot))/DNHI)
        
    Slm_H = -((8*math.pi)/3)*n_H*F*A_j*(tauH[k]/(tauH[k]+tauHe[k]))*(m_e/(velocity[i]*delta_E_H))*(1/3)*math.sqrt((16*math.pi)/5)
    Slm_He = -((8*math.pi)/3)*n_H*F*A_j*(tauHe[k]/(tauH[k]+tauHe[k]))*(m_e/(velocity[i]*delta_E_He))*(1/3)*math.sqrt((16*math.pi)/5)
    Slm_tot = Slm_H + Slm_He # Sum over the species in the source term (H and He)
    return Slm_tot

def get_alm(yH, tauH, tauHe, fracflux, T, THII, THeII, yHe, i, k):
    """
    Function to get the value of a_{l,m} for values of (l, m). However, since the only nonzero value of a_{l,m} is for l=2, m=0, this is the only one that
    is computed. (?????????) The inputs should be postive otherwise the ouptut will not make sense, please note that the function does not check for good inputs.

    Input argument (2)
        required    integer values
                        l = 2, the value of l for which Gani is non-zero (for the purposes of our original Gani solution)
                        m = 0, 2, -2, the values of m that determine how Gani behaves (for the purposes of our original Gani solution)
    Returns
        the value of a_{l,m} (the multipole moment) for the given l and m

    Date of last revision: September 23, 2024
    """
    b = 1 # what is b that makes a_{2,0}(0)=0 ????????????
    # define the placeholders used in overleaf to make final equation more neat
    D_v = (get_D_a(T, THII, THeII, yH, yHe, i)/velocity[i]**2)*((-4*velocity[i]*(velocity[i]+b))+(4*velocity[i]**2(velocity[i]+b)**2)-(2*velocity[i]**2))
    B_v = 2*get_A_a(T, THII, THeII, yH, yHe, i)*(velocity[i]+b)
    C_v = -get_DA_a(T, THII, THeII, yH, yHe, i) - (2*get_A_a(T, THII, THeII, yH, yHe, i)/ velocity[i]) - (6*get_D_theta(T, Te, THII, THeII, yH, yHe, i))
    # put everyting together to compute the value of a_{2,0}
    a_20 = get_Slm(yH, tauH, tauHe, fracflux, i, k)/(-D_v - B_v + C_v)
    return a_20

# Compute Gani for a specific value of sigma and D_theta.
def get_Gani(Te, THII, THeII, yH, yHe, nHtot, k, i):
    """
    Function to get the value of Gani for certian conditions. This function can be used to iterate over a series of slabs in a distribution for which we know
    the velocity in that specific slab, i is used to indicate the slab number being considered. The inputs should be postive otherwise the ouptut will not make
    sense, please note that the function does not check for good inputs.

    Input arguments (7)
        required    float or integer-like values 
                        Te, temperature of electrons in the reionization front
                        THII, temperature of ionized hydrogen in the reionization front
                        THEII, temperature of ionized helium in the reionization front
                        yH, neutral fraction of hydrogen
                        yHe, neutral fraction of helium
                        nHtot = 200, total number of hydrogen atoms in the distribution???
                        k = 1e-12, ???????
                        i, the slab number of the iteration over velocities
    Returns
        the value of Gani for the specific conditions entered into the function
        
    Date of last revision: July 12, 2024
    """
    n_e = get_n_e(yH, yHe) # electron density function
    # this needs to updated to reflect the solution for a_{l,m}
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
