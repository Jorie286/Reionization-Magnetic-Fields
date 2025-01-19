import math
import time
import numpy as np
import scipy.constants as const
from scipy.linalg import solve_banded
import sys

computation_start_time=time.time() # get the time the code cell started computations

# Open and load the reionization temperatures output into Python
data = np.loadtxt(r'output.txt')
tauHdat = np.loadtxt(r'tauH.txt')
tauHedat = np.loadtxt(r'tauHe.txt')
fracflux = np.loadtxt(r'fracflux.txt')

import calc_params

# Compute the electron number density during reionization.
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
def get_D_theta(Te, THII, THeII, yH, yHe, velocity, j):
    """
    Function to get the value of D_theta (the angular diffusion coefficient) for certian conditions. This function can be used to iterate over a series of slabs in a
    distribution for which we know the velocity in that specific slab, j is used to indicate the slab number being considered. Please note that the inputs should be
    postive otherwise the ouptut will not make sense, the function does not check for good inputs.

    Important note: all physical constants are in units ov MKS for easy conversions.
    
    Input arguments (7)
        required    float or integer-like values
                        Te, temperature of electrons in the reionization front
                        THII, temperature of ionized hydrogen in the reionization front
                        THeII, temperature of ionized helium in the reionization front
                        yH, neutral fraction of hydrogen
                        yHe, neutral fraction of helium
			velocity, given speed of electrons from linearly distributed list
			j, the slab number of the iteration over velocities
    Returns
        the value of D_theta for the specific conditions entered into the function
        
    Date of last revision: December 28, 2025
    """
    n_e = get_n_e(yH, yHe) # electron density function (also the number density of m_e)
    n_b1 = ((3*(1+calc_params.z)**3*calc_params.Omega_b*calc_params.H_o**2)/(8*math.pi*calc_params.G))*calc_params.h*(1-yH) # number density of ionized H
    n_b2 = ((3*(1+calc_params.z)**3*calc_params.Omega_b*calc_params.H_o**2)/(8*math.pi*calc_params.G))*calc_params.he*(1-yHe) # number density of ionized He
    # Calculate the columb logarithm.
    lamda_c = ((3/2)*math.log((calc_params.k_B*calc_params.T)/calc_params.R_y))-((1/2)*math.log(64*math.pi*calc_params.a_o**3*n_e))
    
    # Calculate the velocity dispersion (one for each of the species)
    sigma_b1 = math.sqrt((calc_params.k_B*THII)/(calc_params.m_b1))
    sigma_b2 = math.sqrt((calc_params.k_B*THeII)/(calc_params.m_b2))
    sigma_b3 = math.sqrt((calc_params.k_B*Te)/(calc_params.m_e))
    
    numbers = [n_b1, n_b2, n_e, sigma_b1, sigma_b2, sigma_b3] # List of coefficients to be used in calculating D_theta.
    D_final = 0
    for n in range(0,3): # Iterate through numbers and calculate D_theta for each of the species. Returns the sum over all species.
        D_one = (calc_params.q_a**2*calc_params.q_b**2*numbers[0+n]*lamda_c)/(8*math.pi*calc_params.epsilon_o**2*calc_params.m_e**2*calc_params.velocity[j]**3)
        D_two = (1-(numbers[3+n]**2/calc_params.velocity[j]**2))*math.erf(calc_params.velocity[j]/(math.sqrt(2)*numbers[3+n]))+math.sqrt(2/math.pi)*(numbers[3+n]/calc_params.velocity[j])*math.exp(-calc_params.velocity[j]**2/(2*numbers[3+n]**2))
        D_final = D_one*D_two+D_final
    return D_final
    
def get_A_a(Te, THII, THeII, yH, yHe, velocity):
    """
    Function to get the fraciton of the ionizing photons that are absorbed in slab i (a row of the data) and are in a photon energy bin. This function can be used to 
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
        
    Date of last revision: December 28, 2025
    """
    n_e = get_n_e(yH, yHe) # electron density function (also the number density of m_e)
    n_b1 = ((3*(1+calc_params.z)**3*calc_params.Omega_b*calc_params.H_o**2)/(8*math.pi*calc_params.G))*calc_params.h*(1-yH) # number density of ionized H
    n_b2 = ((3*(1+calc_params.z)**3*calc_params.Omega_b*calc_params.H_o**2)/(8*math.pi*calc_params.G))*calc_params.he*(1-yHe) # number density of ionized He

    # Calculate the columb logarithm.
    lambda_c = ((3/2)*math.log((calc_params.k_B*calc_params.T)/calc_params.R_y))-((1/2)*math.log(64*math.pi*calc_params.a_o**3*n_e))

    # Calculate the velocity dispersion (one for each of the species)
    sigma_b1 = math.sqrt((calc_params.k_B*THII)/(calc_params.m_b1))
    sigma_b2 = math.sqrt((calc_params.k_B*THeII)/(calc_params.m_b2))
    sigma_b3 = math.sqrt((calc_params.k_B*Te)/(calc_params.m_e))
    
    A_numbers = [n_b1, n_b2, n_e, sigma_b1, sigma_b2, sigma_b3, calc_params.m_b1, calc_params.m_b2, calc_params.m_e] # List of coefficients to be used in calculating D_theta.
    # set up variables for computing over species
    A_final = 0
    A_one = 0
    A_two = 0
    A_final_neg = 0
    for a in range(0,3): # Iterate through numbers and calculate A_a for each of the species. Returns the sum over all species.
        
        A_one = (calc_params.q_a**2*calc_params.q_b**2*A_numbers[0+a]*((calc_params.m_e/A_numbers[6+a])+1)*lambda_c)/(4*math.pi*calc_params.epsilon_o**2*calc_params.m_e**2*velocity**2)
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

    Date of last revision: December 28, 2025
    """
    n_e = get_n_e(yH, yHe) # electron density function (also the number density of m_e)
    n_b1 = ((3*(1+calc_params.z)**3*calc_params.Omega_b*calc_params.H_o**2)/(8*math.pi*calc_params.G))*calc_params.h*(1-yH) # number density of ionized H
    n_b2 = ((3*(1+calc_params.z)**3*calc_params.Omega_b*calc_params.H_o**2)/(8*math.pi*calc_params.G))*calc_params.he*(1-yHe) # number density of ionized He
    # Calculate the columb logarithm.
    lambda_c = ((3/2)*math.log((calc_params.k_B*calc_params.T)/calc_params.R_y))-((1/2)*math.log(64*math.pi*calc_params.a_o**3*n_e))
    
    # Calculate the velocity dispersion (one for each of the species)
    sigma_b1 = math.sqrt((calc_params.k_B*THII)/(calc_params.m_b1))
    sigma_b2 = math.sqrt((calc_params.k_B*THeII)/(calc_params.m_b2))
    sigma_b3 = math.sqrt((calc_params.k_B*Te)/(calc_params.m_e))
    Da_numbers = [n_b1, n_b2, n_e, sigma_b1, sigma_b2, sigma_b3, calc_params.m_b1, calc_params.m_b2, calc_params.m_e] # List of coefficients to be used in calculating D_theta.
    Da_final = 0
    Da_one = 0
    Da_two = 0
    
    for d in range(0,3): # Iterate through numbers and calculate A_a for each of the species. Returns the sum over all species.
        Da_one = (calc_params.q_a**2*calc_params.q_b**2*Da_numbers[0+d]*((calc_params.m_e/Da_numbers[6+d])+1)*Da_numbers[3+d]**2*lambda_c)/(4*math.pi*calc_params.epsilon_o**2*calc_params.m_e*Da_numbers[6+d]*velocity**3)
        Da_two = math.erf(velocity/(math.sqrt(2)*Da_numbers[3+d])) - math.sqrt(2/math.pi)*(velocity/Da_numbers[3+d])*math.exp(-(velocity**2)/(2*Da_numbers[3+d]**2))
        Da_final = Da_final + (Da_one*Da_two)
        Da_one = 0
        Da_two = 0
    return Da_final

# Source term
def get_Slm(yH, tauHdat, tauHedat, fracflux, k, i, velocity):
    """
    Function to get the value for the source term, S_{2,0}. This is the only nonzero term in the source equation. This function can be used to iterate over a
    series of slabs in a distribution for which we know the velocity in that specific slab, i is used to indicate the slab number being considered. The inputs should 
    be postive otherwise the ouptut will not make sense, please note that the function does not check for good inputs.

    Important note: all physical constants are in units of MKS for easy conversions.

    Input argument (7)
        required    float or integer-like values
                        yH, neutral fraction of hydrogen
                        tauHdat, hydrogen optical depths
                        tauHedat, helium optical depths
                        fracflux, flux fraction in a photon bin
                        k, distribution of wave numbers
                        i, the bin number (time step)
                        velocity, given speed of electrons from linearly distributed list
    Returns
        the value of the source term for the specific conditions entered into the function

    Date of last revision: January 11, 2025
    """
    E_lambda_H = calc_params.I_H + (1/2)*calc_params.m_e*velocity**2 # Energy of H for a photon energy bin, lambda
    E_lambda_He = calc_params.I_He + (1/2)*calc_params.m_e*velocity**2 # Energy of He for a photon energy bin, lambda
    delta_E_H = E_lambda_H*math.log(4)/calc_params.N_NU # Energy bin width for H
    delta_E_He = E_lambda_He*math.log(4)/calc_params.N_NU # Energy bin width for He
    F = (calc_params.vmax*calc_params.n_H*(1+calc_params.f_He))/(1-calc_params.vmax/const.c) # incident flux
    
    # Get the sum for each element in the tauHdat and tauHedat data
    tautot = tauHdat + tauHedat
    tautot=np.reshape(tautot, (calc_params.N_NU,calc_params.NSLAB)) # make sure that the array is the correct shape
    
    # Determine which energy bin the energy of H or He is in for the given velocity
    r_H = calc_params.N_NU
    r_He = calc_params.N_NU
    for r in range(len(calc_params.E_list)):
        if E_lambda_H <= calc_params.E_list[r]:
            r_H = r
            break
            
    for r in range(len(calc_params.E_list)):
        if E_lambda_He <= calc_params.E_list[r]:
            r_He = r
            break
    
    # Get find the slab number and S_{2,0} for H and He
    if r_H<calc_params.N_NU:
        A_j_H = fracflux[r_H]*math.exp(-np.sum(tautot[r_H,:i]))*((-np.expm1(-tautot[r_H,i]))/calc_params.DNHI)
        Slm_H = -((8*math.pi)/3)*calc_params.n_H*F*A_j_H*(tauHdat[r_H,i]/(tauHdat[r_H,i]+tauHedat[r_H,i]))*(calc_params.m_e/(velocity*delta_E_H))*(1/3)*(math.sqrt((16*math.pi)/5))
    else:
        Slm_H = 0
        
    if r_He<calc_params.N_NU:
        A_j_He = fracflux[r_He]*math.exp(-np.sum(tautot[r_He,:i]))*((-np.expm1(-tautot[r_He,i]))/calc_params.DNHI)
        Slm_He = -((8*math.pi)/3)*calc_params.n_H*F*A_j_He*(tauHedat[r_He,i]/(tauHdat[r_He,i]+tauHedat[r_He,i]))*(calc_params.m_e/(velocity*delta_E_He))*(1/3)*(math.sqrt((16*math.pi)/5))
    else:
        Slm_He = 0
    Slm_tot = Slm_H + Slm_He # Sum over the species in the source term (H and He)
    # append the Slm_tot values to a file for later review and plotting
    f_S = open("S20.txt", "a")
    f_S.write(str(Slm_tot))
    f_S.write("\n")
    f_S.close()
    return Slm_tot

def get_alm(Te, THII, THeII, yH, yHe, tauHdat, tauHedat, fracflux, k, i):
    """
    Function to get the value of a_{l,m} for values of (l, m). Uses matrix algebra to solve. However, since the only nonzero value of a_{l,m} is for l=2, m=0, 
    this is the only one that is computed. The inputs should be postive otherwise the ouptut will not make sense, please note that the function
    does not check for good inputs.

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
                        k, distribution of wave numbers
                        i, the bin number (time step)
    Returns
        the value of a_{l,m} (the multipole moment) for the given l and m

    Date of last revision: January 4, 2025
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
    for j in range(len(calc_params.velocity)):
        D_theta_vals = np.append(D_theta_vals, (6*calc_params.velocity[j]**2*get_D_theta(data[i,5], data[i,7], data[i,13], data[i,2], data[i,3], calc_params.velocity, j)))
        # make the vector for the source terms
        Slm_vals = np.append(Slm_vals, (calc_params.velocity[j]**2*get_Slm(data[i,2], tauHdat, tauHedat, fracflux, calc_params.k[::calc_params.k_step], i, calc_params.velocity[j])))
        
        # create indeicies to check if (j*2)+/-1 will be out of range for velocity[j]
        # since we are using velocity_half which runs a half step above or below velocity, we need to change the indexing of these values to account for it.
        plus_1 = (j*2)+2
        minus_1 = (j*2)

        # ensure that the i+/-1 indicies will not be out of range by checking their values
        # Note: velocity_half has twice the number of values as velocity so each step in velocity_half is a "half step" in velocity
        if j>=len(calc_params.velocity)-1:
            A_v_vals = np.append(A_v_vals, ((get_A_a(data[i,5], data[i,7], data[i,13], data[i,2], data[i,3], calc_params.velocity[j])*(calc_params.velocity[j]**2))/(calc_params.velocity[j-1]-calc_params.velocity[j])))
            D_para_vals_1 = np.append(D_para_vals_1, ((get_D_a(data[i,5], data[i,7], data[i,13], data[i,2], data[i,3], calc_params.velocity_half[plus_1])*(calc_params.velocity_half[plus_1]**2))/((calc_params.velocity[j-1]-calc_params.velocity[j])**2)))
            D_para_vals_2 = np.append(D_para_vals_2, ((get_D_a(data[i,5], data[i,7], data[i,13], data[i,2], data[i,3], calc_params.velocity_half[minus_1])*(calc_params.velocity_half[minus_1]**2))/((calc_params.velocity[j-1]-calc_params.velocity[j])**2)))
        else:
            A_v_vals_plus = np.append(A_v_vals_plus, ((get_A_a(data[i,5], data[i,7], data[i,13], data[i,2], data[i,3], calc_params.velocity[j+1])*(calc_params.velocity[j+1]**2))/(calc_params.velocity[j+1]-calc_params.velocity[j])))
            A_v_vals = np.append(A_v_vals, ((get_A_a(data[i,5], data[i,7], data[i,13], data[i,2], data[i,3], calc_params.velocity[j])*(calc_params.velocity[j]**2))/(calc_params.velocity[j+1]-calc_params.velocity[j])))
            D_para_vals_plus = np.append(D_para_vals_plus, ((get_D_a(data[i,5], data[i,7], data[i,13], data[i,2], data[i,3], calc_params.velocity_half[plus_1+2])*(calc_params.velocity_half[plus_1+2]**2))/((calc_params.velocity[j+1]-calc_params.velocity[j])**2)))
            D_para_vals_1 = np.append(D_para_vals_1, ((get_D_a(data[i,5], data[i,7], data[i,13], data[i,2], data[i,3], calc_params.velocity_half[plus_1])*(calc_params.velocity_half[plus_1]**2))/((calc_params.velocity[j+1]-calc_params.velocity[j])**2)))
        
            D_para_vals_minus = np.append(D_para_vals_minus, ((get_D_a(data[i,5], data[i,7], data[i,13], data[i,2], data[i,3], calc_params.velocity_half[minus_1])*(calc_params.velocity_half[minus_1]**2))/((calc_params.velocity[j+1]-calc_params.velocity[j])**2)))
            D_para_vals_2 = np.append(D_para_vals_2, ((get_D_a(data[i,5], data[i,7], data[i,13], data[i,2], data[i,3], calc_params.velocity_half[minus_1])*(calc_params.velocity_half[minus_1]**2))/((calc_params.velocity[j+1]-calc_params.velocity[j])**2)))
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
    #np.savetxt("matrix_test.csv", matrix, delimiter=",")
    # now that we know what the matrix is and what the vector Slm is, we can solve the equation Slm = matrix*a20
    a20 = np.linalg.solve(matrix, Slm_vals)
    return a20

def compute_for_slab_timestep(Te, THII, THeII, yH, yHe, tauHdat, tauHedat, fracflux, k, i):
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
                        k, distribution of wave numbers
                        i, the bin number (time step)
    Returns
        the value of a_{l,m} (the multipole moment) for the given l and m

    Date of last revision: January 4, 2025
    """
    start_time=time.time() # get the time the function started computing
    alm = get_alm(Te, THII, THeII, yH, yHe, tauHdat, tauHedat, fracflux, k, i)
    # write a_{2,0} data to a file
    f = open("a20.txt", "a")
    for a in alm:
        f.write(str(a))
        f.write("\n")
    f.close() # close the a20 test file
    end_time=time.time() # get the time the funtion finished computing
    
    # print out the total time spent on this funciton
    return alm

# Compute Gani for a specific value of sigma and D_theta.
def get_Gani(Te, THII, THeII, yH, yHe, nHtot, tauHdat, tauHedat, fracflux, alm, i, k, j):
    """
    Function to get the value of Gani for certian conditions. This function can be used to iterate over a series of slabs in a distribution for which we know
    the velocity in that specific slab, i is used to indicate the slab number being considered. The inputs should be postive otherwise the ouptut will not make
    sense, please note that the function does not check for good inputs.
    
    Important note: all physical constants are in units of MKS for easy conversions.
    
    Input arguments (13)
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
                        i, the bin number (time step)
                        k, distribution of wave numbers
			j, the slab number of the iteration over velocities
    Returns
        the value of Gani for the specific conditions entered into the function
        
    Date of last revision: January 4, 2025
    """
    n_e = get_n_e(yH, yHe) # electron density function
    Gani = (1/n_e)*calc_params.velocity[j]**2*get_sigmas(calc_params.n_sigmas, (1j*get_D_theta(Te, THII, THeII, yH, yHe, calc_params.velocity, j))/(k*calc_params.velocity[j]))[1]*(math.sqrt(6)*alm)
    return np.array(Gani)
    
# Computes Gani as a sum over the velocities for a row in output.txt
Gani_final = 0
Gani_data = []
for i in range(0, calc_params.NSLAB)): # Iterate through all the rows of data and compute Gani_final (sum over velocities) for each.
    slab_start_time= time.time()
    for k_index in range(0, calc_params.num_k):
        alm = compute_for_slab_timestep(data[i,5], data[i,7], data[i,13], data[i,2], data[i,3], tauHdat, tauHedat, fracflux, calc_params.k[k_index*calc_params.k_step], i)
        # write a20 results to a test file instead of printing them out
        for j in range(0, calc_params.Nv): # Compute the Reimann sum of velocities for a row of data.
            Gani_compute = get_Gani(data[i,5], data[i,7], data[i,13], data[i,2], data[i,3], calc_params.NHtot, tauHdat, tauHedat, fracflux, alm[j], i, calc_params.k[k_index*calc_params.k_step], j)
            Gani_final = Gani_final + Gani_compute # Compute the Reimann sum in place of the integral.f
            Gani_compute = 0 # Reset Gani_compute so it does not interfere with the following iteration
        Gani_data.append(Gani_final) #Add the computed value of Gani to the list of all Gani computed for each row of data.
        Gani_final = 0 # Clear Gani_final so it does not interfere with the next iteration.
    slab_time = time.time()
    print("Time for slab", i, "to finish was", slab_time-slab_start_time, "seconds.")
computation_end_time=time.time() # get the time the code cell finished
# return the total time it spent calculating the values
print("Time for computation to complete:", computation_end_time-computation_start_time, "seconds")
print(Gani_data)

f_G = open("Gani.txt", "a")
for a in Gani_data: # write each value in Gani_data to a new line in the text file
    f_G.write(str(a))
    f_G.write("\n")
f_G.close()
