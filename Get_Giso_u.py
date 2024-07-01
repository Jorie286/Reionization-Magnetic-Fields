import math
import numpy as np
import scipy.constants as const

# Open and load the reionization temperatures output into Python
data = np.loadtxt(r'output.txt')

# Create a distribution of velocities in log space.
velocity = np.logspace(1,8,71)

# Compute D_theta
def get_D_theta(b, T):
    k_B = const.k # Boltzman constant
    R_y = const.Rydberg # Rydberg constant (unit of energy)
    a_o = 5.29177210903*(10**-11) # Bohr radius
    m_a = const.m_p # mass proton???
    m_b = const.m_e # mass electron???
    q_a = const.eV # charge proton???
    q_b = -const.eV # charge electron???
    c_o = 1 # what is this????
    n_e = 1 # electron density (what is the value???)
    D_final = 0 # Start the value of D_theta at zero.
    # Calculate the velocity dispersion
    sigma_b = math.sqrt((k_B*T)/m_b)
    # Calculate the columb logarithm.
    lamda_c = ((3/2)*math.log((k_B*T)/R_y))-((1/2)*math.log(64*math.pi*(a_o**3)*n_e))                    
    for i in range(0, len(velocity)):
        n_b = 1 # What type of function should this have??
        # Calculate the first portion of D_theta.
        D_one = ((q_a**2)*(q_b**2)*lamda_c)/(8*math.pi*(c_o**2)*(m_a**2)*(velocity[i]**3))
        # Calculate the second portion of D_theta
        D_two = ((1-((sigma_b**2)/(velocity[i]**2)))*math.erf(velocity[i]/(math.sqrt(2)*sigma_b)))
        +(math.sqrt(2/math.pi)*(sigma_b/velocity[i])*math.exp(-(velocity[i]**2)/(2*(sigma_b**2))))
        D_final = D_final + (D_one*D_two)
    return D_final

# Compute Giso/u for a specific value of D_theta
def get_Giso_u(Te, THII, THeII, yH, yHe, nHtot, k):
    return 0

# Print the results of get_Giso_u
for i in range(np.shape(data)[0]):
    print(get_Giso_u(data[i,5], data[i,7], data[i,13], data[i,2], data[i,3], 200, 10**-12))
