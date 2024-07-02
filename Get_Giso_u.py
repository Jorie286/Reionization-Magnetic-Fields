import math
import numpy as np
import scipy.constants as const
from scipy.linalg import solve_banded
import scipy.integrate as inte

# Open and load the reionization temperatures output into Python
data = np.loadtxt(r'output.txt')

# Create a distribution of velocities in log space.
velocity = np.logspace(1,8,71)

# Computing D_theta
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
    n_b = 1 # What type of function should this have (should this be nHtot)??
    # Calculate the velocity dispersion
    sigma_b = math.sqrt(((k_B**2)*T)/(m_b**2))
    # Calculate the columb logarithm.
    lamda_c = ((3/2)*math.log((k_B*T)/R_y))-((1/2)*math.log(64*math.pi*(a_o**3)*n_e))
    
    # Calculate the first portion of D_theta.
    D_one = ((q_a**2)*(q_b**2)*n_b*lamda_c)/(8*math.pi*(c_o**2)*(m_a**2)*(velocity[i]**3))
    # Calculate the second portion of D_theta
    D_two = ((1-((sigma_b**2)/(velocity[i]**2)))*math.erf(velocity[i]/(math.sqrt(2)*sigma_b)))+(math.sqrt(2/math.pi)*(sigma_b/velocity[i])*math.exp(-(velocity[i]**2)/(2*(sigma_b**2))))
    D_final = (D_one*D_two)
    return D_final

def get_sigmas(n,c): # m=1, n=number sigma parameters to be solved for, c=iD_theta/kv
    
    # Create a zero matrix and fill it with the diagonal part of the tridiagonal matrix
    ab = np.zeros((3,n), dtype = np.complex128)
    for l in range (1, n+1):
        ab[1,l-1] = -l*(l+1)*c # sigma_{l,m} coefficient
        
    for l in range (1, n):
        ab[0,l] = math.sqrt(((l+2)*(l))/(((2*l)+3)*((2*l)+1))) # sigma_{l+1,m} coefficient
        
    for l in range (2, n+1):
        ab[2,l-2] = math.sqrt(((l+1)*(l-1))/(((2*l)-1)*((2*l)+1))) # sigma_{l-1,m} coefficient
    
    # Create a zero matrix for the b vector of ab*x=b and fill it with the coefficients of each Y_l,m from the RHS of our equation.
    b = np.zeros((n,), dtype=np.complex128)
    b[0] = (-2*math.sqrt(math.pi))/math.sqrt(6)
    x = solve_banded((1, 1), ab, b) # Solve for the x vector
    return x

# Compute Giso/u (the coefficient of proportionality) for a specific values of velocity using get_sigmas and get_D_theta.
def get_Giso_u(Te, THII, THeII, yH, yHe, nHtot, k):
    k_B = const.k # Boltzman constant
    m_e = const.m_e # mass electron???
    n_e = 1 #electron density or nHtot???
    sigma_e = math.sqrt(((k_B**2)*Te)/(m_e**2))
    
    Giso_const = (1/n_e)*((4*math.sqrt(math.pi))/math.sqrt(6))
    Giso = (velocity[i]**2)*(get_sigmas(20, get_D_theta(1, 5*(10**4)))[0])*((n_e*velocity[i])/(((2*math.pi)**(3/2))*sigma_e**5))*math.exp(-(velocity[i]**2)/(2*(sigma_e**2)))
    Giso_u = Giso_const*Giso
    return Giso_u

# Should I compute Giso_u using data from every row of output.txt and sum over velocities for each row?
Giso_final = 0
Giso_list = []
for j in range(0, len(data[:,0])): #Iterate through all the rows of data and compute Giso_final (sum over velocities) for each.
    for i in range(0, 71): # Compute the Reimann sum of velocities for a row of data.
        Giso_compute = get_Giso_u(data[j,5], data[j,7], data[j,13], data[j,2], data[j,3], 200, 10**-12)
        Giso_final = Giso_final + Giso_compute # Compute the Reimann sum in place of the integral.
        Giso_compute = 0 # Reset Giso_compute so it does not interfere with the following iteration
    Giso_list.append(Giso_final) #Add the computed value of Giso_u to the list of all Giso_u computed for each row of data.
    Giso_final = 0 # Clear Giso_final so it does not interfere with the next iteration.
print(Giso_list)
