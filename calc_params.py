import numpy as np
import scipy.constants as const

# define the maximum velocity and number of velocities that we want to use for the compuation
vmax = 5.0e6
Nv = 71
# Create a distribution of velocities in linear space.
velocity = np.linspace(vmax/Nv, vmax, Nv)
delta_v = (vmax-(vmax/Nv))/Nv # get the change in v of each step in the velocity data
# make a linear velocity distribution including "half-steps" for get_alm
# Note: we need an extra half-step down and half step up
velocity_half = np.linspace((vmax/Nv)-((vmax-(vmax/Nv))/(Nv*2)), vmax+((vmax-(vmax/Nv))/(Nv*2)), (Nv*2)+3)


Nk = 101 # number of wavenumbers we want to have in the distribution
num_k = 101 # number of k values we want to use in the calculation
k_step = 1 # number of values we want to skip over in the distribution between each calculation
kmin = -18 # minimum wavenumber
kmax = -8 # maximum wavenumber
# make a distribution of wavenumbers
k = np.logspace(kmin, kmax, Nk)


# Define necessary constants for all computations
k_B = const.k # Boltzmann constant
R_y = const.Rydberg*const.h*const.c # Rydberg constant (unit of energy)
a_o = 5.29177210903e-11 # Bohr radius
m_e = const.m_e # mass of an electron
m_b1 = const.m_p # mass of HII
m_b2 = 2*const.m_p+2*const.m_n+const.m_e # mass of HeII (ionized once so it still has one electron)
q_a = -const.eV # charge of an electron (also the charge of m_b3)
q_b = const.eV # charge of HII and HeII
epsilon_o = const.epsilon_0 # vacuum permittivity
Omega_b = 0.046 # Fraction of the universe made of baryonic matter during reionization
H_o = 2.2618e-18 # Hubble constant
H_o_km_Mpc=H_o*(3.086e+19) # Hubble constant in km/s/Mpc
G = const.G # gravitational constant
z = 7 # redshift
f_He = 0.079 # He abundance
N_A = const.N_A # avogadro's number
h = 0.76*1e3*N_A # number of electrons in 1 kg of hydrogen (same as number of protons)
he = 0.24*1e3*4*N_A # number of outer-shell electrons in 1 kg of helium
T = 5e4 # reionization front temperature (Kelvin)

# Ionization energy of hydrogen and helium (in Joules)
I_H = 13.59*const.eV
I_He = 24.687*const.eV

DNHI = 2.5e20 # width of a grid cell within the reionization front model
N_NU = 128 # number of frequency bins
Timestep = 12000 # timestep from the inital simulation where we gathered the data
NSLAB = 2000 # define the number of slabs we want to use in the calculaiton
NHtot = 200 # total number of hydrogen atoms in the distribution
n_sigmas = 40 # number of sigma terms we want to calculate


# make a list of energies that we are considering with the same length as the number of frequency bins
E_list = I_H* (4**np.linspace(0, 1-(1/N_NU), N_NU))


Y_p_He = 4*f_He/(1+4*f_He) # primordial mass fraction of helium
n_H = ((3*(1+z)**3*Omega_b*H_o**2)/(8*np.pi*G))*h # total number density of all hydrogen
