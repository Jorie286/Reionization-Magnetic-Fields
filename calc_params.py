# define the maximum velocity and number of velocities that we want to use for the compuation
vmax = 5.0e6
Nv = 71
# Create a distribution of velocities in linear space.
velocity = np.linspace(vmax/Nv, vmax, Nv)
# make a linear velocity distribution including "half-steps" for get_alm
# Note: we need an extra half-step down and half step up
velocity_half = np.linspace((vmax/Nv)-((vmax-(vmax/Nv))/(Nv*2)), vmax+((vmax-(vmax/Nv))/(Nv*2)), (Nv*2)+3)


Nk = 81 # number of wavenumbers we want to have in the distribution
num_k = 10 # number of k values we want to use in the calculation
k_step = 8 # number of values we want to skip over in the distribution between each calculation
kmin = -18 # minimum wavenumber
kmax = -10 # maximum wavenumber
# make a distribution of wavenumbers
k = np.logspace(k_min, k_max, n_k_bins)

# Define necessary constants for all computations
k_B = const.k # Boltzmann constant
R_y = const.Rydberg*const.h*const.c # Rydberg constant (unit of energy)
a_o = 5.29177210903e-11 # Bohr radius
m_a = const.m_e # mass of an electron
m_b1 = const.m_p # mass of HII
m_b2 = 2*const.m_p+2*const.m_n+const.m_e # mass of HeII (ionized once so it still has one electron)
q_a = -const.eV # charge of an electron (also the charge of m_b3)
q_b = const.eV # charge of HII and HeII
epsilon_o = const.epsilon_0 # vacuum permittivity
Omega_b = 0.046 # Fraction of the universe made of baryonic matter during reionization
H_o = 2.2618e-18 # Hubble constant
G = const.G # gravitational constant
z = 7 # redshift
m_e = const.m_e # mass of an electron
DNHI = 2.5e20
f_He = 0.079 # He abundance

h = 4.5767e26 # multiplicative factor of ionozed fraction of hydrogen in determining number density
he = 3.6132e25 # multiplicative factor of ionozed fraction of helium in determining number density

# Ionization energy of hydrogen and helium (in Joules)
I_H = 13.59*const.eV
I_He = 24.687*const.eV

T = 5e4 # reionization front temperature (Kelvin)

# make a list of energies that we are considering with the same length as the number of frequency bins
E_list = I_H* (4**np.linspace(0, 1-(1/N_NU), N_NU))

# timestep from the inital simulation where we gathered the data
Timestep = 12000
# number of frequency bins
N_NU = 128
# define the number of slabs we want to use in the calculaiton
NSLAB = 2000
# total number of hydrogen atoms in the distribution
NHtot = 200
# number of sigma terms we want to calculate
n_sigmas = 20
