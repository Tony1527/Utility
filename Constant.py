import numpy as np

g_ss_run_mode="Release"

g_h=6.62607015e-34  #(J·s)
g_hbar=g_h/(2*np.pi)   #(J·s)
g_em=9.1093837015e-31    #(kg)
g_eV=1.6021766e-19  #1eV (J)  
g_e=1.6021766e-19   #(C)
g_meV=1e-3*g_eV  #J  (1meV)
g_a_B=5.29177210e-11 #m Bohr Radius (m)
g_eps_0=8.854187817e-12 #vacuum dielectric constant
g_miu_0=4*np.pi*1e-7    #(N/A^2)
g_kB=1.380649e-23 #Boltzmann constant (J/K)
g_c0=1/np.sqrt(g_eps_0*g_miu_0) #299792458 #3e8 #speed of light (m/s)
g_c=g_c0
g_KB=1024
g_MB=1024*g_KB
g_GB=1024*g_MB
g_TB=1024*g_GB
g_wn_cm=g_h*g_c0*1e2 #Energy-wavenumber (cm^-1)
g_phi_0=g_h/(g_e)    #Magnetic Flux Quantum h/e, in some books, it is h/(2e)
g_R_K=g_h/(g_e**2)   #von Klitzing constant
g_cond_0=1/g_R_K     #1/R_K (Hall Conductance Quantum?)

g_Am=1e-10           #Ångstrom
g_nm=1e-9           #nm
g_um=1e-6           #um
g_mm=1e-3           #mm
g_cm=1e-2           #cm
g_dm=1e-1           #dm
g_m=1               #m


g_Hz=1
g_KHz=1e3
g_MHz=1e6
g_GHz=1e9
g_THz=1e12

g_Pauli_0=np.eye(2,dtype=complex)
g_Pauli_x=np.array([[0,1],\
                    [1,0]],dtype=complex)
g_Pauli_y=np.array([[0,-1j],\
                    [1j,0]],dtype=complex)
g_Pauli_z=np.array([[1,0],\
                    [0,-1]],dtype=complex)

g_Pauli=[g_Pauli_0,g_Pauli_x,g_Pauli_y,g_Pauli_z]