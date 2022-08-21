import numpy as np

g_ss_run_mode="Release"

g_h=6.62607015e-34  #(J·s)
g_hbar=g_h/(2*np.pi)   #(J·s)
g_em=9.10956e-31    #(kg)
g_eV=1.6021766208e-19  #1eV (J)  
g_e=1.6021766208e-19   #(C)
g_meV=1e-3*g_eV  #J  (1meV)
g_a_B=5.2917721067e-11 #m Bohr Radius (m)
g_eps_0=8.854187817e-12 #vacuum dielectric constant
g_kB=1.380649e-23 #Boltzmann constant (J/K)
g_c0=299792458 #speed of light (m/s)
g_KB=1024
g_MB=1024*g_KB
g_GB=1024*g_MB
g_TB=1024*g_GB


g_A=1e-10           #Ångstrom
g_nm=1e-9           #nm

g_Pauli_0=np.eye(2,dtype=complex)
g_Pauli_x=np.array([[0,1],\
                    [1,0]],dtype=complex)
g_Pauli_y=np.array([[0,-1j],\
                    [1j,0]],dtype=complex)
g_Pauli_z=np.array([[1,0],\
                    [0,-1]],dtype=complex)

g_Pauli=[g_Pauli_0,g_Pauli_x,g_Pauli_y,g_Pauli_z]