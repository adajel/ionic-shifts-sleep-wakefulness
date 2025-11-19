import os
import dolfin as df
import numpy as np
import mm_glial as ode
from membrane import MembraneModel

import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as colors

cmap = cm.get_cmap('viridis')
norm = colors.Normalize(vmin=0, vmax=11 - 1)
#cmap = cm.get_cmap('tab10')


mesh = df.UnitSquareMesh(2, 2)
V = df.FunctionSpace(mesh, 'Discontinuous Lagrange Trace', 0)

facet_f = df.MeshFunction('size_t', mesh, mesh.topology().dim()-1, 0)
tag = 0

membrane = MembraneModel(ode, facet_f=facet_f, tag=tag, V=V)

V_index_g = ode.state_indices('V_g')
K_e_index = ode.state_indices('K_e')
K_g_index = ode.state_indices('K_g')
Na_e_index = ode.state_indices('Na_e')
Na_g_index = ode.state_indices('Na_g')
Cl_e_index = ode.state_indices('Cl_e')
Cl_g_index = ode.state_indices('Cl_g')

potential_history_g = []
K_e_history = []
K_g_history = []
Na_e_history = []
Na_g_history = []
Cl_e_history = []
Cl_g_history = []

g_syn_bar = 0
stimulus = {'stim_amplitude': g_syn_bar}

dt = 1.0             # ms
Tstop = 600*1000     # ms

for _ in range(int(Tstop/dt)):
    membrane.step_lsoda(dt=dt, stimulus=stimulus)

    potential_history_g.append(1*membrane.states[:, V_index_g])
    K_e_history.append(1*membrane.states[:, K_e_index])
    K_g_history.append(1*membrane.states[:, K_g_index])
    Na_e_history.append(1*membrane.states[:, Na_e_index])
    Na_g_history.append(1*membrane.states[:, Na_g_index])
    Cl_e_history.append(1*membrane.states[:, Cl_e_index])
    Cl_g_history.append(1*membrane.states[:, Cl_g_index])

potential_history_g = np.array(potential_history_g)
K_e_history = np.array(K_e_history)
K_g_history = np.array(K_g_history)
Na_e_history = np.array(Na_e_history)
Na_g_history = np.array(Na_g_history)
Cl_e_history = np.array(Cl_e_history)
Cl_g_history = np.array(Cl_g_history)

print("phi_M_g_init =", potential_history_g[-1, 2])
print("K_e_init =", K_e_history[-1, 2])
print("K_g_init =", K_g_history[-1, 2])
print("Na_e_init =", Na_e_history[-1, 2])
print("Na_g_init =", Na_g_history[-1, 2])
print("Cl_e_init =", Cl_e_history[-1, 2])
print("Cl_g_init =", Cl_g_history[-1, 2])

g_leak_Na_g = 0.1      # Na leak conductivity (mS/cm**2)
g_leak_K_g  = 1.696    # K leak conductivity (mS/cm**2)
g_leak_Cl_g = 0.05     # Cl leak conductivity (mS/cm**2)
I_max_g = 10.75975     # max pump strength (muA/cm^2)

m_K = 1.5              # threshold ECS K (mol/m^3)
m_Na = 10              # threshold ICS Na (mol/m^3)

C_M = 1.0              # Faraday's constant (mC/ mol)

# Physical parameters (PDEs)
temperature = 307e3            # temperature (m K)
R = 8.315e3                    # Gas Constant (m J/(K mol))
F = 96500e3                    # Faraday's constant (mC/ mol)

ICS_vol = 3.42e-11/2.0         # ICS volume (cm^3)
ECS_vol = 7.08e-11             # ECS volume (cm^3)
surface = 2.29e-6              # membrane surface (cm²)

K_e_init = K_e_history[0, 2]
K_g_init = K_g_history[0, 2]
Na_e_init = Na_e_history[0, 2]
Na_g_init = Na_g_history[0, 2]
Cl_e_init = Cl_e_history[0, 2]
Cl_g_init = Cl_g_history[0, 2]

# set conductance
phi_M_g = potential_history_g[:, 2]

K_e = K_e_history[:, 2]
K_g = K_g_history[:, 2]
Na_e = Na_e_history[:, 2]
Na_g = Na_g_history[:, 2]
Cl_e = Cl_e_history[:, 2]
Cl_g = Cl_g_history[:, 2]

E_K_g = R * temperature / F * np.log(K_e/K_g)
E_K_init = R * temperature / F * np.log(K_e_init/K_g_init)
E_Na_g = R * temperature / F * np.log(Na_e/Na_g)

i_pump = I_max_g \
       * (K_e / (K_e + m_K)) \
       * (Na_g ** (1.5) / (Na_g ** (1.5) + m_Na ** (1.5)))

dphi = phi_M_g - E_K_g
A = 1 + np.exp(18.4/42.4)
B = 1 + np.exp(-(0.1186e3 + E_K_init)/0.0441e3)
C = 1 + np.exp((dphi + 0.0185e3)/0.0425e3)
D = 1 + np.exp(-(0.1186e3 + phi_M_g)/0.0441e3)
g_Kir = np.sqrt(K_e/K_e_init)*(A*B)/(C*D)

# define and return current
I_Kir = g_leak_K_g * g_Kir*(phi_M_g - E_K_g)
I_Na = g_leak_Na_g * (phi_M_g - E_Na_g)

time = np.arange(0, Tstop/1000, dt/1000) # convert to seconds

# ODE plots
fig = plt.figure(figsize=(12, 12))
ax = plt.gca()

k_dec = 2.8e-9 * F
j_decay = - k_dec*(K_e - K_e_init)

stim_amplitude = 0.5
i_stim = [stim_amplitude * (10  < t) * (t < 400) for t in time]
i_dec = [- k_dec*(K_e_t - K_e_init) for K_e_t in K_e]

# initial osmolarity (i.e. sum of ions compartment-wise)
osmolarity_e_init = K_e_init + Na_e_init + Cl_e_init
osmolarity_g_init = K_g_init + Na_g_init + Cl_g_init

# osmolarity i.e. sum of ECS concentrations
osmolarity_e = [x + y + z - osmolarity_e_init for x, y, z in zip(K_e, Na_e, Cl_e)]
# osmolarity i.e. sum of glial concentrations
#
osmolarity_g = [x + y + z - osmolarity_g_init for x, y, z in zip(K_g, Na_g, Cl_g)]

# diffetence in osmolarity across membrane
osmolarity_diff = [x - y for x, y in zip(osmolarity_g, osmolarity_e)]

ax1 = fig.add_subplot(4,3,1)
plt.plot(time, potential_history_g[:, 2], linewidth=3)
plt.ylabel(r'Glial membrane potential (mV)')

ax2 = fig.add_subplot(4,3,2)
plt.plot(time, i_stim, linewidth=3, label="stim")
plt.plot(time, i_dec, linewidth=3, label="decay")
plt.ylabel(r'Input currents ($\mu$A/$\rm cm^{2}$)')
plt.legend()

c0 = cmap(norm(0))
c1 = cmap(norm(1))
c2 = cmap(norm(2))
c3 = cmap(norm(3))
c4 = cmap(norm(4))
c5 = cmap(norm(5))
c6 = cmap(norm(6))
c7 = cmap(norm(7))
c8 = cmap(norm(8))
c9 = cmap(norm(9))

ax5 = fig.add_subplot(4,3,4)
plt.plot(time, Na_e_history[:, 2], linewidth=3, color=c1)
plt.ylabel(r'ECS Na$^+$(mM)')

ax6 = fig.add_subplot(4,3,5)
plt.plot(time, K_e_history[:, 2], linewidth=3, color=c2)
plt.ylabel(r'ECS K$^+$ (mM)')

ax6 = fig.add_subplot(4,3,6)
plt.plot(time, Cl_e_history[:, 2], linewidth=3, color=c3)
plt.ylabel(r'ECS Cl$^-$ (mM)')

ax7 = fig.add_subplot(4,3,7)
plt.plot(time, Na_g_history[:, 2],linewidth=3, color=c4)
plt.ylabel(r'Glial Na$^+$ (mM)')
plt.xlabel("time (s)")

ax8 = fig.add_subplot(4,3,8)
plt.plot(time, K_g_history[:, 2],linewidth=3, color=c5)
plt.ylabel(r'Glia K$^+$ (mM)')
plt.xlabel("time (s)")

ax8 = fig.add_subplot(4,3,9)
plt.plot(time, Cl_g_history[:, 2],linewidth=3, color=c6)
plt.ylabel(r'Glia Cl$^-$ (mM)')
plt.xlabel("time (s)")

ax7 = fig.add_subplot(4,3,10)
plt.plot(time, osmolarity_e, linewidth=3, color=c7)
plt.ylabel(r'ECS osmolarity (mM)')
plt.xlabel("time (s)")

ax8 = fig.add_subplot(4,3,11)
plt.plot(time, osmolarity_g, linewidth=3, color=c8)
plt.ylabel(r'Glial osmolarity (mM)')
plt.xlabel("time (s)")

ax8 = fig.add_subplot(4,3,12)
plt.plot(time, osmolarity_diff, linewidth=3, color=c9)
plt.ylabel(r'osmolarity diff (mM)')
plt.xlabel("time (s)")

# make pretty
ax.axis('off')
plt.tight_layout()

# save figure to file
plt.savefig('ode_dynamics.svg', format='svg')
plt.close()
