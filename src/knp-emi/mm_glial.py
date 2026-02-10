# Gotran generated code for the "hodgkin_huxley_squid_axon_model_1952_original" model

import numpy as np
import math

def init_state_values(**values):
    """
    Initialize state values
    """
    # Init values
    phi_M_init = -85.84503411546689

    init_values = np.array([phi_M_init], dtype=np.float64)

    # State indices and limit checker
    state_ind = dict([("V", 0)])

    assert len(init_values) == len(state_ind)

    for state_name, value in values.items():
        if state_name not in state_ind:
            raise ValueError("{0} is not a state.".format(state_name))
        ind = state_ind[state_name]

        # Assign value
        init_values[ind] = value

    return init_values

def init_parameter_values(**values):
    """
    Initialize parameter values
    """
    # Membrane parameters
    g_leak_Na = 0.1         # Na leak conductivity (mS/cm**2)
    g_leak_K  = 1.696       # K leak conductivity (mS/cm**2)
    g_leak_Cl = 0.05        # Cl leak conductivity (mS/cm**2)

    m_K = 1.5               # threshold ECS K (mol/m^3)
    m_Na = 10               # threshold ICS Na (mol/m^3)
    I_max = 10.75975        # max pump strength (muA/cm^2)

    K_e_init = 3.092970607490389
    K_i_init = 99.3100014897692

    # Set initial parameter values
    init_values = np.array([g_leak_Cl, g_leak_Na, g_leak_K,
                            0, 0,
                            0, 0, 0,
                            m_K, m_Na, I_max,
                            K_e_init, K_i_init,
                            0, 0,
                            0, 0,
                            0, 0,
                            0, 0, 0,
                            0],
                            dtype=np.float64)

    # Parameter indices and limit checker
    param_ind = dict([("g_leak_Cl", 0), ("g_leak_Na", 1), ("g_leak_K", 2),
                      ("Cm", 3), ("stim_amplitude", 4),
                      ("I_ch_Na", 5), ("I_ch_K", 6), ("I_ch_Cl", 7),
                      ("m_K", 8), ("m_Na", 9), ("I_max", 10),
                      ("K_e_init", 11), ("K_i_init", 12),
                      ("K_e", 13), ("K_i", 14),
                      ("Na_e", 15), ("Na_i", 16),
                      ("Cl_e", 17), ("Cl_i", 18),
                      ("z_Na", 19), ("z_K", 20), ("z_Cl", 21),
                      ("psi", 22)])

    assert len(init_values) == len(param_ind)

    for param_name, value in values.items():
        if param_name not in param_ind:
            raise ValueError("{0} is not a parameter.".format(param_name))
        ind = param_ind[param_name]

        # Assign value
        init_values[ind] = value

    return init_values

def state_indices(*states):
    """
    State indices
    """
    state_inds = dict([("V", 0)])

    indices = []
    for state in states:
        if state not in state_inds:
            raise ValueError("Unknown state: '{0}'".format(state))
        indices.append(state_inds[state])
    if len(indices)>1:
        return indices
    else:
        return indices[0]

def parameter_indices(*params):
    """
    Parameter indices
    """

    # Parameter indices and limit checker
    param_inds = dict([("g_leak_Cl", 0), ("g_leak_Na", 1), ("g_leak_K", 2),
                      ("Cm", 3), ("stim_amplitude", 4),
                      ("I_ch_Na", 5), ("I_ch_K", 6), ("I_ch_Cl", 7),
                      ("m_K", 8), ("m_Na", 9), ("I_max", 10),
                      ("K_e_init", 11), ("K_i_init", 12),
                      ("K_e", 13), ("K_i", 14),
                      ("Na_e", 15), ("Na_i", 16),
                      ("Cl_e", 17), ("Cl_i", 18),
                      ("z_Na", 19), ("z_K", 20), ("z_Cl", 21),
                      ("psi", 22)])

    indices = []
    for param in params:
        if param not in param_inds:
            raise ValueError("Unknown param: '{0}'".format(param))
        indices.append(param_inds[param])
    if len(indices)>1:
        return indices
    else:
        return indices[0]

from numbalsoda import lsoda_sig
from numba import njit, cfunc, jit
import numpy as np
import timeit
import math

@cfunc(lsoda_sig, nopython=True) 
def rhs_numba(t, states, values, parameters):
    """
    Compute the right hand side of the ODE system
    """

    g_leak_Cl = parameters[0]
    g_leak_Na = parameters[1]
    g_leak_K = parameters[2]
    Cm = parameters[3]
    stim_amplitude = parameters[4]
    I_ch_Na = parameters[5]
    I_ch_K = parameters[6]
    I_ch_Cl = parameters[7]
    m_K = parameters[8]
    m_Na = parameters[9]
    I_max = parameters[10]
    K_e_init = parameters[11]
    K_i_init = parameters[12]
    K_e = parameters[13]
    K_i = parameters[14]
    Na_e = parameters[15]
    Na_i = parameters[16]
    Cl_e = parameters[17]
    Cl_i = parameters[18]
    z_Na = parameters[19]
    z_K = parameters[20]
    z_Cl = parameters[21]
    psi = parameters[22]

    E_Na = 1/psi * 1/z_K * math.log(Na_e/Na_i)
    E_K = 1/psi * 1/z_K * math.log(K_e/K_i)
    E_Cl = 1/psi * 1/z_Cl * math.log(Cl_e/Cl_i)

    # Physical parameters (PDEs)
    temperature = 307e3            # temperature (m K)
    R = 8.315e3                    # Gas Constant (m J/(K mol))
    F = 96500e3                    # Faraday's constant (mC/ mol)

    i_pump = I_max \
           * (K_e / (K_e + m_K)) \
           * (Na_i ** (1.5) / (Na_i ** (1.5) + m_Na ** (1.5)))

    # set conductance
    E_K_init = R * temperature / F * np.log(K_e_init/K_i_init)
    dphi = states[0] - E_K
    A = 1 + np.exp(18.4/42.4)                                  # shorthand
    B = 1 + np.exp(-(0.1186e3 + E_K_init)/0.0441e3)            # shorthand
    C = 1 + np.exp((dphi + 0.0185e3)/0.0425e3)                 # shorthand
    D = 1 + np.exp(-(0.1186e3 + states[0])/0.0441e3)           # shorthand
    g_Kir = np.sqrt(K_e/K_e_init)*(A*B)/(C*D)

    # define and return current
    i_Kir = g_leak_K * g_Kir * (states[0] - E_K)               # umol/(cm^2*ms)

    # Expressions for the Sodium channel component
    i_Na = g_leak_Na * (states[0] - E_Na) + 3 * i_pump

    # Expressions for the Potassium channel component
    i_K = i_Kir - 2 * i_pump

    # Expressions for the Chloride channel component
    i_Cl = g_leak_Cl * (states[0] - E_Cl)

    # set I_ch_Na
    parameters[5] = i_Na
    # set I_ch_K
    parameters[6] = i_K
    # set I_ch_Cl
    parameters[7] = i_Cl

    # update membrane potential
    values[0] = (- i_K - i_Na - i_Cl)/Cm
