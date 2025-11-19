# Gotran generated with modifications

import numpy as np
import math

def init_state_values(**values):
    """
    Initialize state values
    """
    phi_M_g_init = -85.85124831753518
    K_e_init = 3.0920573693394466
    Na_e_init = 144.6059647211733
    Cl_e_init = 133.62405379744834

    K_g_init = 99.31378261644232
    Na_g_init = 15.77700575749503
    Cl_g_init = 5.204858020491508

    init_values = np.array([phi_M_g_init, \
                            K_e_init, K_g_init, \
                            Na_e_init, Na_g_init, \
                            Cl_e_init, Cl_g_init, \
                            ], dtype=np.float_)

    # State indices and limit checker
    state_inds = dict([("V_g", 0),
                       ("K_e", 1), ("K_g", 2),
                       ("Na_e", 3), ("Na_g", 4),
                       ("Cl_e", 4), ("Cl_g", 5)])

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
    g_Na_bar = 0            # Na max conductivity (mS/cm**2)
    g_K_bar = 0             # K max conductivity (mS/cm**2)
    g_leak_Na_n = 0.0       # Na leak conductivity (mS/cm**2)
    g_leak_K_n  = 0.0       # K leak conductivity (mS/cm**2)

    g_leak_Na_g = 0.1      # Na leak conductivity (mS/cm**2)
    g_leak_K_g  = 1.696    # K leak conductivity (mS/cm**2)
    g_leak_Cl_g = 0.05     # Cl leak conductivity (mS/cm**2)

    m_K = 1.5              # threshold ECS K (mol/m^3)
    m_Na = 10              # threshold ICS Na (mol/m^3)
    I_max = 10.75975       # max pump strength (muA/cm^2)

    C_M = 1.0              # Faraday's constant (mC/ mol)

    # Set initial parameter values
    init_values = np.array([g_leak_Na_g, g_leak_K_g,
                            g_leak_Cl_g,
                            C_M, 0,
                            m_K, m_Na, I_max], dtype=np.float_)

    # Parameter indices and limit checker
    param_ind = dict([("g_leak_Na_g", 0), ("g_leak_K_g", 1),
                      ("g_leak_Cl_g", 2),
                      ("Cm", 3), ("stim_amplitude", 4),
                      ("m_K", 5), ("m_Na", 6), ("I_max", 7)])

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
    # State indices and limit checker
    state_inds = dict([("V_g", 0),
                       ("K_e", 1), ("K_g", 2),
                       ("Na_e", 3), ("Na_g", 4),
                       ("Cl_e", 5), ("Cl_g", 6)])

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

    param_inds = dict([("g_leak_Na_g", 0), ("g_leak_K_g", 1),
                       ("g_leak_Cl_g", 2),
                       ("Cm", 3), ("stim_amplitude", 4),
                       ("m_K", 5), ("m_Na", 6), ("I_max", 7)])

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
    Compute the right hand side of the\
        hodgkin_huxley_squid_axon_model_1952_original ODE
    """

    # Assign states
    #assert(len(states)) == 4

    # Assign parameters
    #assert(len(parameters)) == 11

    # # Init return args
    # if values is None:
    #     values = np.zeros((4,), dtype=np.float_)
    # else:
    #     assert isinstance(values, np.ndarray) and values.shape == (4,)

    # Physical parameters (PDEs)
    temperature = 307e3            # temperature (m K)
    R = 8.315e3                    # Gas Constant (m J/(K mol))
    F = 96500e3                    # Faraday's constant (mC/ mol)

    ICS_vol = 3.42e-11/2.0         # ICS volume (cm^3)
    ECS_vol = 7.08e-11             # ECS volume (cm^3)
    surface = 2.29e-6              # membrane surface (cm²)

    K_e_init = 3.0920573693394466
    K_g_init = 99.31378261644232

    K_e = states[1]
    K_g = states[2]

    Na_e = states[3]
    Na_g = states[4]

    Cl_e = states[5]
    Cl_g = states[6]

    g_leak_Na_g = parameters[0]
    g_leak_K_g = parameters[1]
    g_leak_Cl_g = parameters[2]

    Cm = parameters[3]
    stim_amplitude = parameters[4]

    m_K = parameters[5]
    m_Na = parameters[6]
    I_max = parameters[7]

    k_dec = 2.8e-9 * F
    stim_amplitude = 0.5

    E_Na = R * temperature / F * np.log(Na_e/Na_g)
    E_K = R * temperature / F * np.log(K_e/K_g)
    E_Cl = - R * temperature / F * np.log(Cl_e/Cl_g)
    E_K_init = R * temperature / F * np.log(K_e_init/K_g_init)

    # Expressions for the membrane component
    i_stim = stim_amplitude * (10 * 1000 < t) * (t < 400 * 1000)

    # Decay factor for [K]_e (m/s)
    j_decay = - k_dec*(K_e - K_e_init)

    i_pump_g = I_max \
             * (K_e / (K_e + m_K)) \
             * (Na_g ** (1.5) / (Na_g ** (1.5) + m_Na ** (1.5)))

    # Set conductance
    dphi = states[0] - E_K
    A = 1 + np.exp(18.4/42.4)                                  # shorthand
    B = 1 + np.exp(-(0.1186e3 + E_K_init)/0.0441e3)            # shorthand
    C = 1 + np.exp((dphi + 0.0185e3)/0.0425e3)                 # shorthand
    D = 1 + np.exp(-(0.1186e3 + states[0])/0.0441e3)           # shorthand
    g_Kir = np.sqrt(K_e/K_e_init)*(A*B)/(C*D)

    # Define and return current
    I_Kir = g_leak_K_g * g_Kir*(states[0] - E_K)

    # Expressions for the Sodium channel component
    i_Na_g = g_leak_Na_g * (states[0] - E_Na) + 3 * i_pump_g

    # Expressions for the Potassium channel component
    i_K_g = I_Kir - 2 * i_pump_g

    # Expressions for the Chloride channel component
    i_Cl_g = g_leak_Cl_g * (states[0] - E_Cl)

    # Expression for phi_M_g
    values[0] = (- i_K_g - i_Na_g - i_Cl_g)/Cm

    # Expression for K_e
    values[1] = (i_K_g + i_stim + j_decay) * surface / (F * ECS_vol)

    # Expression for K_g
    values[2] = - i_K_g  * surface / (F * ICS_vol)

    # Expression for Na_e
    values[3] = (i_Na_g - i_stim - j_decay) * surface / (F * ECS_vol)

    # Expression for Na_g
    values[4] = - i_Na_g * surface / (F * ICS_vol)

    # Expression for Cl_e
    values[5] = - i_Cl_g  * surface / (F * ECS_vol)

    # Expression for Cl_g
    values[6] = i_Cl_g  * surface / (F * ECS_vol)
