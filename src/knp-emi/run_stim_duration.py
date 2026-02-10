#!/usr/bin/python3
from knpemi.emiWeakForm import emi_system, create_functions_emi
from knpemi.knpWeakForm import knp_system, create_functions_knp

from knpemi.pdeSolver import create_solver_emi
from knpemi.pdeSolver import create_solver_knp

from knpemi.utils import set_initial_conditions, setup_membrane_model
from knpemi.utils import interpolate_to_membrane
from knpemi.utils import update_ode_variables
from knpemi.utils import update_pde_variables

import mm_glial as mm_glial
import mm_hh as mm_hh

import dolfinx
import adios4dolfinx
import scifem
from mpi4py import MPI
import numpy as np
import argparse
import yaml

from ufl import (
        ln,
        SpatialCoordinate,
        conditional,
        And,
        lt,
        le,
        gt,
        ge,
)

# Define colors for printing
class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

i_res = "-"
e_res = "+"

comm = MPI.COMM_WORLD

def write_to_file_sub(xdmf, fname, tag, phi, c, ion_list, t):
    # Write potential to file
    xdmf.write_function(phi[tag], t=float(t))
    adios4dolfinx.write_function(fname, phi[tag], time=float(t))

    for idx in range(len(ion_list)):
        # Determine the function source based on the index
        is_last = (idx == len(ion_list) - 1)
        c_tag = ion_list[-1][f'c_{tag}'] if is_last else c[tag][idx]

        # Write concentration to file
        xdmf.write_function(c_tag, t=float(t))
        adios4dolfinx.write_function(fname, c_tag, time=float(t))

    return


def write_to_file_mem(xdmf, fname, tag, mesh, ct, ion_list, subdomain_list, phi_M, c, t):
    # Write potential to file
    xdmf.write_function(phi_M[tag], t=float(t))
    adios4dolfinx.write_function(fname, phi_M[tag], time=float(t))

    # Write traces of concentrations on membrane to file
    for idx, ion in enumerate(ion_list):
        # Determine the function source based on the index
        is_last = (idx == len(ion_list) - 1)
        c_e = ion_list[-1]['c_0'] if is_last else c[0][idx]
        c_i = ion_list[-1][f'c_{tag}'] if is_last else c[tag][idx]
        # Get extra and intracellular traces
        Q = phi_M[tag].function_space
        k_e, k_i = interpolate_to_membrane(c_e, c_i, Q, mesh, ct, subdomain_list, tag)
        # Write to file
        xdmf.write_function(k_e, t=float(t))
        xdmf.write_function(k_i, t=float(t))
        adios4dolfinx.write_function(fname, k_e, time=float(t))
        adios4dolfinx.write_function(fname, k_i, time=float(t))

    return


def solve_odes(c_prev, phi_M_prev, ion_list, stim_params, dt,
        mesh, ct, subdomain_list, k):
    """ Solve ODEs (membrane models) for each membrane tag in each subdomain """

    # Solve ODEs for all cells (i.e. all subdomains but the ECS)
    for tag, subdomain in subdomain_list.items():
        if tag > 0:
            # Get tag and membrane potential
            phi_M_prev_sub = phi_M_prev[tag]
            for mem_model in subdomain['mem_models']:

                # Update ODE variables based on PDE output
                ode_model = mem_model['ode']
                update_ode_variables(
                    ode_model, c_prev, phi_M_prev_sub, ion_list, subdomain_list,
                    mesh, ct, tag, k
                )

                # Solve ODEs
                ode_model.step_lsoda(
                        dt=dt,
                        stimulus=stim_params['stimulus'],
                        stimulus_locator=stim_params['stimulus_locator']
                )

                # Update PDE variables based on ODE output
                ode_model.get_membrane_potential(phi_M_prev_sub)

                # Update src terms for next PDE step based on ODE output
                for ion, I_ch_k in mem_model['I_ch_k'].items():
                    ode_model.get_parameter("I_ch_" + ion, I_ch_k)

    return


def read_mesh(mesh_file):

    # Set ghost mode
    ghost_mode = dolfinx.mesh.GhostMode.shared_facet

    with dolfinx.io.XDMFFile(comm, mesh_file, 'r') as xdmf:
        # Read mesh and cell tags
        mesh = xdmf.read_mesh(ghost_mode=ghost_mode)
        ct = xdmf.read_meshtags(mesh, name='cell_marker')

        # Create facet entities, facet-to-cell connectivity and cell-to-cell connectivity
        mesh.topology.create_entities(mesh.topology.dim-1)
        mesh.topology.create_connectivity(mesh.topology.dim-1, mesh.topology.dim)
        mesh.topology.create_connectivity(mesh.topology.dim, mesh.topology.dim)

        # Read facets
        ft = xdmf.read_meshtags(mesh, name='facet_marker')

    xdmf.close()

    return mesh, ct, ft


def solve_system(config):
    """ Solve system (PDEs and ODEs) """

    mesh_file = config['mesh_file'] # path to mesh file
    fname = config["fname"]         # directory for saving results

    print(f'{bcolors.OKBLUE}Reading mesh from {mesh_file} ...')
    mesh, ct, ft = read_mesh(mesh_file)
    print(f'mesh read. ms{bcolors.ENDC}')

    # Read mesh and create sub-meshes for extra and intracellular domains and
    # for cellular membranes / interfaces (for solving ODEs)

    # Create subdomains (extracellular space and cells) with:
    # - tag (subdomain tag)
    # - membrane tags (all tags marking the membrane of the cell)
    # - ode_models : dictionary with membrane models (value) and corresponding facet tag (key)
    # Note that the tags set here must match the cell tags and facet tags from the mesh
    ECS = {"name":"ECS",
           "tag":0,              # NB! ECS tag must always be zero.
    }
    neuron = {"name":"neuron",
             "tag":1,
             "membrane_tags":[1],
             "ode_models":{1:mm_hh},
    }

    glial = {"name":"glial",
             "tag":2,
             "membrane_tags":[2],
             "ode_models":{2:mm_glial},
    }

    # Extract sub-meshes
    mesh_sub_0, sub_to_parent_0, sub_vertex_to_parent_0, _, _ = scifem.extract_submesh(mesh, ct, ECS['tag'])

    mesh_sub_1, sub_to_parent_1, sub_vertex_to_parent_1, _, _ = scifem.extract_submesh(mesh, ct, neuron['tag'])
    mesh_mem_1, mem_to_parent_1, mem_vertex_to_parent_1, _, _ = scifem.extract_submesh(mesh, ft, neuron['membrane_tags'])

    mesh_sub_2, sub_to_parent_2, sub_vertex_to_parent_2, _, _ = scifem.extract_submesh(mesh, ct, glial['tag'])
    mesh_mem_2, mem_to_parent_2, mem_vertex_to_parent_2, _, _ = scifem.extract_submesh(mesh, ft, glial['membrane_tags'])

    # Set sub meshes ECS domain
    ECS["mesh_sub"] = mesh_sub_0
    ECS["sub_to_parent"] = sub_to_parent_0
    ECS["sub_vertex_to_parent"] = sub_vertex_to_parent_0

    # Set sub meshes neuron domain
    neuron["mesh_sub"] = mesh_sub_1
    neuron["sub_to_parent"] = sub_to_parent_1
    neuron["sub_vertex_to_parent"] = sub_vertex_to_parent_1
    neuron["mesh_mem"] = mesh_mem_1
    neuron["mem_to_parent"] = mem_to_parent_1

    # Set sub meshes neuron domain
    glial["mesh_sub"] = mesh_sub_2
    glial["sub_to_parent"] = sub_to_parent_2
    glial["sub_vertex_to_parent"] = sub_vertex_to_parent_2
    glial["mesh_mem"] = mesh_mem_2
    glial["mem_to_parent"] = mem_to_parent_2

    subdomain_list = {ECS['tag']:ECS, neuron['tag']:neuron, glial['tag']:glial}

    # Time variables (PDEs)
    t = dolfinx.fem.Constant(mesh, 0.0)

    dt = 0.1                         # global time step (ms)
    Tstop = config["Tstop"]          # ms
    n_steps_ODE = 25                 # number of ODE steps

    # Physical parameters
    C_M = 1.0                        # capacitance
    temperature = 307e3              # temperature (mK)
    F = 96500e3                      # Faraday's constant (mC/mol)
    R = 8.315e3                      # Gas Constant (mJ/(K*mol))
    D_Na = 1.33e-8                   # diffusion coefficients Na (cm/ms)
    D_K = 1.96e-8                    # diffusion coefficients K (cm/ms)
    D_Cl = 2.03e-8                   # diffusion coefficients Cl (cm/ms)
    psi = F / (R * temperature)      # shorthand
    C_phi = C_M / dt                 # shorthand

    # Initial values
    K_e_init = 3.092970607490389
    K_n_init = 124.13988964240784
    K_g_init = 99.3100014897692

    Na_e_init = 144.60625137617149
    Na_n_init = 12.850454639128186
    Na_g_init = 15.775818906083778

    Cl_e_init = 133.62525154406637
    Cl_n_init = 5.0
    Cl_g_init = 5.203660274163705

    lambda_i = config["lambda_i"]
    lambda_e = config["lambda_e"]

    # Set background charge / immobile ions
    rho_z = -1
    rho_e = Na_e_init + K_e_init - Cl_e_init
    rho_n = Na_n_init + K_n_init - Cl_n_init
    rho_g = Na_g_init + K_g_init - Cl_g_init

    rho = {'z':rho_z,
             0:dolfinx.fem.Constant(mesh_sub_0, rho_e),
             1:dolfinx.fem.Constant(mesh_sub_1, rho_n),
             2:dolfinx.fem.Constant(mesh_sub_2, rho_g),
    }

    # Set parameters
    physical_parameters = {'dt':dolfinx.fem.Constant(mesh, dt),
                           'n_steps_ODE':dolfinx.fem.Constant(mesh, dt),
                           'F':dolfinx.fem.Constant(mesh, F),
                           'psi':dolfinx.fem.Constant(mesh, psi),
                           'C_phi':dolfinx.fem.Constant(mesh, C_phi),
                           'C_M':dolfinx.fem.Constant(mesh, C_M),
                           'R':dolfinx.fem.Constant(mesh, R),
                           'temperature':dolfinx.fem.Constant(mesh, temperature),
                           'rho':rho,
    }

    # diffusion coefficients for each sub-domain
    D_Na_sub = {0:dolfinx.fem.Constant(mesh_sub_0, D_Na/(lambda_e**2)),
                1:dolfinx.fem.Constant(mesh_sub_1, D_Na/(lambda_i**2)),
                2:dolfinx.fem.Constant(mesh_sub_2, D_Na/(lambda_i**2)),
    }
    D_K_sub = {0:dolfinx.fem.Constant(mesh_sub_0, D_K/(lambda_e**2)),
               1:dolfinx.fem.Constant(mesh_sub_1, D_K/(lambda_i**2)),
               2:dolfinx.fem.Constant(mesh_sub_2, D_K/(lambda_i**2)),
    }
    D_Cl_sub = {0:dolfinx.fem.Constant(mesh_sub_0, D_Cl/(lambda_e**2)),
                1:dolfinx.fem.Constant(mesh_sub_1, D_Cl/(lambda_i**2)),
                2:dolfinx.fem.Constant(mesh_sub_2, D_Cl/(lambda_i**2)),
    }

    # initial concentrations for each sub-domain
    Na_init = {0:dolfinx.fem.Constant(mesh_sub_0, Na_e_init), \
               1:dolfinx.fem.Constant(mesh_sub_1, Na_n_init),
               2:dolfinx.fem.Constant(mesh_sub_2, Na_g_init),
    }
    K_init = {0:dolfinx.fem.Constant(mesh_sub_0, K_e_init), \
              1:dolfinx.fem.Constant(mesh_sub_1, K_n_init),
              2:dolfinx.fem.Constant(mesh_sub_2, K_g_init),
    }
    Cl_init = {0:dolfinx.fem.Constant(mesh_sub_0, Cl_e_init), \
               1:dolfinx.fem.Constant(mesh_sub_1, Cl_n_init),
               2:dolfinx.fem.Constant(mesh_sub_2, Cl_g_init),
    }

    # Create ions (channel conductivity is set below in the membrane model)
    Na = {'c_init':Na_init,
          'bdry': dolfinx.fem.Constant(mesh_sub_0, (0.0, 0.0)),
          'z':1.0,
          'name':'Na',
          'D':D_Na_sub}

    K = {'c_init':K_init,
          'bdry': dolfinx.fem.Constant(mesh_sub_0, (0.0, 0.0)),
         'z':1.0,
         'name':'K',
         'D':D_K_sub}

    Cl = {'c_init':Cl_init,
          'bdry': dolfinx.fem.Constant(mesh_sub_0, (0.0, 0.0)),
          'z':-1.0,
          'name':'Cl',
          'D':D_Cl_sub}

    # Create ion list. NB! The last ion in list will be eliminated
    ion_list = [K, Cl, Na]

    phi, phi_M_prev = create_functions_emi(subdomain_list, degree=1)
    c, c_prev = create_functions_knp(subdomain_list, ion_list, degree=1)

    # Strength of source term
    f_input = config["f_input"]
    f_decay = config["f_decay"] * F * (c_prev[0][0] - K_init[0])

    # Frequency of source term (application of source term)
    end_time = config["end_time"]       # turn source term off after end_time (ms)
    start_time = config["start_time"]   # turn source term off after end_time (ms)

    f_condition = And(ge(t, start_time), le(t, end_time))
    f_source_K = conditional(f_condition, f_input, 0)  - f_decay
    f_source_Na = conditional(f_condition, - f_input, 0) + f_decay

    K["f_source"] = f_source_K
    Na["f_source"] = f_source_Na

    # Set initial conditions for PDE system
    set_initial_conditions(ion_list, subdomain_list, c_prev)

    # Membrane parameters
    g_syn_bar = 0.0                     # synaptic conductivity (S/m**2)
    # Set stimulus ODE
    stimulus = {'stim_amplitude': g_syn_bar}
    stimulus_locator = lambda x: (x[0] < 20e-4)

    # Set membrane parameters
    stim_params = {'stimulus':stimulus,
                   'stimulus_locator':stimulus_locator}

    # setup membrane models for each cell
    mem_models_neuron = setup_membrane_model(
            stim_params, physical_parameters, neuron['ode_models'],
            ft, phi_M_prev[neuron['tag']].function_space, ion_list
    )

    # setup membrane models for each cell
    mem_models_glial = setup_membrane_model(
            stim_params, physical_parameters, glial['ode_models'],
            ft, phi_M_prev[glial['tag']].function_space, ion_list
    )

    # Add membrane models to each cell in subdomain list
    subdomain_list[neuron['tag']]['mem_models'] = mem_models_neuron
    subdomain_list[glial['tag']]['mem_models'] = mem_models_glial

    # Create variational form emi problem
    a_emi, p_emi, L_emi = emi_system(
            mesh, ct, ft, physical_parameters, ion_list, subdomain_list,
            phi, phi_M_prev, c_prev, dt,
    )

    # Create variational form knp problem
    a_knp, p_knp, L_knp = knp_system(
            mesh, ct, ft, physical_parameters, ion_list, subdomain_list,
            phi, phi_M_prev, c, c_prev, dt,
    )

    # Specify entity maps for each sub-mesh to ensure correct assembly
    entity_maps = [sub_to_parent_0, \
                   sub_to_parent_1, mem_to_parent_1,
                   sub_to_parent_2, mem_to_parent_2]

    # Set solver parameters EMI (True is direct, and False is iterate) 
    direct_emi = False
    rtol_emi = 1E-6
    atol_emi = 1E-40
    threshold_emi = 0.9

    # Set solver parameters KNP (True is direct, and False is iterate) 
    direct_knp = False
    rtol_knp = 1E-7
    atol_knp = 2E-40
    threshold_knp = 0.75

    # Create solver emi problem
    problem_emi = create_solver_emi(
            a_emi, L_emi, phi, entity_maps, subdomain_list, comm,
            direct=direct_emi, p=p_emi, atol=atol_emi, rtol=rtol_emi,
            threshold=threshold_emi
    )

    # Create solver knp problem
    problem_knp = create_solver_knp(
            a_knp, L_knp, c, entity_maps, subdomain_list, comm,
            direct=direct_knp, p=p_knp, atol=atol_knp, rtol=rtol_knp,
            threshold=threshold_knp
    )

    # Crate dictionary for storing XDMF files and checkpoint filenames
    xdmf_sub = {}; xdmf_mem = {}
    fname_bp_sub = {}; fname_bp_mem = {}

    # Create files (XDMF and checkpoint) for saving results
    for tag, subdomain in subdomain_list.items():
        xdmf = dolfinx.io.XDMFFile(comm, f"results/{fname}/results_sub_{tag}.xdmf", "w")
        xdmf.write_mesh(subdomain['mesh_sub'])
        adios4dolfinx.write_mesh(f"results/{fname}/checkpoint_sub_{tag}.bp", subdomain['mesh_sub'])
        xdmf_sub[tag] = xdmf
        fname_bp_sub[tag] = f"results/{fname}/checkpoint_sub_{tag}.bp"

        # Write membrane potential to file for all cellular subdomains (i.e. all subdomain but ECS)
        if tag > 0:
            xdmf = dolfinx.io.XDMFFile(comm, f"results/{fname}/results_mem_{tag}.xdmf", "w")
            xdmf.write_mesh(subdomain['mesh_mem'])
            adios4dolfinx.write_mesh(f"results/{fname}/checkpoint_mem_{tag}.bp", subdomain['mesh_mem'])
            xdmf_mem[tag] = xdmf
            fname_bp_mem[tag] = f"results/{fname}/checkpoint_mem_{tag}.bp"

    for k in range(int(round(Tstop/float(dt)))):
        print(f'{bcolors.OKBLUE}solving for t = {float(t):.2f} ms{bcolors.ENDC}')

        # Solve ODEs
        solve_odes(
                c_prev, phi_M_prev, ion_list, stim_params,
                dt, mesh, ct, subdomain_list, k
            )

        # Solve PDEs
        problem_emi.solve()
        problem_knp.solve()

        # Update PDEs variables for next time step
        update_pde_variables(
                c, c_prev, phi, phi_M_prev, physical_parameters,
                ion_list, subdomain_list, mesh, ct,
        )

        # Update time
        t.value = float(t + dt)

        # Write results to file
        for tag, subdomain in subdomain_list.items():
            # concentrations and potentials from previous time step to file
            write_to_file_sub(xdmf_sub[tag], fname_bp_sub[tag], tag, phi, c, ion_list, t)
            # membrane potential to file for all cellular subdomains (i.e. all subdomain but ECS)
            if tag > 0:
                write_to_file_mem(xdmf_mem[tag], fname_bp_mem[tag], tag, mesh, ct, ion_list, subdomain_list, phi_M_prev, c, t)

    # Close XDMF files
    for tag, subdomain in subdomain_list.items():
        xdmf_sub[tag].close()
        if tag > 0:
            xdmf_mem[tag].close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-c",
        metavar="config.yml",
        help="path to config file",
        type=str,
    )
    conf_arg = vars(parser.parse_args())
    config_file_path = conf_arg["c"]

    with open(f"config_files/{config_file_path}.yml") as conf_file:
        config = yaml.load(conf_file, Loader=yaml.FullLoader)

    solve_system(config)
