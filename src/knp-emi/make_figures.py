import matplotlib as mpl
from mpl_toolkits.axes_grid1 import make_axes_locatable

import matplotlib.pyplot as plt
import numpy as np
import argparse
import yaml
import os

import dolfinx
import adios4dolfinx
from mpi4py import MPI
import scifem

# Set font & text parameters
font = {'family' : 'serif',
        'weight' : 'bold',
        'size'   : 13}

plt.rc('font', **font)
plt.rc('text', usetex=True)
mpl.rcParams['image.cmap'] = 'jet'

comm = MPI.COMM_WORLD

def get_time_series_sub(checkpoint_fname, point, tag, dt, Tstop):

    # Read mesh
    mesh_sub = adios4dolfinx.read_mesh(checkpoint_fname, MPI.COMM_WORLD)

    # Create function space and functions for storing data
    V = dolfinx.fem.functionspace(mesh_sub, ("CG", 1))
    K = dolfinx.fem.Function(V)
    Cl = dolfinx.fem.Function(V)
    Na = dolfinx.fem.Function(V)
    phi = dolfinx.fem.Function(V)

    # Create list for point evaluation of functions over time
    Nas = []
    Ks = []
    Cls = []
    phis = []

    t = dt
    while t <= Tstop:
        print(f"Reading data, subdomain {tag}, t = {t:.2f}")
        # Read results from file
        adios4dolfinx.read_function(checkpoint_fname, K, time=t, name=f"c_K_{tag}")
        adios4dolfinx.read_function(checkpoint_fname, Cl, time=t, name=f"c_Cl_{tag}")
        adios4dolfinx.read_function(checkpoint_fname, Na, time=t, name=f"c_Na_{tag}")
        adios4dolfinx.read_function(checkpoint_fname, phi, time=t, name=f"phi_{tag}")

        # Append (results) function evaluated in point to list
        Ks.append(scifem.evaluate_function(K, point)[0][0])
        Cls.append(scifem.evaluate_function(Cl, point)[0][0])
        Nas.append(scifem.evaluate_function(Na, point)[0][0])
        phis.append(scifem.evaluate_function(phi, point)[0][0])

        t += dt

    return Nas, Ks, Cls, phis

def get_time_series_mem(checkpoint_fname, point, tag, dt, Tstop):
    # Read mesh
    mesh_mem = adios4dolfinx.read_mesh(checkpoint_fname, MPI.COMM_WORLD)
    # Create function space and function for storing data
    V = dolfinx.fem.functionspace(mesh_mem, ("CG", 1))

    phi_M = dolfinx.fem.Function(V)
    tr_Ki = dolfinx.fem.Function(V)
    tr_Ke = dolfinx.fem.Function(V)
    tr_Nai = dolfinx.fem.Function(V)
    tr_Nae = dolfinx.fem.Function(V)
    tr_Cli = dolfinx.fem.Function(V)
    tr_Cle = dolfinx.fem.Function(V)

    # Create list for point evaluation of functions over time
    phi_Ms = []
    tr_K_es = []
    tr_K_is = []
    tr_Cl_es = []
    tr_Cl_is = []
    tr_Na_es = []
    tr_Na_is = []

    t = dt
    while t <= Tstop:
        print(f"Reading data, membrane {tag}, t = {t:.2f}")

        # Membrane potential
        adios4dolfinx.read_function(checkpoint_fname, phi_M, time=t, name=f"phi_M_{tag}")
        phi_Ms.append(scifem.evaluate_function(phi_M, point)[0][0])

        # Trace of K from ICS
        adios4dolfinx.read_function(checkpoint_fname, tr_Ki, time=t, name=f"c_K_{tag}")
        tr_K_is.append(scifem.evaluate_function(tr_Ki, point)[0][0])

        # Trace of Cl from ICS
        adios4dolfinx.read_function(checkpoint_fname, tr_Cli, time=t, name=f"c_Cl_{tag}")
        tr_Cl_is.append(scifem.evaluate_function(tr_Cli, point)[0][0])

        # Trace of Na from ICS
        adios4dolfinx.read_function(checkpoint_fname, tr_Nai, time=t, name=f"c_Na_{tag}")
        tr_Na_is.append(scifem.evaluate_function(tr_Nai, point)[0][0])

        # Trace of K from ECS
        adios4dolfinx.read_function(checkpoint_fname, tr_Ke, time=t, name=f"c_K_{0}")
        tr_K_es.append(scifem.evaluate_function(tr_Ke, point)[0][0])

        # Trace of Cl from ECS
        adios4dolfinx.read_function(checkpoint_fname, tr_Cle, time=t, name=f"c_Cl_{0}")
        tr_Cl_es.append(scifem.evaluate_function(tr_Cle, point)[0][0])

        # Trace of Na from ECS
        adios4dolfinx.read_function(checkpoint_fname, tr_Nae, time=t, name=f"c_Na_{0}")
        tr_Na_es.append(scifem.evaluate_function(tr_Nae, point)[0][0])

        t += dt

    return phi_Ms, tr_K_es, tr_K_is, tr_Na_es, tr_Na_is, tr_Cl_es, tr_Cl_is

def plot_3D_concentration(fname_in, fname_out, dt, Tstop, x, tag):

    temperature = 300e3 # temperature (K)
    F = 96485e3         # Faraday's constant (C/mol)
    R = 8.314e3         # Gas constant (J/(K*mol))

    time = np.arange(0, Tstop-dt, dt)

    x_M = x['M'][0]
    y_M = x['M'][1]
    z_M = x['M'][2]

    x_i = x['i'][0]
    y_i = x['i'][1]
    z_i = x['i'][2]

    x_e = x['e'][0]
    y_e = x['e'][1]
    z_e = x['e'][2]

    point_e = np.array([[x_e, y_e, z_e]])
    point_i = np.array([[x_i, y_i, z_i]])
    point_M = np.array([[x_M, y_M, z_M]])

    checkpoint_fname_e = f'results/{fname_in}/checkpoint_sub_0.bp'
    checkpoint_fname_i = f'results/{fname_in}/checkpoint_sub_{tag}.bp'
    checkpoint_fname_M = f'results/{fname_in}/checkpoint_mem_{tag}.bp'

    # get timeseries in points
    phi_M, tr_K_e, tr_K_i, tr_Na_e, tr_Na_i, tr_Cl_e, tr_Cl_i = get_time_series_mem(checkpoint_fname_M, point_M, tag, dt, Tstop)
    Na_e, K_e, Cl_e, phi_e = get_time_series_sub(checkpoint_fname_e, point_e, 0, dt, Tstop)
    Na_i, K_i, Cl_i, phi_i = get_time_series_sub(checkpoint_fname_i, point_i, tag, dt, Tstop)

    temperature = 300e3; F = 96485e3; R = 8.314e3
    # Calculate Nernst potentials
    E_Na = R * temperature / F * np.log(np.array(tr_Na_e) / np.array(tr_Na_i))
    E_K = R * temperature / F * np.log(np.array(tr_K_e) / np.array(tr_K_i))

    g_leak_K  = 1.696       # K leak conductivity (mS/cm**2)
    m_K = 1.5               # threshold ECS K (mol/m^3)
    m_Na = 10               # threshold ICS Na (mol/m^3)
    I_max = 10.75975        # max pump strength (muA/cm^2)
    i_pump = I_max * (np.array(tr_K_e) / (np.array(tr_K_e) + m_K)) \
                   * (np.array(tr_Na_i) ** (1.5) / (np.array(tr_Na_i) ** (1.5) + m_Na ** (1.5)))

    # Set conductance
    K_e_init = 3.092970607490389
    K_i_init = 99.3100014897692

    E_K_init = R * temperature / F * np.log(K_e_init/K_i_init)
    dphi = np.array(phi_M) - np.array(E_K)
    A = 1 + np.exp(18.4/42.4)                       # shorthand
    B = 1 + np.exp(-(0.1186e3 + E_K_init)/0.0441e3) # shorthand
    C = 1 + np.exp((dphi + 0.0185e3)/0.0425e3)      # shorthand
    D = 1 + np.exp(-(0.1186e3 + np.array(phi_M))/0.0441e3)    # shorthand

    g_Kir = np.sqrt(np.array(tr_K_e)/K_e_init)*(A*B)/(C*D)

    # define and return current
    i_kir = g_leak_K * g_Kir * (np.array(phi_M) - E_K) # umol/(cm^2*ms)

    # Concentration plots
    fig = plt.figure(figsize=(12*0.9,12*0.9))
    ax = plt.gca()

    ax1 = fig.add_subplot(3,3,1)
    plt.title(r'Na$^+$ concentration (ECS)')
    plt.ylabel(r'[Na]$_e$ (mM)')
    plt.plot(Na_e, linewidth=3, color='b')

    ax3 = fig.add_subplot(3,3,2)
    plt.title(r'K$^+$ concentration (ECS)')
    plt.ylabel(r'[K]$_e$ (mM)')
    plt.plot(K_e, linewidth=3, color='b')

    ax3 = fig.add_subplot(3,3,3)
    plt.title(r'Cl$^-$ concentration (ECS)')
    plt.ylabel(r'[Cl]$_e$ (mM)')
    plt.plot(Cl_e, linewidth=3, color='b')

    ax2 = fig.add_subplot(3,3,4)
    plt.title(r'Na$^+$ concentration (ICS)')
    plt.ylabel(r'[Na]$_i$ (mM)')
    plt.plot(Na_i,linewidth=3, color='r')

    ax2 = fig.add_subplot(3,3,5)
    plt.title(r'K$^+$ concentration (ICS)')
    plt.ylabel(r'[K]$_i$ (mM)')
    plt.plot(K_i,linewidth=3, color='r')

    ax2 = fig.add_subplot(3,3,6)
    plt.title(r'Cl$^-$ concentration (ICS)')
    plt.ylabel(r'[Cl]$_i$ (mM)')
    plt.plot(Cl_i,linewidth=3, color='r')

    ax5 = fig.add_subplot(3,3,7)
    plt.title(r'Membrane potential')
    plt.ylabel(r'$\phi_M$ (mV)')
    plt.xlabel(r'time (ms)')
    plt.plot(phi_M, linewidth=3)

    # make pretty
    ax.axis('off')
    plt.tight_layout()

    # save figure to file
    plt.savefig(f'results/{fname_in}/summary_{fname_out}.svg', format='svg')

    f_phi_M = open(f'results/{fname_in}/phi_M_{fname_out}.txt', "w")
    for p in phi_M:
        f_phi_M.write("%.10f \n" % p)
    f_phi_M.close()

    f_K_e = open(f'results/{fname_in}/K_ECS_{fname_out}.txt', "w")
    for p in K_e:
        f_K_e.write("%.10f \n" % p)
    f_K_e.close()

    f_K_i = open(f'results/{fname_in}/K_ICS_{fname_out}.txt', "w")
    for p in K_i:
        f_K_i.write("%.10f \n" % p)
    f_K_i.close()

    f_Na_e = open(f'results/{fname_in}/Na_ECS_{fname_out}.txt', "w")
    for p in Na_e:
        f_Na_e.write("%.10f \n" % p)
    f_Na_e.close()

    f_Na_i = open(f'results/{fname_in}/Na_ICS_{fname_out}.txt', "w")
    for p in Na_i:
        f_Na_i.write("%.10f \n" % p)
    f_Na_i.close()

    f_i_pump = open(f'results/{fname_in}/i_pump.txt', "w")
    for p in i_pump:
        f_i_pump.write("%.10f \n" % p)
    f_i_pump.close()

    f_i_kir = open(f'results/{fname_in}/i_kir.txt', "w")
    for p in i_kir:
        f_i_kir.write("%.10f \n" % p)
    f_i_kir.close()

    f_tr_K_e = open(f'results/{fname_in}/tr_K_e.txt', "w")
    for p in tr_K_e:
        f_tr_K_e.write("%.10f \n" % p)
    f_tr_K_e.close()

    f_tr_K_i = open(f'results/{fname_in}/tr_K_i.txt', "w")
    for p in tr_K_i:
        f_tr_K_i.write("%.10f \n" % p)
    f_tr_K_i.close()

    f_tr_Na_e = open(f'results/{fname_in}/tr_Na_e.txt', "w")
    for p in tr_Na_e:
        f_tr_Na_e.write("%.10f \n" % p)
    f_tr_Na_e.close()

    f_tr_Na_i = open(f'results/{fname_in}/tr_Na_i.txt', "w")
    for p in tr_Na_i:
        f_tr_Na_i.write("%.10f \n" % p)
    f_tr_Na_i.close()


    f_tr_Cl_e = open(f'results/{fname_in}/tr_Cl_e.txt', "w")
    for p in tr_Cl_e:
        f_tr_Cl_e.write("%.10f \n" % p)
    f_tr_Cl_e.close()

    f_tr_Cl_i = open(f'results/{fname_in}/tr_Cl_i.txt', "w")
    for p in tr_Cl_i:
        f_tr_Cl_i.write("%.10f \n" % p)
    f_tr_Cl_i.close()

    return

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

    fname_in = config["fname"]

    fname_out_N = "neuron"
    fname_out_G = "glial"
    tag_N = 1
    tag_G = 2

    # create directory for figures
    if not os.path.isdir(f'results/{fname_in}'):
        os.mkdir(f'results/{fname_in}')

    # create figures
    dt = 0.1
    Tstop = config["Tstop"]

    # EMI points glial
    x_M = 0.00026834450833705247
    y_M = 0.0002889436164406373
    z_M = 0.00022057539244152102
    x_i = 0.0002757962756580815
    y_i = 0.00028978895336808524
    z_i = 0.00024707838038751177
    x_e = 0.00027393821446464905
    y_e = 0.0002511162579901399
    z_e = 0.0002376715140603816
    x_G = {'M':[x_M, y_M, z_M],
           'i':[x_i, y_i, z_i],
           'e':[x_e, y_e, z_e],
    }

    plot_3D_concentration(fname_in, fname_out_G, dt, Tstop, x_G, tag_G)

    """
    # EMI points neuron
    x_M = 0.00021805911552094111
    y_M = 0.00022208269041793245
    z_M = 0.00023494927229732336
    x_i = 0.00021895646088814492
    y_i = 0.00023021958580729074
    z_i = 0.00023207341100176107
    x_e = 0.00025308818813279027
    y_e = 0.00023698776419221233
    z_e = 0.00023301680991154913
    x_N = {'M':[x_M, y_M, z_M],
           'i':[x_i, y_i, z_i],
           'e':[x_e, y_e, z_e],
    }

    plot_3D_concentration(fname_in, fname_out_N, dt, Tstop, x_N, tag_N)
    """
