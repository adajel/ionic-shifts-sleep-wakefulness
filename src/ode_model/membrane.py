import knpemidg.dlt_dof_extraction as dlt
import dolfin as df
import numpy as np

from numbalsoda import lsoda

class MembraneModel():
    '''ODE on membrane defined by tagged facet function'''
    def __init__(self, ode, facet_f, tag, V):
        '''
        Facets where facet_f[facet] == tag are governed by this ode whose
        source terms will be taken from V
        '''

        mesh = facet_f.mesh()
        assert mesh.topology().dim()-1 == facet_f.dim()
        assert isinstance(tag, int)
        # ODE will talk to PDE via a H(div)-trace function - we need to
        # know which indices of that function will be used for the communication
        assert dlt.is_dlt_scalar(V)
        self.V = V

        self.facets, indices = dlt.get_indices(V, facet_f, (tag, ))
        self.indices = indices.flatten()

        self.dof_locations = V.tabulate_dof_coordinates()[self.indices]
        # For every spatial point there is an ODE with states/parameters which need
        # to be tracked
        nodes = len(self.indices)
        self.nodes = nodes

        print(self.indices)
        print(self.dof_locations)

        print(len(self.indices))
        print(len(self.dof_locations))

        self.states = np.array([ode.init_state_values() for _ in range(nodes)])

        self.parameters = np.array([ode.init_parameter_values() for _ in range(nodes)])

        self.tag = tag
        self.ode = ode
        self.prefix = ode.__name__
        self.time = 0

        df.info(f'\t{self.prefix} Number of ODE points on the membrane {nodes}')

    # --- Setting ODE state/parameter based on a FEM function
    def set_state(self, which, u, locator=None):
        '''Set ODE based on PDE function `u`'''
        return self.__set_ODE('state', which, u, locator=locator)

    def set_parameter(self, which, u, locator=None):
        '''Set ODE based on PDE function `u`'''
        return self.__set_ODE('parameter', which, u, locator=locator)

    # --- Getting PDE state/parameter based on a FEM function
    def get_state(self, which, u, locator=None):
        '''Set PDE function `u` based on ODE'''
        return self.__get_PDE('state', which, u, locator=locator)

    def get_parameter(self, which, u, locator=None):
        '''Set ODE based on PDE function `u`'''
        return self.__get_PDE('parameter', which, u, locator=locator)

    # --- Setting ODE states/parameters to "constant" values at certain locations
    def set_state_values(self, value_dict, locator=None):
        ''' param_name -> (lambda x: value)'''
        return self.__set_ODE_values('state', value_dict, locator=locator)

    def set_parameter_values(self, value_dict, locator=None):
        ''' param_name -> (lambda x: value)'''
        return self.__set_ODE_values('parameter', value_dict, locator=locator)

    # --- Convenience
    def set_membrane_potential(self, u, locator=None):
        '''Update PDE potentials from the ODE solver'''
        return self.set_state('V', u, locator=locator)

    def get_membrane_potential(self, u, locator=None):
        '''Update PDE potentials from the ODE solver'''
        return self.get_state('V', u, locator=locator)

    @property
    def V_index(self):
        return self.ode.state_indices('V')

    # ---- ODE integration ------
    def step_lsoda(self, dt, stimulus, stimulus_locator=None):
        '''Solve the ODEs forward by dt with optional stimulus'''
        if stimulus is None: stimulus = {}

        ode_rhs_address = self.ode.rhs_numba.address

        if stimulus_locator is None:
            stimulus_locator = lambda x: True
        stimulus_mask = np.fromiter(map(stimulus_locator, self.dof_locations), dtype=bool)

        df.info(f'\t{self.prefix} Stepping {self.nodes} ODEs')

        timer = df.Timer('ODE step LSODA')
        timer.start()
        tsteps = np.array([self.time, self.time+dt])
        for row, is_stimulated in enumerate(stimulus_mask):  # Local 
            row_parameters = self.parameters[row]

            if is_stimulated:
                for key, value in stimulus.items():
                    row_parameters[self.ode.parameter_indices(key)] = value

            current_state = self.states[row]

            new_state, success = lsoda(ode_rhs_address,
                                       current_state,
                                       tsteps,
                                       data=row_parameters,
                                       rtol=1.0e-8, atol=1.0e-10)
            assert success
            self.states[row, :] = new_state[-1]
        self.time = tsteps[-1]
        dt = timer.stop()
        df.info(f'\t{self.prefix} Stepped {self.nodes} ODES in {dt}s')

        return self.states

    # --- Work horses
    def __set_ODE(self, what, which, u, locator=None):
        '''ODE setting '''
        (get_index, destination) = {
            'state': (self.ode.state_indices, self.states),
            'parameter': (self.ode.parameter_indices, self.parameters)
        }[what]
        the_index = get_index(which)

        assert self.V.ufl_element() == u.function_space().ufl_element()

        lidx = np.arange(self.nodes)
        if locator is not None:
            lidx = lidx[np.fromiter(map(locator, self.dof_locations), dtype=bool)]

        source = u.vector().get_local()
        if len(lidx) > 0:
            destination[lidx, the_index] = source[self.indices[lidx]]
        return self.states

    def __get_PDE(self, what, which, u, locator=None):
        '''Update PDE potentials from the ODE solver'''
        (get_index, source) = {
            'state': (self.ode.state_indices, self.states),
            'parameter': (self.ode.parameter_indices, self.parameters)
        }[what]
        the_index = get_index(which)

        assert self.V.ufl_element() == u.function_space().ufl_element()

        lidx = np.arange(self.nodes)
        if locator is not None:
            lidx = lidx[np.fromiter(map(locator, self.dof_locations), dtype=bool)]

        destination = u.vector().get_local()

        if len(lidx) > 0:
            destination[self.indices[lidx]] = source[lidx, the_index]
        u.vector().set_local(destination)
        df.as_backend_type(u.vector()).update_ghost_values()

        return u

    def __set_ODE_values(self, what, value_dict, locator=None):
        '''Batch setter'''
        (destination, get_col) = {
            'state': (self.states, self.ode.state_indices),
            'parameter': (self.parameters, self.ode.parameter_indices)
        }[what]

        lidx = np.arange(self.nodes)
        if locator is not None:
            lidx = lidx[np.fromiter(map(locator, self.dof_locations), dtype=bool)]
        df.info(f'\t{self.prefix} Set {what} for {len(lidx)} ODES')

        if len(lidx) == 0: return destination

        coords = self.dof_locations[lidx]
        for param in value_dict:
            col = get_col(param)
            get_value = value_dict[param]
            for row, x in zip(lidx, coords):  # Counts the odes
                destination[row, col] = get_value(x)
        return destination

# --------------------------------------------------------------------

def make_global(f, mesh, subdomains):

    mesh = subdomains.mesh()
    subdomains = subdomains

    # DG space for projecting coefficients
    Q = df.FunctionSpace(mesh, "DG", 0)
    q = df.Function(Q, name="diff")
    num_cells_local = len(q.vector().get_local())

    for key, value in f.items():
        cell_indices = np.array(subdomains.where_equal(key))
        q.vector()[cell_indices[cell_indices<num_cells_local]] = float(value)
    q.vector().apply("insert")

    return q

if __name__ == '__main__':
    import os
    from knpemidg.utils import pcws_constant_project
    from knpemidg.utils import interface_normal, plus, minus

    #import mm_hh_id as ode
    import mm_hh as ode

    resolution = 2

    # Get mesh, subdomains, surfaces paths
    here = os.path.abspath(os.path.dirname(__file__))
    mesh_prefix = os.path.join(here, 'meshes/2D/')
    mesh_path = mesh_prefix + 'mesh_' + str(resolution) + '.xml'
    subdomains_path = mesh_prefix + 'subdomains_' + str(resolution) + '.xml'
    surfaces_path = mesh_prefix + 'surfaces_' + str(resolution) + '.xml'

    # generate mesh if it does not exist
    if not os.path.isfile(mesh_path):
        from make_mesh_2D import main
        main(["-r", str(resolution), "-d", mesh_prefix])

    mesh = df.Mesh(mesh_path)
    subdomains = df.MeshFunction('size_t', mesh, subdomains_path)
    surfaces = df.MeshFunction('size_t', mesh, surfaces_path)

    Q = df.FunctionSpace(mesh, 'Discontinuous Lagrange Trace', 0)
    V = df.FunctionSpace(mesh, 'DG', 1)

    x, y = Q.tabulate_dof_coordinates().T
    tag = 1

    membrane = MembraneModel(ode, facet_f=surfaces, tag=tag, V=Q)
    #membrane.set_membrane_potential(u)

    # set stimulus ODE
    g_syn_bar = 10
    stimulus = {'stim_amplitude': g_syn_bar}

    # Physical parameters
    C_M = 0.02                       # capacitance
    temperature = 300                # temperature (K)
    F = 96485                        # Faraday's constant (C/mol)
    R = 8.314                        # Gas Constant (J/(K*mol))
    psi = F / (R * temperature)      # shorthand
    dt = 1.0e-4/25
    Tstop = 1.0e-2

    C_phi = C_M / dt                 # shorthand

    # Initial values
    Na_i_init = 12.838513108648856   # Intracellular Na concentration
    Na_e_init = 100.71925900027354   # extracellular Na concentration
    K_i_init  = 124.15397583491901   # intracellular K concentration
    K_e_init = 3.3236967382705265    # extracellular K concentration

    """
    c_Na = make_global({0:Na_e_init, 1:Na_i_init}, mesh, subdomains)
    c_K = make_global({0:K_e_init, 1:K_i_init}, mesh, subdomains)

    n_g = interface_normal(subdomains, mesh)

    # set extracellular trace of K concentration at membrane
    #K_e = plus(c_K, n_g)
    #membrane.set_parameter('K_e', pcws_constant_project(K_e, Q))

    # set intracellular trace of Na concentration at membrane
    #Na_i = minus(c_Na, n_g)
    #membrane.set_parameter('Na_i', pcws_constant_project(Na_i, Q))

    #membrane.set_parameter_values({'Cm': lambda x: C_M})

    # calculate and set Nernst potential for current ion (+ is ECS, - is ICS)
    E_K = R * temperature / (F) * df.ln(plus(c_K, n_g) / minus(c_K, n_g))
    E_Na = R * temperature / (F) * df.ln(plus(c_Na, n_g) / minus(c_Na, n_g))
    membrane.set_parameter('E_Na', pcws_constant_project(E_Na, Q))
    membrane.set_parameter('E_K', pcws_constant_project(E_K, Q))

    print(np.unique(pcws_constant_project(E_Na, Q).vector().get_local()))
    print(np.unique(pcws_constant_project(E_K, Q).vector().get_local()))
    """

    #import sys
    #sys.exit(0)

    V_index = ode.state_indices('V')
    potential_history = []

    for _ in range(int(Tstop/dt)):

        membrane.step_lsoda(dt=dt, stimulus=stimulus)
        potential_history.append(1*membrane.states[:, V_index])
        #membrane.get_membrane_potential(u)

    potential_history = np.array(potential_history)

    import matplotlib.pyplot as plt

    plt.figure()
    plt.plot(potential_history[:, 2])
    plt.savefig("plot.png")

    # TODO:
    # - consider a test where we have dy/dt = A(x)y with y(t=0) = y0
    # - after stepping u should be fine
    # - add forcing:  dy/dt = A(x)y + f(t) with y(t=0) = y0
    # - things are currently quite slow -> multiprocessing?
    # - rely on cbc.beat?
