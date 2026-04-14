import dolfinx
import scifem
import numpy as np
from mpi4py import MPI

comm = MPI.COMM_WORLD

"""
Script for reading mesh and writing submesh to file (with same cell tags and
facet tags as original mesh). The submesh consists of mesh cells with tag 0
(ECS) and tag 1 (glial cell).
"""

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

if __name__ == "__main__":

    mesh_file = 'meshes/remarked_mesh/mesh.xdmf'
    mesh, ct, ft = read_mesh(mesh_file)

    # Re-mesh with only tag 0 and 1
    mesh_sub, sub_to_parent, sub_vertex_to_parent, _, _ = \
            scifem.extract_submesh(mesh, ct, [0, 1])

    # Transfer cell tags to submesh
    sub_ct, _ = scifem.transfer_meshtags_to_submesh(
            ct, mesh_sub, sub_vertex_to_parent, sub_to_parent
        )
    sub_ct.name="cell_marker"

    tdim = mesh_sub.topology.dim

    # Create facet entities, facet-to-cell connectivity and cell-to-cell connectivity
    mesh_sub.topology.create_entities(tdim-1)
    mesh_sub.topology.create_connectivity(tdim-1, tdim)
    mesh_sub.topology.create_connectivity(tdim, tdim)

    # Create gamma tags
    gamma_facets = scifem.find_interface(sub_ct, 1, 0)
    exterior_facets = dolfinx.mesh.exterior_facet_indices(mesh_sub.topology)

    facet_map = mesh_sub.topology.index_map(mesh_sub.topology.dim - 1)
    num_facets_local = facet_map.size_local + facet_map.num_ghosts
    facets = np.arange(num_facets_local, dtype=np.int32)

    interface_marker = 1
    boundary_marker = 1101

    marker = np.full_like(facets, 0, dtype=np.int32)
    marker[gamma_facets] = interface_marker
    marker[exterior_facets] = boundary_marker
    sub_ft = dolfinx.mesh.meshtags(
        mesh_sub, mesh_sub.topology.dim - 1, np.arange(num_facets_local, dtype=np.int32), marker
    )
    sub_ft.name = "facet_marker"

    # write new mesh to file
    filename = "meshes/ECS_astro/mesh.xdmf"
    with dolfinx.io.XDMFFile(mesh_sub.comm, filename, 'w') as file:
       file.write_mesh(mesh_sub)
       file.write_meshtags(sub_ct, mesh_sub.geometry)
       file.write_meshtags(sub_ft, mesh_sub.geometry)
    file.close()
