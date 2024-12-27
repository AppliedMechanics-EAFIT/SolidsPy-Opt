# %%
%matplotlib widget
import matplotlib.pyplot as plt 
from matplotlib import colors
import numpy as np 
from numpy.typing import NDArray
from typing import List, Tuple, AnyStr
from scipy.sparse.linalg import spsolve

from utils.structures import * 
from utils.solver import * 
from utils.volumes import * 

import solidspy.assemutil as ass 
import solidspy.postprocesor as pos 
import solidspy.uelutil as uel 
np.seterr(divide='ignore', invalid='ignore')

def ESO_stress(
    nodes: NDArray[np.float64], 
    els: NDArray[np.int_], 
    mats: NDArray[np.float64], 
    loads: NDArray[np.float64], 
    idx_BC: NDArray[np.float64], 
    niter: int, 
    RR: float, 
    ER: float, 
    volfrac: float, 
    plot: bool = False,
    dim_problem: int = 2,
    nnodes: int = 4
) -> None:
    """
    Performs Evolutionary Structural Optimization (ESO) based on stress for a 3D structure.

    Parameters
    ----------
    nodes : ndarray
        Array of node coordinates and boundary conditions with the format 
        [node number, x, y, z, BC], where BC indicates boundary conditions.
    els : ndarray
        Array of elements with the format [element number, material, node 1, node 2, ...].
    mats : ndarray
        Array of material properties with the format [E, nu, rho], where:
        - E is the Young's modulus,
        - nu is the Poisson's ratio,
        - rho is the density.
    loads : ndarray
        Array of elements with the format [node number, X load magnitud, Y load magnitud, Z load magnitud].
    idx_BC : ndarray
        Array of node indices with boundary conditions applied.
    niter : int
        Number of iterations for the ESO process.
    RR : float
        Initial relative stress threshold for removing elements.
    ER : float
        Increment of the relative stress threshold (RR) for each iteration.
    volfrac : float
        Target volume fraction for the optimized structure, expressed as a fraction of the initial volume.
    plot : bool, optional
        If True, plot the initial and optimized meshes. Defaults to False.
    dim_problem : int, optional
        Dimension of the problem (2 for 2D, 3 for 3D). Default is 2.
    nnodes : int, optional
        Number of nodes per element. For 2D problems it can be 3 or 4 and for 3D problems 4 or 8. Default is 4.

    Returns
    -------
    ELS : ndarray
        Array of the optimized elements after the ESO process.
    nodes : ndarray
        Array of the optimized nodes after the ESO process.
    UC : ndarray
      Array with the displacements.
    E_nodes : ndarray
        Strains evaluated at the nodes.
    S_nodes : ndarray
        Stresses evaluated at the nodes.

    Notes
    -----
    - This function performs structural optimization by iteratively removing elements with low relative stress.
    - The relative stress threshold (RR) increases in each iteration to progressively refine the structure.
    - The optimization stops either after reaching the specified number of iterations or if the target volume fraction is achieved.

    Process
    -------
    1. Assemble the global stiffness matrix and load vector.
    2. Solve the linear system to compute nodal displacements.
    3. Calculate element stresses and strains.
    4. Compute the von Mises stress for each element and identify elements to remove.
    5. Update the structure by removing selected elements and their associated nodes.
    6. Repeat until the specified number of iterations or target volume fraction is reached.

    Visualization
    -------------
    If `plot` is True, the function will generate:
    - A plot of the initial and optimized structures showing displacements, strains, and stresses.
    - A filled contour plot of the final optimized mesh.

    Example
    -------
    >>> optimized_els, optimized_nodes = ESO_stress(
    ...     nodes, els, mats, idx_BC, niter=10, RR=0.8, ER=0.05, volfrac=0.5, plot=True
    ... )
    """
    assert dim_problem in [2, 3], "dim_problem must be either 2 (for 2D) or 3 (for 3D)"
    assert nnodes in [3, 4, 8], "nnodes must be either 3, 4 (for 2D) or 4, 8 (for 3D)"

    uel_func = None
    if dim_problem == 2:
        if nnodes == 3:
            uel_func = uel.elast_tri3
        else:
            uel_func = uel.elast_quad4
    elif dim_problem == 3:
        if nnodes == 4:
            uel_func = uel.elast_tet4
        else:
            uel_func = uel.elast_hex8

    elsI = np.copy(els)

    # System assembly
    assem_op, IBC, neq = ass.DME(nodes[:, -dim_problem:], els, ndof_el_max=nnodes*dim_problem)
    stiff_mat, _ = ass.assembler(els, mats, nodes[:, :-dim_problem], neq, assem_op, uel=uel_func)
    rhs_vec = ass.loadasem(loads, IBC, neq)

    # System solution
    disp = spsolve(stiff_mat, rhs_vec)
    UCI = pos.complete_disp(IBC, nodes, disp, ndof_node=dim_problem)
    E_nodesI, S_nodesI = pos.strain_nodes_3d(nodes, els, mats[:,:2], UCI) if dim_problem==3 else pos.strain_nodes(nodes, els, mats[:,:2], UCI)

    Vi = calculate_mesh_volume(nodes, els) if dim_problem==3 else calculate_mesh_area(nodes, els)
    V_opt = Vi * volfrac

    ELS = None
    for _ in range(niter):
        print("Number of elements: {}".format(els.shape[0]))

        # Check equilibrium
        Vi = calculate_mesh_volume(nodes, els) if dim_problem==3 else calculate_mesh_area(nodes, els)
        if not np.allclose(stiff_mat.dot(disp)/stiff_mat.max(), rhs_vec/stiff_mat.max()) or Vi < V_opt: 
            break
        
        # Storage the solution
        ELS = els
        
        # System assembly
        assem_op, IBC, neq = ass.DME(nodes[:, -dim_problem:], els, ndof_el_max=nnodes*dim_problem)
        stiff_mat, _ = ass.assembler(els, mats, nodes[:, :-dim_problem], neq, assem_op, uel=uel_func)
        rhs_vec = ass.loadasem(loads, IBC, neq)

        # System solution
        disp = spsolve(stiff_mat, rhs_vec)
        UC = pos.complete_disp(IBC, nodes, disp, ndof_node=dim_problem)
        E_nodes, S_nodes = pos.strain_nodes_3d(nodes, els, mats[:,:2], UC) if dim_problem==3 else pos.strain_nodes(nodes, els, mats[:,:2], UC)
        E_els, S_els = strain_els(els, E_nodes, S_nodes) # Calculate strains and stresses in elements

        vons = np.sqrt(S_els[:,0]**2 - (S_els[:,0]*S_els[:,1]) + S_els[:,1]**2 + 3*S_els[:,2]**2)

        # Remove/add elements
        RR_el = vons/vons.max() # Relative stress
        mask_del = RR_el < RR # Mask for elements to be deleted
        mask_els = protect_elsESO(els, loads, idx_BC, nnodes) # Mask of elements to do not remove
        mask_del *= mask_els  
        els = np.delete(els, mask_del, 0) # Delete elements
        del_nodeESO(nodes, els, nnodes, dim_problem) # Remove nodes

        RR += ER

    if plot and dim_problem == 2:
        pos.fields_plot(elsI, nodes, UCI, E_nodes=E_nodesI, S_nodes=S_nodesI) # Plot initial mesh
        pos.fields_plot(ELS, nodes, UC, E_nodes=E_nodes, S_nodes=S_nodes) # Plot optimized mesh

        fill_plot = np.ones(E_nodes.shape[0])
        plt.figure()
        tri = pos.mesh2tri(nodes, ELS)
        plt.tricontourf(tri, fill_plot, cmap='binary')
        plt.axis("image");
    elif plot and dim_problem == 3:
        pos.fields_plot_3d(nodes, ELS, loads, idx_BC, S_nodes, E_nodes, nnodes=8, data_type='stress', show_BC=True, show_loads=True, arrow_scale=2.0, arrow_color="blue", cmap="viridis", show_axes=True, show_bounds=True, show_edges=False)

    return ELS, nodes, UC, E_nodes, S_nodes

# %%
# load_directions = np.array([[0, 1, 0]])
# load_positions = np.array([[0, 0, 1]])
load_directions = np.array([[0, 1, 0], [1, 0, 0], [0, 0, -1]])
load_positions = np.array([[5, 5, 9], [1, 1, 9], [8, 8, 9]])

# Call the function
nodes, mats, els, loads, idx_BC = structure_3d(
    L=10, 
    H=10, 
    W=10, 
    E=206.8e9, 
    v=0.28, 
    nx=10, 
    ny=10, 
    nz=10, 
    dirs=load_directions, 
    positions=load_positions
)

els, nodes, UC, E_nodes, S_nodes = ESO_stress(
    nodes=nodes, 
    els=els, 
    mats=mats, 
    loads=loads, 
    idx_BC=idx_BC, 
    niter=200, 
    RR=0.005, 
    ER=0.05, 
    volfrac=0.5, 
    plot=True,
    dim_problem=3, 
    nnodes=8)

# %%
nodes, mats, els, loads, idx_BC = structures(
    L=60, 
    H=60, 
    nx=60, 
    ny=60, 
    dirs=np.array([[0, -1]]), 
    positions=np.array([[15, 1]]), 
    n=1)

els, nodes, UC, E_nodes, S_nodes = ESO_stress(
    nodes=nodes, 
    els=els, 
    mats=mats, 
    loads=loads, 
    idx_BC=idx_BC, 
    niter=200, 
    RR=0.001, 
    ER=0.01, 
    volfrac=0.5, 
    plot=True,
    dim_problem=2, 
    nnodes=4)