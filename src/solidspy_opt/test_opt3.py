# %%
%matplotlib widget
import matplotlib.pyplot as plt 
from matplotlib import colors
import numpy as np 
from numpy.typing import NDArray
from typing import List, Tuple, AnyStr
from scipy.sparse.linalg import spsolve

from utils.beams import * 
from utils.solver import * 
from utils.volumes import * 

import solidspy.assemutil as ass 
import solidspy.postprocesor as pos 
import solidspy.uelutil as uel 
np.seterr(divide='ignore', invalid='ignore')

def BESO(
    nodes: NDArray[np.float64], 
    els: NDArray[np.int_], 
    mats: NDArray[np.float64], 
    loads: NDArray[np.float64], 
    idx_BC: NDArray[np.float64], 
    niter: int, 
    t: float, 
    ER: float, 
    volfrac: float, 
    plot: bool = False,
    dim_problem: int = 2,
    nnodes: int = 4
) -> None:
    """
    Performs Bi-directional Evolutionary Structural Optimization (BESO) for a 3D structure.

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
        Number of iterations for the BESO process.
    t : float
        Target stress ratio for adding or removing elements.
    ER : float
        Increment of the stress ratio (t) for each iteration.
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
        Array of the optimized elements after the BESO process.
    nodes : ndarray
        Array of the optimized nodes after the BESO process.
    UC : ndarray
      Array with the displacements.
    E_nodes : ndarray
        Strains evaluated at the nodes.
    S_nodes : ndarray
        Stresses evaluated at the nodes.

    Notes
    -----
    - This function performs structural optimization by iteratively adding or removing elements based on stress ratios.
    - The stress ratio (t) adjusts in each iteration to refine the structure, achieving a balance between material addition and removal.
    - The optimization stops either after reaching the specified number of iterations or if the target volume fraction is achieved.

    Process
    -------
    1. Assemble the global stiffness matrix and load vector.
    2. Solve the linear system to compute nodal displacements.
    3. Calculate element stresses and strains.
    4. Compute the von Mises stress for each element and identify elements to add or remove.
    5. Update the structure by modifying the elements and their associated nodes.
    6. Repeat until the specified number of iterations or target volume fraction is reached.

    Visualization
    -------------
    If `plot` is True, the function will generate:
    - A plot of the initial and optimized structures showing displacements, strains, and stresses.
    - A filled contour plot of the final optimized mesh.

    Example
    -------
    >>> optimized_els, optimized_nodes = BESO(
    ...     nodes, els, mats, idx_BC, niter=10, t=0.8, ER=0.05, volfrac=0.5, plot=True
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

    r_min = np.linalg.norm(nodes[0,1:-dim_problem] - nodes[1,1:-dim_problem]) * 1 # Radius for the sensitivity filter
    adj_nodes = adjacency_nodes(nodes, els, nnodes) # Adjacency nodes
    # centers = center_els(nodes, els) # Centers of elements
    centers = calculate_element_centers(nodes, els, dim_problem, nnodes)

    V = calculate_mesh_volume(nodes, els) if dim_problem==3 else calculate_mesh_area(nodes, els)
    V_opt = V * volfrac

    # Initialize variables.
    ELS = None
    mask = np.ones(els.shape[0], dtype=bool) # Mask of elements to be removed
    sensi_I = None  
    C_h = np.zeros(niter) # History of compliance
    error = 1000 

    for i in range(niter):
        # Calculate the optimal design array elements
        els_del = els[mask].copy() # Elements to be removed

        print("Number of elements: {}".format(els_del.shape[0]))

        # Check equilibrium
        Vi = calculate_mesh_volume(nodes, els_del) if dim_problem==3 else calculate_mesh_area(nodes, els_del)
        if not np.allclose(stiff_mat.dot(disp)/stiff_mat.max(), rhs_vec/stiff_mat.max()) or Vi < V_opt: 
            break

        # Storage the solution
        ELS = els_del 

        # System assembly
        assem_op, IBC, neq = ass.DME(nodes[:, -dim_problem:], els_del, ndof_el_max=nnodes*dim_problem)
        stiff_mat, _ = ass.assembler(els_del, mats, nodes[:, :-dim_problem], neq, assem_op, uel=uel_func)
        rhs_vec = ass.loadasem(loads, IBC, neq)

        # System solution
        disp = spsolve(stiff_mat, rhs_vec)
        UC = pos.complete_disp(IBC, nodes, disp, ndof_node=dim_problem)
        E_nodes, S_nodes = pos.strain_nodes_3d(nodes, els_del, mats[:,:2], UC) if dim_problem==3 else pos.strain_nodes(nodes, els, mats[:,:2], UC)

        # Sensitivity filter
        sensi_e = sensitivity_elsBESO(nodes, mats, els, mask, UC, uel_func, nnodes, dim_problem) # Calculate the sensitivity of the elements
        sensi_nodes = sensitivity_nodes(nodes, adj_nodes, centers, sensi_e, dim_problem) # Calculate the sensitivity of the nodes
        sensi_number = sensitivity_filter(nodes, centers, sensi_nodes, r_min, dim_problem) # Perform the sensitivity filter

        # Average the sensitivity numbers to the historical information 
        if i > 0: 
            sensi_number = (sensi_number + sensi_I)/2 # Average the sensitivity numbers to the historical information
        sensi_number = sensi_number/sensi_number.max() # Normalize the sensitivity numbers

        # Check if the optimal volume is reached and calculate the next volume
        V_r = False
        if Vi <= V_opt:
            els_k = els_del.shape[0]
            V_r = True
            break
        else:
            V_k = Vi * (1 + ER) if Vi < V_opt else Vi * (1 - ER)

        # Remove/add threshold
        sensi_sort = np.sort(sensi_number)[::-1] # Sort the sensitivity numbers
        els_k = els_del.shape[0]*V_k/Vi # Number of elements to be removed
        alpha_del = sensi_sort[int(els_k)] # Threshold for removing elements

        # Remove/add elements
        mask = sensi_number > alpha_del # Mask of elements to be removed
        mask_els = protect_els(els[np.invert(mask)], els.shape[0], loads, idx_BC, nnodes) # Mask of elements to be protected
        mask = np.bitwise_or(mask, mask_els) 
        del_node(nodes, els[mask], loads, idx_BC, dim_problem, nnodes) # Delete nodes

        # Calculate the strain energy and storage it 
        C = 0.5*rhs_vec.T@disp
        C_h[i] = C
        if i > 10: error = C_h[i-5:].sum() - C_h[i-10:-5].sum()/C_h[i-5:].sum()

        # Check for convergence
        if error <= t and V_r == True:
            print("convergence")
            break

        # Save the sensitvity number for the next iteration
        sensi_I = sensi_number.copy()

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
nodes, mats, els, loads, idx_BC = beam_3d(
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

els, nodes, UC, E_nodes, S_nodes = BESO(
    nodes=nodes, 
    els=els, 
    mats=mats, 
    loads=loads, 
    idx_BC=idx_BC, 
    niter=200, 
    t=0.005, 
    ER=0.005, 
    volfrac=0.5, 
    plot=True,
    dim_problem=3, 
    nnodes=8)

# %%
nodes, mats, els, loads, idx_BC = beam(
    L=60, 
    H=60, 
    nx=60, 
    ny=60, 
    dirs=np.array([[0, -1]]), 
    positions=np.array([[15, 1]]), 
    n=1)

els, nodes, UC, E_nodes, S_nodes = BESO(
    nodes=nodes, 
    els=els, 
    mats=mats, 
    loads=loads, 
    idx_BC=idx_BC, 
    niter=200, 
    t=0.0001, 
    ER=0.005, 
    volfrac=0.5, 
    plot=False,
    dim_problem=2, 
    nnodes=4)