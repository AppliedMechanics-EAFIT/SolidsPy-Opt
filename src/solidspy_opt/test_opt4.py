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

def SIMP(
    nodes: NDArray[np.float64], 
    els: NDArray[np.int_], 
    mats: NDArray[np.float64], 
    loads: NDArray[np.float64], 
    idx_BC: NDArray[np.float64], 
    niter: int, 
    penal: float, 
    volfrac: float, 
    dimensions: List[int]=None,
    plot: bool = False,
    dim_problem: int = 2,
    nnodes: int = 4
) -> None:
    """
    Performs Structural Optimization using the Solid Isotropic Material with Penalization (SIMP) method for a 3D structure.

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
        Array of boundary conditions.
    niter : int
        Number of iterations for the SIMP process.
    penal : float
        Penalization factor used to enforce material stiffness in intermediate densities.
    volfrac : float
        Target volume fraction for
    dimensions : list
        List of dimensions of the problem. Only need it for 2D problems. [nx, ny].
    nnodes : int, optional
        Number of nodes per element. For 2D problems it can be 3 or 4 and for 3D problems 4 or 8. Default is 4.

    Returns
    -------
    ELS : ndarray
        Array of the optimized elements after the SIMP process.
    nodes : ndarray
        Array of the optimized nodes after the SIMP process.

    Notes
    -----
    - The SIMP method iteratively updates element densities to optimize the material distribution.
    - Element stiffness is penalized for intermediate densities to encourage a binary (solid or void) material distribution.
    - The optimization stops after reaching the specified number of iterations or if the target volume fraction is achieved.

    Process
    -------
    1. Initialize element densities to satisfy the target volume fraction.
    2. Assemble the global stiffness matrix and load vector.
    3. Solve the linear system to compute nodal displacements.
    4. Compute element stiffnesses and objective function (e.g., compliance).
    5. Update element densities using the Optimality Criteria method.
    6. Repeat until the specified number of iterations or target volume fraction is reached.

    Visualization
    -------------
    If `plot` is True, the function will generate:
    - A plot of the initial and optimized structures showing displacements, strains, and stresses.
    - A filled contour plot of the final optimized mesh.

    Example
    -------
    >>> optimized_els, optimized_nodes = SIMP(
    ...     nodes, els, mats, niter=50, penal=3.0, volfrac=0.5, plot=True
    ... )

    """
    assert dim_problem in [2, 3], "dim_problem must be either 2 (for 2D) or 3 (for 3D)"
    assert nnodes in [3, 4, 8], "nnodes must be either 3, 4 (for 2D) or 4, 8 (for 3D)"

    uel_func = None
    if dim_problem == 2:
        assert dimensions is not None, "For 2D problems, the dimensions parameter cannot be None"
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

    # Initialize variables
    Emin=1e-9 # Minimum young modulus of the material
    Emax=1.0 # Maximum young modulus of the material

    # Initialize the design variables
    change = 10 # Change in the design variable
    g = 0 # Constraint
    nels = els.shape[0] # Number of elements
    rho = 0.5 * np.ones(nels, dtype=float) # Initialize the density
    sensi_rho = np.ones(nels) # Initialize the sensitivity
    rho_old = rho.copy() # Initialize the density history
    d_c = np.ones(nels) # Initialize the design change

    r_min = np.linalg.norm(nodes[0,1:-dim_problem] - nodes[1,1:-dim_problem]) * 4 # Radius for the sensitivity filter
    centers = calculate_element_centers(nodes, els, dim_problem, nnodes)

    iter = 0
    for _ in range(niter):
        iter += 1

        # Check convergence
        if change < 0.01:
            print('Convergence reached')
            break

        # Change density 
        mats[:,2] = Emin+rho**penal*(Emax-Emin)

        # System assembly
        assem_op, IBC, neq = ass.DME(nodes[:, -dim_problem:], els, ndof_el_max=nnodes*dim_problem)
        # stiff_mat, _ = ass.assembler(els, mats, nodes[:, :-dim_problem], neq, assem_op, uel=uel_func)
        stiff_mat = sparse_assem(els, nodes, mats, neq, assem_op, dim_problem, uel=uel_func)
        rhs_vec = ass.loadasem(loads, IBC, neq)

        # System solution
        disp = spsolve(stiff_mat, rhs_vec)
        UC = pos.complete_disp(IBC, nodes, disp, ndof_node=dim_problem)
        E_nodes, S_nodes = pos.strain_nodes_3d(nodes, els, mats[:,:2], UC) if dim_problem==3 else pos.strain_nodes(nodes, els, mats[:,:2], UC)

        compliance = rhs_vec.T.dot(disp)

        # Sensitivity analysis
        # sensi_rho[:] = (np.dot(UC[els[:,-nnodes:]].reshape(els.shape[0],nnodes*dim_problem),kloc) * UC[els[:,-4:]].reshape(els.shape[0],nnodes*dim_problem) ).sum(1)
        sensi_rho = sensitivity_elsSIMP(nodes, mats, els, UC, uel_func, nnodes, dim_problem)
        d_c[:] = (-penal*rho**(penal-1)*(Emax-Emin))*sensi_rho
        d_c[:] = density_filter(centers, r_min, rho, d_c)

        # Optimality criteria
        rho_old[:] = rho
        rho[:], g = optimality_criteria(nels, rho, d_c, g)

        # Compute the change
        change = np.linalg.norm(rho.reshape(nels,1)-rho_old.reshape(nels,1),np.inf)
        print(change)

        # Check equilibrium
        if not np.allclose(stiff_mat.dot(disp)/stiff_mat.max(), rhs_vec/stiff_mat.max()):
            break

    if plot and dim_problem == 2:
        pos.fields_plot(elsI, nodes, UCI, E_nodes=E_nodesI, S_nodes=S_nodesI) # Plot initial mesh
        pos.fields_plot(els, nodes, UC, E_nodes=E_nodes, S_nodes=S_nodes) # Plot optimized mesh

        plt.ion() 
        fig,ax = plt.subplots()
        ax.imshow(-rho.reshape(dimensions[0], dimensions[1]), cmap='gray', interpolation='none',norm=colors.Normalize(vmin=-1,vmax=0))
        ax.set_title('Predicted')
        fig.show()
    elif plot and dim_problem == 3:
        pos.fields_plot_3d(nodes, els, loads, idx_BC, S_nodes, E_nodes, rho=rho, nnodes=8, data_type='stress', show_BC=True, show_loads=True, arrow_scale=2.0, arrow_color="blue", cmap="viridis", show_axes=True, show_bounds=True, show_edges=False)

    return rho, nodes, UC, E_nodes, S_nodes

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

rho, nodes, UC, E_nodes, S_nodes = SIMP(
    nodes=nodes, 
    els=els, 
    mats=mats, 
    loads=loads, 
    idx_BC=idx_BC, 
    niter=200, 
    penal=3, 
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

rho, nodes, UC, E_nodes, S_nodes = SIMP(
    nodes=nodes, 
    els=els, 
    mats=mats, 
    loads=loads, 
    idx_BC=idx_BC, 
    niter=200, 
    penal=3, 
    volfrac=0.5, 
    dimensions=[60, 60],
    plot=True,
    dim_problem=2, 
    nnodes=4)