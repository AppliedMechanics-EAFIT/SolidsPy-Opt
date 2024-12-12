# %%
%matplotlib widget
import matplotlib.pyplot as plt 
from matplotlib import colors
import numpy as np 
from numpy.typing import NDArray
from typing import List, Tuple, AnyStr
from scipy.sparse.linalg import spsolve

from Utils.beams import * 
from Utils.solver import * 
from Utils.volumes import * 

import solidspy.assemutil as ass 
import solidspy.postprocesor as pos 
import solidspy.uelutil as uel 
np.seterr(divide='ignore', invalid='ignore')

# %% ESO stress based

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
        Array of elements with the format [element number, X load magnitud, Y load magnitud, Z load magnitud].
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
        mask_els = protect_elsESO(els, loads, idx_BC) # Mask of elements to do not remove
        mask_del *= mask_els  
        els = np.delete(els, mask_del, 0) # Delete elements
        del_nodeESO(nodes, els, nnodes, dim_problem) # Remove nodes

        RR += ER

    if plot:
        pos.fields_plot(elsI, nodes, UCI, E_nodes=E_nodesI, S_nodes=S_nodesI) # Plot initial mesh
        pos.fields_plot(ELS, nodes, UC, E_nodes=E_nodes, S_nodes=S_nodes) # Plot optimized mesh

        fill_plot = np.ones(E_nodes.shape[0])
        plt.figure()
        tri = pos.mesh2tri(nodes, ELS)
        plt.tricontourf(tri, fill_plot, cmap='binary')
        plt.axis("image");

    return ELS, nodes, UC, E_nodes, S_nodes

# %% Eso stiff based

def ESO_stiff(
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
    Performs Evolutionary Structural Optimization (ESO) based on stiff for a beam structure.

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
        Array of elements with the format [element number, X load magnitud, Y load magnitud, Z load magnitud].
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
    >>> optimized_els, optimized_nodes = ESO_stiff(
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

    elsI= np.copy(els)

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
        
        # System assembly
        assem_op, IBC, neq = ass.DME(nodes[:, -dim_problem:], els, ndof_el_max=nnodes*dim_problem)
        stiff_mat, _ = ass.assembler(els, mats, nodes[:, :-dim_problem], neq, assem_op, uel=uel_func)
        rhs_vec = ass.loadasem(loads, IBC, neq)

        # System solution
        disp = spsolve(stiff_mat, rhs_vec)
        UC = pos.complete_disp(IBC, nodes, disp, ndof_node=dim_problem)
        E_nodes, S_nodes = pos.strain_nodes_3d(nodes, els, mats[:,:2], UC) if dim_problem==3 else pos.strain_nodes(nodes, els, mats[:,:2], UC)

        # Compute Sensitivity number
        sensi_number = sensitivity_elsESO(nodes, mats, els, UC) # Sensitivity number
        mask_del = sensi_number < RR # Mask of elements to be removed
        mask_els = protect_elsESO(els, loads, idx_BC) # Mask of elements to do not remove
        mask_del *= mask_els # Mask of elements to be removed and not protected
        ELS = els # Save last iteration elements
        
        # Remove/add elements
        els = np.delete(els, mask_del, 0) # Remove elements
        del_nodeESO(nodes, els) # Remove nodes

        RR += ER

    if plot:
        pos.fields_plot(elsI, nodes, UCI, E_nodes=E_nodesI, S_nodes=S_nodesI) # Plot initial mesh
        pos.fields_plot(ELS, nodes, UC, E_nodes=E_nodes, S_nodes=S_nodes) # Plot optimized mesh

        fill_plot = np.ones(E_nodes.shape[0])
        plt.figure()
        tri = pos.mesh2tri(nodes, ELS)
        plt.tricontourf(tri, fill_plot, cmap='binary')
        plt.axis("image");

    return ELS, nodes, UC, E_nodes, S_nodes


# %% BESO

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
        Array of elements with the format [element number, X load magnitud, Y load magnitud, Z load magnitud].
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

    r_min = np.linalg.norm(nodes[0,1:3] - nodes[1,1:3]) * 1 # Radius for the sensitivity filter
    adj_nodes = adjacency_nodes(nodes, els) # Adjacency nodes
    centers = center_els(nodes, els) # Centers of elements

    Vi = calculate_mesh_volume(nodes, els) if dim_problem==3 else calculate_mesh_area(nodes, els)
    V_opt = Vi * volfrac

    # Initialize variables.
    ELS = None
    mask = np.ones(els.shape[0], dtype=bool) # Mask of elements to be removed
    sensi_I = None  
    C_h = np.zeros(niter) # History of compliance
    error = 1000 

    for i in range(niter):
        print("Number of elements: {}".format(els.shape[0]))

        # Calculate the optimal design array elements
        els_del = els[mask].copy() # Elements to be removed
        V = calculate_mesh_area(nodes, els_del) # Volume of the structure

        # Check equilibrium
        Vi = calculate_mesh_volume(nodes, els) if dim_problem==3 else calculate_mesh_area(nodes, els)
        if not np.allclose(stiff_mat.dot(disp)/stiff_mat.max(), rhs_vec/stiff_mat.max()) or Vi < V_opt: 
            break

        # Storage the solution
        ELS = els_del 

        # System assembly
        assem_op, IBC, neq = ass.DME(nodes[:, -dim_problem:], els, ndof_el_max=nnodes*dim_problem)
        stiff_mat, _ = ass.assembler(els, mats, nodes[:, :-dim_problem], neq, assem_op, uel=uel_func)
        rhs_vec = ass.loadasem(loads, IBC, neq)

        # System solution
        disp = spsolve(stiff_mat, rhs_vec)
        UC = pos.complete_disp(IBC, nodes, disp, ndof_node=dim_problem)
        E_nodes, S_nodes = pos.strain_nodes_3d(nodes, els, mats[:,:2], UC) if dim_problem==3 else pos.strain_nodes(nodes, els, mats[:,:2], UC)

        # Sensitivity filter
        sensi_e = sensitivity_elsBESO(nodes, mats, els, mask, UC) # Calculate the sensitivity of the elements
        sensi_nodes = sensitivity_nodes(nodes, adj_nodes, centers, sensi_e) # Calculate the sensitivity of the nodes
        sensi_number = sensitivity_filter(nodes, centers, sensi_nodes, r_min) # Perform the sensitivity filter

        # Average the sensitivity numbers to the historical information 
        if i > 0: 
            sensi_number = (sensi_number + sensi_I)/2 # Average the sensitivity numbers to the historical information
        sensi_number = sensi_number/sensi_number.max() # Normalize the sensitivity numbers

        # Check if the optimal volume is reached and calculate the next volume
        V_r = False
        if V <= V_opt:
            els_k = els_del.shape[0]
            V_r = True
            break
        else:
            V_k = V * (1 + ER) if V < V_opt else V * (1 - ER)

        # Remove/add threshold
        sensi_sort = np.sort(sensi_number)[::-1] # Sort the sensitivity numbers
        els_k = els_del.shape[0]*V_k/V # Number of elements to be removed
        alpha_del = sensi_sort[int(els_k)] # Threshold for removing elements

        # Remove/add elements
        mask = sensi_number > alpha_del # Mask of elements to be removed
        mask_els = protect_els(els[np.invert(mask)], els.shape[0], loads, idx_BC) # Mask of elements to be protected
        mask = np.bitwise_or(mask, mask_els) 
        del_node(nodes, els[mask], loads, idx_BC) # Delete nodes

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

    if plot:
        pos.fields_plot(elsI, nodes, UCI, E_nodes=E_nodesI, S_nodes=S_nodesI) # Plot initial mesh
        pos.fields_plot(ELS, nodes, UC, E_nodes=E_nodes, S_nodes=S_nodes) # Plot optimized mesh

        fill_plot = np.ones(E_nodes.shape[0])
        plt.figure()
        tri = pos.mesh2tri(nodes, ELS)
        plt.tricontourf(tri, fill_plot, cmap='binary')
        plt.axis("image");

    return ELS, nodes, UC, E_nodes, S_nodes

# %% SIMP

def SIMP(
    nodes: NDArray[np.float64], 
    els: NDArray[np.int_], 
    mats: NDArray[np.float64], 
    loads: NDArray[np.float64], 
    niter: int, 
    penal: float, 
    volfrac: float, 
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
        Array of elements with the format [element number, X load magnitud, Y load magnitud, Z load magnitud].
    niter : int
        Number of iterations for the SIMP process.
    penal : float
        Penalization factor used to enforce material stiffness in intermediate densities.
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
        if nnodes == 3:
            uel_func = uel.elast_tri3
        else:
            uel_func = uel.elast_quad4
    elif dim_problem == 3:
        if nnodes == 4:
            uel_func = uel.elast_tet4
        else:
            uel_func = uel.elast_hex8

    # Initialize variables
    Emin=1e-9 # Minimum young modulus of the material
    Emax=1.0 # Maximum young modulus of the material

    # Initialize the design variables
    change = 10 # Change in the design variable
    g = 0 # Constraint
    rho = 0.5 * np.ones(ny*nx, dtype=float) # Initialize the density
    sensi_rho = np.ones(ny*nx) # Initialize the sensitivity
    rho_old = rho.copy() # Initialize the density history
    d_c = np.ones(ny*nx) # Initialize the design change

    r_min = np.linalg.norm(nodes[0,1:3] - nodes[1,1:3]) * 4 # Radius for the sensitivity filter
    centers = center_els(nodes, els) # Calculate centers

    E = mats[0,0] # Young modulus
    nu = mats[0,1] # Poisson ratio
    k = np.array([1/2-nu/6,1/8+nu/8,-1/4-nu/12,-1/8+3*nu/8,-1/4+nu/12,-1/8-nu/8,nu/6,1/8-3*nu/8]) # Coefficients
    kloc = E/(1-nu**2)*np.array([ [k[0], k[1], k[2], k[3], k[4], k[5], k[6], k[7]], 
    [k[1], k[0], k[7], k[6], k[5], k[4], k[3], k[2]],
    [k[2], k[7], k[0], k[5], k[6], k[3], k[4], k[1]],
    [k[3], k[6], k[5], k[0], k[7], k[2], k[1], k[4]],
    [k[4], k[5], k[6], k[7], k[0], k[1], k[2], k[3]],
    [k[5], k[4], k[3], k[2], k[1], k[0], k[7], k[6]],
    [k[6], k[3], k[4], k[1], k[2], k[7], k[0], k[5]],
    [k[7], k[2], k[1], k[4], k[3], k[6], k[5], k[0]]]); # Local stiffness matrix
    assem_op, bc_array, neq = ass.DME(nodes[:, -2:], els, ndof_el_max=8) 

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
        stiff_mat = sparse_assem(els, mats, neq, assem_op, kloc)
        rhs_vec = ass.loadasem(loads, bc_array, neq)

        # System solution
        disp = spsolve(stiff_mat, rhs_vec)
        UC = pos.complete_disp(bc_array, nodes, disp)

        compliance = rhs_vec.T.dot(disp)

        # Sensitivity analysis
        sensi_rho[:] = (np.dot(UC[els[:,-4:]].reshape(nx*ny,8),kloc) * UC[els[:,-4:]].reshape(nx*ny,8) ).sum(1)
        d_c[:] = (-penal*rho**(penal-1)*(Emax-Emin))*sensi_rho
        d_c[:] = density_filter(centers, r_min, rho, d_c)

        # Optimality criteria
        rho_old[:] = rho
        rho[:], g = optimality_criteria(nx, ny, rho, d_c, g)

        # Compute the change
        change = np.linalg.norm(rho.reshape(nx*ny,1)-rho_old.reshape(nx*ny,1),np.inf)

        # Check equilibrium
        if not np.allclose(stiff_mat.dot(disp)/stiff_mat.max(), rhs_vec/stiff_mat.max()):
            break

    if plot:
        plt.ion() 
        fig,ax = plt.subplots()
        ax.imshow(-rho.reshape(nx,ny), cmap='gray', interpolation='none',norm=colors.Normalize(vmin=-1,vmax=0))
        ax.set_title('Predicted')
        fig.show()

    return rho