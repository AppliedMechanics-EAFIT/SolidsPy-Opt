# %%
import matplotlib.pyplot as plt 
from matplotlib import colors
import numpy as np 
from scipy.sparse.linalg import spsolve

from solidspy_opt.Utils.beams import * 
from solidspy_opt.Utils.solver import * 

import solidspy.assemutil as ass 
import solidspy.postprocesor as pos 
np.seterr(divide='ignore', invalid='ignore')

# %% ESO stress based

def ESO_stress(length, height, nx, ny, dirs, positions, niter, RR, ER, volfrac, plot=False):
    """
    Performs Evolutionary Structural Optimization (ESO) based on stress for a beam structure.

    Parameters
    ----------
    length : float
        The length of the beam.
    height : float
        The height of the beam.
    nx : int
        The number of elements in the x direction.
    ny : int
        The number of elements in the y direction.
    dirs : list
        List of directions.
    positions : list
        List of positions.
    niter : int
        The number of iterations for the ESO process.
    RR : float
        The relative stress threshold for removing elements.
    ER : float
        The increment of RR for each iteration.
    volfrac : float
        The volume fraction for the optimal structure.
    plot : bool, optional
        If True, plot the initial and optimized mesh. Defaults to False.

    Returns
    -------
    ELS: ndarray
        The optimized elements of the structure.
    nodes: ndarray
        The optimized nodes of the structure.
    """
    nodes, mats, els, loads, BC = beam(L=length, H=height, nx=nx, ny=ny, dirs=dirs, positions=positions, n=1)
    elsI = np.copy(els)

    # System assembly
    assem_op, IBC, neq = ass.DME(nodes[:, -2:], els, ndof_el_max=8)
    stiff_mat, _ = ass.assembler(els, mats, nodes[:, :3], neq, assem_op)
    rhs_vec = ass.loadasem(loads, IBC, neq)

    # System solution
    disp = spsolve(stiff_mat, rhs_vec)
    UCI = pos.complete_disp(IBC, nodes, disp)
    E_nodesI, S_nodesI = pos.strain_nodes(nodes, els, mats[:,:2], UCI)

    V_opt = volume(els, length, height, nx, ny).sum() * volfrac # Optimal volume

    ELS = None
    for _ in range(niter):
        print("Number of elements: {}".format(els.shape[0]))

        # Check equilibrium
        if not np.allclose(stiff_mat.dot(disp)/stiff_mat.max(), rhs_vec/stiff_mat.max()) or volume(els, length, height, nx, ny).sum() < V_opt: 
            print('hollaa')
            break

        ELS = els
        
        # System assembly
        assem_op, IBC, neq = ass.DME(nodes[:, -2:], els, ndof_el_max=8)
        stiff_mat, _ = ass.assembler(els, mats, nodes[:, :3], neq, assem_op)
        rhs_vec = ass.loadasem(loads, IBC, neq)

        # System solution
        disp = spsolve(stiff_mat, rhs_vec)
        UC = pos.complete_disp(IBC, nodes, disp)
        E_nodes, S_nodes = pos.strain_nodes(nodes, els, mats[:,:2], UC)
        E_els, S_els = strain_els(els, E_nodes, S_nodes) # Calculate strains and stresses in elements

        vons = np.sqrt(S_els[:,0]**2 - (S_els[:,0]*S_els[:,1]) + S_els[:,1]**2 + 3*S_els[:,2]**2)

        # Remove/add elements
        RR_el = vons/vons.max() # Relative stress
        mask_del = RR_el < RR # Mask for elements to be deleted
        mask_els = protect_elsESO(els, loads, BC) # Mask of elements to do not remove
        mask_del *= mask_els  
        els = np.delete(els, mask_del, 0) # Delete elements
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

    return ELS, nodes


# %% Eso stiff based

def ESO_stiff(length, height, nx, ny, dirs, positions, niter, RR, ER, volfrac, plot=False):
    """
    Performs Evolutionary Structural Optimization (ESO) based on stiff for a beam structure.

    Parameters
    ----------
    length : float
        The length of the beam.
    height : float
        The height of the beam.
    nx : int
        The number of elements in the x direction.
    ny : int
        The number of elements in the y direction.
    dirs : list
        List of directions.
    positions : list
        List of positions.
    niter : int
        The number of iterations for the ESO process.
    RR : float
        The relative stress threshold for removing elements.
    ER : float
        The increment of RR for each iteration.
    volfrac : float
        The volume fraction for the optimal structure.
    plot : bool, optional
        If True, plot the initial and optimized mesh. Defaults to False.

    Returns
    -------
    ELS: ndarray
        The optimized elements of the structure.
    nodes: ndarray
        The optimized nodes of the structure.
    """
    nodes, mats, els, loads, BC = beam(L=length, H=height, nx=nx, ny=ny, dirs=dirs, positions=positions, n=1)
    elsI= np.copy(els)

    # System assembly
    assem_op, IBC, neq = ass.DME(nodes[:, -2:], els, ndof_el_max=8)
    stiff_mat, _ = ass.assembler(els, mats, nodes[:, :3], neq, assem_op)
    rhs_vec = ass.loadasem(loads, IBC, neq)

    # System solution
    disp = spsolve(stiff_mat, rhs_vec)
    UCI = pos.complete_disp(IBC, nodes, disp)
    E_nodesI, S_nodesI = pos.strain_nodes(nodes, els, mats[:,:2], UCI)

    niter = 200
    RR = 0.005 # Initial removal ratio
    ER = 0.05 # Removal ratio increment
    V_opt = volume(els, length, height, nx, ny).sum() * volfrac # Optimal volume
    ELS = None
    for _ in range(niter):
        # Check equilibrium
        if not np.allclose(stiff_mat.dot(disp)/stiff_mat.max(), rhs_vec/stiff_mat.max()) or volume(els, length, height, nx, ny).sum() < V_opt: 
            break # Check equilibrium/volume and stop if not
        
        # System assembly
        assem_op, IBC, neq = ass.DME(nodes[:, -2:], els, ndof_el_max=8)
        stiff_mat, _ = ass.assembler(els, mats, nodes[:, :3], neq, assem_op)
        rhs_vec = ass.loadasem(loads, IBC, neq)

        # System solution
        disp = spsolve(stiff_mat, rhs_vec)
        UC = pos.complete_disp(IBC, nodes, disp)
        E_nodes, S_nodes = pos.strain_nodes(nodes, els, mats[:,:2], UC)
        E_els, S_els = strain_els(els, E_nodes, S_nodes) # Calculate strains and stresses in elements
        print("Number of elements: {}".format(els.shape[0]))

        # Compute Sensitivity number
        sensi_number = sensitivity_elsESO(nodes, mats, els, UC) # Sensitivity number
        mask_del = sensi_number < RR # Mask of elements to be removed
        mask_els = protect_elsESO(els, loads, BC) # Mask of elements to do not remove
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

    return ELS, nodes


# %% BESO

def BESO(length, height, nx, ny, dirs, positions, niter, t, ER, volfrac, plot=False):
    """
    Performs Evolutionary Structural Optimization (ESO) based on stiff for a beam structure.

    Parameters
    ----------
    length : float
        The length of the beam.
    height : float
        The height of the beam.
    nx : int
        The number of elements in the x direction.
    ny : int
        The number of elements in the y direction.
    dirs : list
        List of directions.
    positions : list
        List of positions.
    niter : int
        The number of iterations for the ESO process.
    t : float
        Threshold for error.
    ER : float
        The increment of RR for each iteration.
    volfrac : float
        The volume fraction for the optimal structure.
    plot : bool, optional
        If True, plot the initial and optimized mesh. Defaults to False.

    Returns
    -------
    ELS: ndarray
        The optimized elements of the structure.
    nodes: ndarray
        The optimized nodes of the structure.
    """
    nodes, mats, els, loads, BC = beam(L=length, H=height, nx=nx, ny=ny, dirs=dirs, positions=positions, n=1)
    elsI = np.copy(els)

    # System assembly
    assem_op, IBC, neq = ass.DME(nodes[:, -2:], els, ndof_el_max=8)
    stiff_mat, _ = ass.assembler(els, mats, nodes[:, :3], neq, assem_op)
    rhs_vec = ass.loadasem(loads, IBC, neq)

    # System solution
    disp = spsolve(stiff_mat, rhs_vec)
    UCI = pos.complete_disp(IBC, nodes, disp)
    E_nodesI, S_nodesI = pos.strain_nodes(nodes, els, mats[:,:2], UCI)

    r_min = np.linalg.norm(nodes[0,1:3] - nodes[1,1:3]) * 1 # Radius for the sensitivity filter
    adj_nodes = adjacency_nodes(nodes, els) # Adjacency nodes
    centers = center_els(nodes, els) # Centers of elements

    Vi = volume(els, length, height, nx, ny) # Initial volume
    V_opt = Vi.sum() * volfrac # Optimal volume

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
        V = Vi[mask].sum() # Volume of the structure

        # Check equilibrium
        if not np.allclose(stiff_mat.dot(disp)/stiff_mat.max(), rhs_vec/stiff_mat.max()) or volume(els, length, height, nx, ny).sum() < V_opt: 
            break

        # Storage the solution
        ELS = els_del 

        # System assembly
        assem_op, IBC, neq = ass.DME(nodes[:, -2:], els, ndof_el_max=8)
        stiff_mat, _ = ass.assembler(els, mats, nodes[:, :3], neq, assem_op)
        rhs_vec = ass.loadasem(loads, IBC, neq)

        # System solution
        disp = spsolve(stiff_mat, rhs_vec)
        UC = pos.complete_disp(IBC, nodes, disp)
        E_nodes, S_nodes = pos.strain_nodes(nodes, els, mats[:,:2], UC)
        E_els, S_els = strain_els(els, E_nodes, S_nodes) # Calculate strains and stresses in elements

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
        mask_els = protect_els(els[np.invert(mask)], els.shape[0], loads, BC) # Mask of elements to be protected
        mask = np.bitwise_or(mask, mask_els) 
        del_node(nodes, els[mask], loads, BC) # Delete nodes

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

    return ELS, nodes

# %% SIMP

def SIMP(length, height, nx, ny, dirs, positions, niter, penal, plot=False):
    """
    Performs Solid Isotropic Material with Penalization (SIMP) for topology optimization.

    Parameters
    ----------
    length : float
        The length of the beam.
    height : float
        The height of the beam.
    nx : int
        The number of elements in the x direction.
    ny : int
        The number of elements in the y direction.
    dirs : list
        List of directions.
    positions : list
        List of positions.
    niter : int
        The number of iterations for the SIMP process.
    penal : float
        Penalization factor used in the SIMP method.
    plot : bool, optional
        If True, plot the initial and optimized mesh. Defaults to False.

    Returns
    -------
    rho: ndarray
        The optimized density distribution of the structure.
    """
    # Initialize variables
    Emin=1e-9 # Minimum young modulus of the material
    Emax=1.0 # Maximum young modulus of the material

    nodes, mats, els, loads, _ = beam(L=length, H=height, nx=nx, ny=ny, dirs=dirs, positions=positions, n=1)

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