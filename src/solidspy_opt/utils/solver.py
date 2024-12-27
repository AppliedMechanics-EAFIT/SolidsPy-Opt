import numpy as np
from scipy.sparse import coo_matrix
from scipy.spatial.distance import cdist
import solidspy.uelutil as uel 
import solidspy.assemutil as ass 

def protect_els(els, nels, loads, BC, nnodes):
    """
    Compute an mask array with the elements that don't must be deleted.
    
    Parameters
    ----------
    els : ndarray
        Array with models elements
    nels : ndarray
        Number of elements
    loads : ndarray
        Array with models loads
    BC : ndarray 
        Boundary conditions nodes
        
    Returns
    -------
    mask_els : ndarray 
        Array with the elements that don't must be deleted.
    """   
    mask_els = np.zeros(nels, dtype=bool)
    protect_nodes = np.hstack((loads[:,0], BC)).astype(int)
    protect_index = None
    for p in protect_nodes:
        protect_index = np.argwhere(els[:, -nnodes:] == p)[:,0]
        mask_els[els[protect_index,0]] = True
        
    return mask_els

def del_node(nodes, els, loads, BC, dim_problem, nnodes):
    """
    Retricts nodes dof that aren't been used and free up the nodes that are in use.
    
    Parameters
    ----------
    nodes : ndarray
        Array with models nodes
    els : ndarray
        Array with models elements
    loads : ndarray
        Array with models loads
    BC : ndarray 
        Boundary conditions nodes

    Returns
    -------
    """   
    protect_nodes = np.hstack((loads[:,0], BC)).astype(int)
    for n in nodes[:,0]:
        if n not in els[:, -nnodes:]:
            nodes[int(n), -dim_problem:] = -1
        elif n not in protect_nodes and n in els[:, -nnodes:]:
            nodes[int(n), -dim_problem:] = 0


def volume(els, length, height, nx, ny):
    """
    Compute volume.
    
    Parameters
    ----------
    els : ndarray
        Array with models elements.
    length : ndarray
        Length of the beam.
    height : ndarray
        Height of the beam.
    nx : float
        Number of elements in x direction.
    ny : float
        Number of elements in y direction.

    Return 
    ----------
    V: float
        Volume of the structure.
    """

    dy = length / nx
    dx = height / ny
    V = dx * dy * np.ones(els.shape[0])

    return V

def sensitivity_elsBESO(nodes, mats, els, mask, UC, uel_func, nnodes, dim_problem):
    """
    Calculate the sensitivity number for each element.
    
    Parameters
    ----------
    nodes : ndarray
        Array with models nodes
    mats : ndarray
        Array with models materials
    els : ndarray
        Array with models elements
    mask : ndarray
        Mask of optimal estructure
    UC : ndarray
        Displacements at nodes

    Returns
    -------
    sensi_number : ndarray
        Sensitivity number for each element.
    """   
    sensi_number = []
    for el in els:
        if mask[el[0]] == False:
            sensi_number.append(0)
            continue
        params = tuple(mats[el[2], :])
        node_el = el[-nnodes:]
        elcoor = nodes[node_el, 1:-dim_problem]
        kloc, _ = uel_func(elcoor, params)

        U_el = UC[node_el]
        U_el = np.reshape(U_el, (nnodes*dim_problem ,1))
        a_i = 0.5 * U_el.T.dot(kloc.dot(U_el))[0,0]
        sensi_number.append(a_i)
    sensi_number = np.array(sensi_number)
    sensi_number = sensi_number/sensi_number.max()

    return sensi_number

def sensitivity_elsSIMP(nodes, mats, els, UC, uel_func, nnodes, dim_problem):
    """
    Calculate the sensitivity number for each element.
    
    Parameters
    ----------
    nodes : ndarray
        Array with models nodes
    mats : ndarray
        Array with models materials
    els : ndarray
        Array with models elements
    mask : ndarray
        Mask of optimal estructure
    UC : ndarray
        Displacements at nodes

    Returns
    -------
    sensi_number : ndarray
        Sensitivity number for each element.
    """   
    sensi_number = []
    for el in els:
        params = tuple(mats[el[2], :])
        node_el = el[-nnodes:]
        elcoor = nodes[node_el, 1:-dim_problem]
        kloc, _ = uel_func(elcoor, params)

        U_el = UC[node_el]
        U_el = np.reshape(U_el, (nnodes*dim_problem ,1))
        a_i = U_el.T.dot(kloc.dot(U_el))[0,0]
        sensi_number.append(a_i)
    sensi_number = np.array(sensi_number)
    sensi_number = sensi_number/sensi_number.max()

    return sensi_number

def adjacency_nodes(nodes, els, nnodes):
    """
    Create an adjacency matrix for the elements connected to each node.
    
    Parameters
    ----------
    nodes : ndarray
        Array with models nodes.
    els : ndarray
        Array with models elements.
        
    Returns
    -------
    adj_nodes : ndarray, nodes.shape[0]
        Adjacency elements for each node.
    """
    adj_nodes = []
    for n in nodes[:, 0]:
        adj_els = np.argwhere(els[:, -nnodes:] == n)[:,0]
        adj_nodes.append(adj_els)
    return adj_nodes

def center_els(nodes, els):
    """
    Calculate the center of each element.
    
    Parameters
    ----------
    nodes : ndarray
        Array with models nodes.
    els : ndarray
        Array with models elements.
        
    Returns
    -------
    centers : ndarray, nodes.shape[0]
        Center of each element.
    """
    centers = []
    for el in els:
        n = nodes[el[-4:], 1:3]
        center = np.array([n[1,0] + (n[0,0] - n[1,0])/2, n[2,1] + (n[0,1] - n[2,1])/2])
        centers.append(center)
    centers = np.array(centers)
    return centers

def sensitivity_nodes(nodes, adj_nodes, centers, sensi_els, dim_problem):
    """
    Calculate the sensitivity of each node.
    
    Parameters
    ----------
    nodes : ndarray
        Array with models nodes
    adj_nodes : ndarray
        Adjacency matrix of nodes
    centers : ndarray
        Array with center of elements
    sensi_els : ndarra
        Sensitivity of each element without filter
        
    Returns
    -------
    sensi_nodes : ndarray
        Sensitivity of each nodes
    """
    sensi_nodes = []
    for n in nodes:
        connected_els = np.array(adj_nodes[int(n[0])], dtype=int)
        if connected_els.shape[0] > 1:
            delta = centers[connected_els] - n[1:-dim_problem]
            r_ij = np.linalg.norm(delta, axis=1) 
            w_i = 1/(connected_els.shape[0] - 1) * (1 - r_ij/r_ij.sum())
            sensi = (w_i * sensi_els[connected_els]).sum(axis=0)
        else:
            sensi = sensi_els[connected_els[0]]
        sensi_nodes.append(sensi)
    sensi_nodes = np.array(sensi_nodes)

    return sensi_nodes

import numpy as np

def sensitivity_filter(nodes, centers, sensi_nodes, r_min, dim_problem):
    """
    Performs the sensitivity filter.

    Parameters
    ----------
    nodes : ndarray
        Array with the model's nodes.
        Shape: (n_nodes, d) -- typically includes X, Y, Z coords, etc.
    centers : ndarray
        Array with the centers of the elements (where we want to compute the filtered sensitivity).
        Shape: (n_elements, dim_problem)
    sensi_nodes : ndarray
        Array with nodal sensitivities.
        Shape: (n_nodes, )
    r_min : float
        Minimum distance for the filter radius.
    dim_problem : int
        Number of spatial dimensions (e.g., 2 or 3).
        
    Returns
    -------
    sensi_els : ndarray
        Sensitivity of each element with the filter applied.
        Shape: (n_elements, )
    """
    sensi_els = []
    for c in centers:
        delta = nodes[:, 1:-dim_problem] - c
        r_ij = np.linalg.norm(delta, axis=1)
        omega_i = (r_ij < r_min)
        n_omega = omega_i.sum() 

        if n_omega == 0:
            sensi_els.append(0.0)
        elif n_omega == 1:
            idx = np.where(omega_i)[0][0]
            sensi_els.append(sensi_nodes[idx])
        else:
            # We can safely apply the original formula
            w = 1.0 / (n_omega - 1) * (1.0 - r_ij[omega_i] / r_ij[omega_i].sum())
            filtered_sensi = (w * sensi_nodes[omega_i]).sum() / w.sum()
            sensi_els.append(filtered_sensi)

    sensi_els = np.array(sensi_els)

    # Normalize by maximum for post-processing
    max_val = np.max(sensi_els)  # Watch out for possible zero or negative max
    if max_val > 0:
        sensi_els /= max_val

    return sensi_els


def sensitivity_elsESO(nodes, mats, els, UC, uel_func, dim_problem, nnodes):
    """
    Calculate the sensitivity number for each element.
    
    Parameters
    ----------
    nodes : ndarray
        Array with models nodes
    mats : ndarray
        Array with models materials
    els : ndarray
        Array with models elements
    UC : ndarray
        Displacements at nodes

    Returns
    -------
    sensi_number : ndarray
        Sensitivity number for each element.
    """   
    sensi_number = []
    for el in els:
        params = tuple(mats[el[2], :])
        node_el = el[-nnodes:]
        elcoor = nodes[node_el, 1:-dim_problem]
        kloc, _ = uel_func(elcoor, params)

        U_el = UC[node_el]
        U_el = np.reshape(U_el, (nnodes*dim_problem ,1))
        a_i = 0.5 * U_el.T.dot(kloc.dot(U_el))[0,0]
        sensi_number.append(a_i)
    sensi_number = np.array(sensi_number)
    sensi_number = sensi_number/sensi_number.max()

    return sensi_number

def strain_els(els, E_nodes, S_nodes):
    """
    Compute the elements strains and stresses.
    
    Get from: https://github.com/AppliedMechanics-EAFIT/SolidsPy/blob/master/solidspy/solids_GUI.py
    
    Parameters
    ----------
    els : ndarray
        Array with models elements
    E_nodes : ndarray
        Strains at nodes.
    S_nodes : ndarray
        Stresses at nodes.
        
    Returns
    -------
    E_els : ndarray (nnodes, 3)
        Strains at elements.
    S_els : ndarray (nnodes, 3)
        Stresses at elements.
    """   
    
    E_els = []
    S_els = []
    for el in els:
        strain_nodes = np.take(E_nodes, list(el[3:]), 0)
        stress_nodes = np.take(S_nodes, list(el[3:]), 0)
        strain_elemt = (strain_nodes[0] + strain_nodes[1] + strain_nodes[2] + strain_nodes[3]) / 4
        stress_elemt = (stress_nodes[0] + stress_nodes[1] + stress_nodes[2] + stress_nodes[3]) / 4
        E_els.append(strain_elemt)
        S_els.append(stress_elemt)
    E_els = np.array(E_els)
    S_els = np.array(S_els)
    
    return E_els, S_els

def protect_elsESO(els, loads, BC, nnodes):
    """
    Compute an mask array with the elements that don't must be deleted.
    
    Parameters
    ----------
    els : ndarray
        Array with models elements
    loads : ndarray
        Array with models loads
    BC : ndarray 
        Boundary conditions nodes
        
    Returns
    -------
    mask_els : ndarray 
        Array with the elements that don't must be deleted.
    """   
    mask_els = np.ones_like(els[:,0], dtype=bool)
    protect_nodes = np.hstack((loads[:,0], BC)).astype(int)
    protect_index = None
    for p in protect_nodes:
        protect_index = np.argwhere(els[:, -nnodes:] == p)[:,0]
        mask_els[protect_index] = False
        
    return mask_els

def del_nodeESO(nodes, els, nnodes, dim_problem):
    """
    Retricts nodes dof that aren't been used.
    
    Parameters
    ----------
    nodes : ndarray
        Array with models nodes
    els : ndarray
        Array with models elements

    Returns
    -------
    """   
    n_nodes = nodes.shape[0]
    for n in range(n_nodes):
        if n not in els[:, -nnodes:]:
            nodes[n, -dim_problem:] = -1

def sparse_assem(elements, nodes, mats, neq, assem_op, dim_problem, uel):
    """
    Assembles the global stiffness matrix
    using a sparse storing scheme

    Parameters
    ----------
    elements : ndarray (int)
      Array with the number for the nodes in each element.
    mats    : ndarray (float)
      Array with the material profiles.
    neq : int
      Number of active equations in the system.
    assem_op : ndarray (int)
      Assembly operator.
    uel : callable function (optional)
      Python function that returns the local stiffness matrix.
    kloc : ndarray 
      Stiffness matrix of a single element

    Returns
    -------
    stiff : sparse matrix (float)
      Array with the global stiffness matrix in a sparse
      Compressed Sparse Row (CSR) format.
    """
    rows = []
    cols = []
    stiff_vals = []
    nels = elements.shape[0]
    for ele in range(nels):
        kloc, _ = ass.retriever(elements, mats, nodes[:, :-dim_problem], ele, uel=uel)
        kloc_ = kloc * mats[elements[ele, 0], 2]
        ndof = kloc.shape[0]
        dme = assem_op[ele, :ndof]
        for row in range(ndof):
            glob_row = dme[row]
            if glob_row != -1:
                for col in range(ndof):
                    glob_col = dme[col]
                    if glob_col != -1:
                        rows.append(glob_row)
                        cols.append(glob_col)
                        stiff_vals.append(kloc_[row, col])

    stiff = coo_matrix((stiff_vals, (rows, cols)), shape=(neq, neq)).tocsr()

    return stiff
    
def optimality_criteria(nel, rho, d_c, g):
    """
    Optimality criteria method.

    Parameters
    ----------
    nel : int
        Number of elements.
    rho : ndarray
        Array with the density of each element.
    d_c : ndarray
        Array with the derivative of the compliance.
    g : float
        Volume constraint.

    Returns
    -------
    rho_new : ndarray
        Array with the new density of each element.
    gt : float
        Volume constraint.
    """
    l1=0
    l2=1e9
    move=0.2
    rho_new=np.zeros(nel)
    while (l2-l1)/(l1+l2)>1e-3: 
        lmid=0.5*(l2+l1)
        rho_new[:]= np.maximum(0.0,np.maximum(rho-move,np.minimum(1.0,np.minimum(rho+move,rho*np.sqrt(-d_c/lmid)))))
        gt=g+np.sum(((rho_new-rho)))
        if gt>0 :
            l1=lmid
        else:
            l2=lmid
    return (rho_new, gt)



def density_filter(centers, r_min, rho, d_rho):
    """
    Performe the sensitivity filter.
    
    Parameters
    ----------
    centers : ndarray
        Array with the centers of each element.
    r_min : float
        Minimum radius of the filter.
    rho : ndarray
        Array with the density of each element.
    d_rho : ndarray
        Array with the derivative of the density of each element.
        
    Returns
    -------
    densi_els : ndarray
        Sensitivity of each element with filter
    """
    dist = cdist(centers, centers, 'euclidean')
    delta = r_min - dist
    H = np.maximum(0.0, delta)
    densi_els = (rho*H*d_rho).sum(1)/(H.sum(1)*np.maximum(0.001,rho))

    return densi_els

def center_els(nodes, els):
    """
    Calculate the center of each element.
    
    Parameters
    ----------
    nodes : ndarray
        Array with models nodes.
    els : ndarray
        Array with models elements.
        
    Returns
    -------
    centers : 
        Centers of each element.
    """
    centers = np.zeros((els.shape[0], 2))
    for el in els:
        n = nodes[el[-4:], 1:3]
        center = np.array([n[1,0] + (n[0,0] - n[1,0])/2, n[2,1] + (n[0,1] - n[2,1])/2])
        centers[int(el[0])] = center

    return centers

def calculate_element_centers(nodes, els, dim_problem=2, nnodes=4):
    """
    Calculate the center of each element for 2D and 3D meshes.
    
    Parameters
    ----------
    nodes : ndarray
        Array of node coordinates with format [node number, x, y, z, BC].
    els : ndarray
        Array of elements with format [element number, material, node 1, node 2, ...].
    dim_problem : int, optional
        Dimension of the problem (2 for 2D, 3 for 3D). Default is 2.
    nnodes : int, optional
        Number of nodes per element. For 2D: 3 or 4, for 3D: 4 or 8. Default is 4.
    
    Returns
    -------
    centers : ndarray
        Array containing the center coordinates of each element.
        For 2D: shape (n_elements, 2)
        For 3D: shape (n_elements, 3)
    """
    # Input validation
    if dim_problem not in [2, 3]:
        raise ValueError("dim_problem must be 2 or 3")
    
    valid_nnodes = {2: [3, 4], 3: [4, 8]}
    if nnodes not in valid_nnodes[dim_problem]:
        raise ValueError(f"For {dim_problem}D problems, nnodes must be {valid_nnodes[dim_problem]}")
    
    # Initialize centers array
    n_elements = els.shape[0]
    centers = np.zeros((n_elements, dim_problem))
    
    # Calculate center for each element
    for el in els:
        el_num = int(el[0])
        # Get node indices (skip element number and material)
        node_indices = el[2:2+nnodes].astype(int)
        # Get coordinates of element nodes (exclude node number and BC)
        node_coords = nodes[node_indices, 1:1+dim_problem]
        
        if dim_problem == 2:
            if nnodes == 3:  # Triangle
                # Centroid of triangle = average of vertices
                center = np.mean(node_coords, axis=0)
            else:  # Quadrilateral
                # For quad elements, use the average of all vertices
                center = np.mean(node_coords, axis=0)
        
        else:  # 3D
            if nnodes == 4:  # Tetrahedron
                # Centroid of tetrahedron = average of vertices
                center = np.mean(node_coords, axis=0)
            else:  # Hexahedron (8 nodes)
                # For hex elements, use the average of all vertices
                center = np.mean(node_coords, axis=0)
        
        centers[el_num] = center
    
    return centers