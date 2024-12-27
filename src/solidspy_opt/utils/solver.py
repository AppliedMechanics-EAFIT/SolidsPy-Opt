from typing import List, Callable, Tuple
import numpy as np
from numpy.typing import NDArray
from scipy.sparse import coo_matrix
from scipy.spatial.distance import cdist
import solidspy.uelutil as uel
import solidspy.assemutil as ass


def protect_els(
    els: NDArray[np.float64],
    nels: int,
    loads: NDArray[np.float64],
    BC: NDArray[np.float64],
    nnodes: int
) -> NDArray[np.bool_]:
    """
    Compute a mask array with the elements that must not be deleted.

    Parameters
    ----------
    els : NDArray[np.float64]
        Array with model elements. Shape: (n_els, ?).
        The last `nnodes` columns typically contain the node indices.
    nels : int
        Number of elements.
    loads : NDArray[np.float64]
        Array with the model loads. The first column usually has node indices.
    BC : NDArray[np.float64]
        Array with the boundary condition node indices.
    nnodes : int
        Number of nodes per element (e.g., 4 for a 2D quad).

    Returns
    -------
    mask_els : NDArray[np.bool_]
        Boolean array of shape (nels,) indicating which elements must NOT be deleted.
    """
    mask_els = np.zeros(nels, dtype=bool)
    protect_nodes = np.hstack((loads[:, 0], BC)).astype(int)
    protect_index = None
    for p in protect_nodes:
        protect_index = np.argwhere(els[:, -nnodes:] == p)[:, 0]
        mask_els[els[protect_index, 0].astype(int)] = True

    return mask_els


def del_node(
    nodes: NDArray[np.float64],
    els: NDArray[np.float64],
    loads: NDArray[np.float64],
    BC: NDArray[np.float64],
    dim_problem: int,
    nnodes: int
) -> None:
    """
    Restricts the DOFs of nodes that are not used and frees up the nodes that are in use.

    Parameters
    ----------
    nodes : NDArray[np.float64]
        Array with model nodes. Shape: (n_nodes, ?).
        The last `dim_problem` entries typically store the BC flags/dofs.
    els : NDArray[np.float64]
        Array with model elements. Shape: (n_els, ?).
    loads : NDArray[np.float64]
        Array with the model loads. The first column usually has node indices.
    BC : NDArray[np.float64]
        Array with the boundary condition node indices.
    dim_problem : int
        Number of spatial dimensions (e.g., 2 or 3).
    nnodes : int
        Number of nodes per element.

    Returns
    -------
    None
    """
    protect_nodes = np.hstack((loads[:, 0], BC)).astype(int)
    for n in nodes[:, 0]:
        if n not in els[:, -nnodes:]:
            nodes[int(n), -dim_problem:] = -1
        elif (n not in protect_nodes) and (n in els[:, -nnodes:]):
            nodes[int(n), -dim_problem:] = 0


def volume(
    els: NDArray[np.float64],
    length: float,
    height: float,
    nx: int,
    ny: int
) -> NDArray[np.float64]:
    """
    Compute the (2D) "volume" (area) for each element.

    Parameters
    ----------
    els : NDArray[np.float64]
        Array with model elements. Shape: (n_els, ?).
    length : float
        Total length (e.g., in the x-direction).
    height : float
        Total height (e.g., in the y-direction).
    nx : int
        Number of elements in the x-direction.
    ny : int
        Number of elements in the y-direction.

    Returns
    -------
    V : NDArray[np.float64]
        Array of shape (n_els,) with the area (or "volume" in 2D) of each element.
    """
    dy = length / nx
    dx = height / ny
    V = dx * dy * np.ones(els.shape[0])
    return V


def sensitivity_elsBESO(
    nodes: NDArray[np.float64],
    mats: NDArray[np.float64],
    els: NDArray[np.float64],
    mask: NDArray[np.bool_],
    UC: NDArray[np.float64],
    uel_func: Callable[[NDArray[np.float64], Tuple[float, ...]], Tuple[NDArray[np.float64], NDArray[np.float64]]],
    nnodes: int,
    dim_problem: int
) -> NDArray[np.float64]:
    """
    Calculate the sensitivity number for each element for the BESO method.

    Parameters
    ----------
    nodes : NDArray[np.float64]
        Array with model nodes. Shape: (n_nodes, ?).
    mats : NDArray[np.float64]
        Array with model materials. Shape: (n_materials, ?).
    els : NDArray[np.float64]
        Array with model elements. Shape: (n_els, ?).
    mask : NDArray[np.bool_]
        Boolean mask indicating which elements are currently active.
    UC : NDArray[np.float64]
        Displacements at nodes. Shape: (n_nodes,).
    uel_func : Callable
        Function that returns the local stiffness matrix (and possibly other data).
    nnodes : int
        Number of nodes per element.
    dim_problem : int
        Number of spatial dimensions (2 or 3).

    Returns
    -------
    sensi_number : NDArray[np.float64]
        Normalized sensitivity array of shape (n_els,).
    """
    sensi_number = []
    for el in els:
        if not mask[int(el[0])]:
            sensi_number.append(0.0)
            continue
        params = tuple(mats[int(el[2]), :])
        node_el = el[-nnodes:].astype(int)
        elcoor = nodes[node_el, 1:-dim_problem]
        kloc, _ = uel_func(elcoor, params)

        U_el = UC[node_el]
        U_el = np.reshape(U_el, (nnodes * dim_problem, 1))
        a_i = 0.5 * U_el.T.dot(kloc.dot(U_el))[0, 0]
        sensi_number.append(a_i)
    sensi_number = np.array(sensi_number)
    max_val = sensi_number.max() if sensi_number.size > 0 else 1.0
    sensi_number /= max_val
    return sensi_number


def sensitivity_elsSIMP(
    nodes: NDArray[np.float64],
    mats: NDArray[np.float64],
    els: NDArray[np.float64],
    UC: NDArray[np.float64],
    uel_func: Callable[[NDArray[np.float64], Tuple[float, ...]], Tuple[NDArray[np.float64], NDArray[np.float64]]],
    nnodes: int,
    dim_problem: int
) -> NDArray[np.float64]:
    """
    Calculate the sensitivity number for each element for the SIMP method.

    Parameters
    ----------
    nodes : NDArray[np.float64]
        Array with model nodes. Shape: (n_nodes, ?).
    mats : NDArray[np.float64]
        Array with model materials. Shape: (n_materials, ?).
    els : NDArray[np.float64]
        Array with model elements. Shape: (n_els, ?).
    UC : NDArray[np.float64]
        Displacements at nodes. Shape: (n_nodes,).
    uel_func : Callable
        Function that returns the local stiffness matrix.
    nnodes : int
        Number of nodes per element.
    dim_problem : int
        Number of spatial dimensions (2 or 3).

    Returns
    -------
    sensi_number : NDArray[np.float64]
        Normalized sensitivity array of shape (n_els,).
    """
    sensi_number = []
    for el in els:
        params = tuple(mats[int(el[2]), :])
        node_el = el[-nnodes:].astype(int)
        elcoor = nodes[node_el, 1:-dim_problem]
        kloc, _ = uel_func(elcoor, params)

        U_el = UC[node_el]
        U_el = np.reshape(U_el, (nnodes * dim_problem, 1))
        a_i = U_el.T.dot(kloc.dot(U_el))[0, 0]
        sensi_number.append(a_i)
    sensi_number = np.array(sensi_number)
    max_val = sensi_number.max() if sensi_number.size > 0 else 1.0
    sensi_number /= max_val
    return sensi_number


def adjacency_nodes(
    nodes: NDArray[np.float64],
    els: NDArray[np.float64],
    nnodes: int
) -> List[NDArray[np.int_]]:
    """
    Create a list of element indices connected to each node.

    Parameters
    ----------
    nodes : NDArray[np.float64]
        Array with model nodes. Shape: (n_nodes, ?).
        The first column is typically the node ID.
    els : NDArray[np.float64]
        Array with model elements. Shape: (n_els, ?).
        The last `nnodes` columns contain node indices.
    nnodes : int
        Number of nodes per element.

    Returns
    -------
    adj_nodes : List[NDArray[np.int_]]
        A list of length n_nodes. adj_nodes[i] is an array of element indices
        that are connected to node i (based on node ID in the first column).
    """
    adj_nodes = []
    for n in nodes[:, 0].astype(int):
        adj_els = np.argwhere(els[:, -nnodes:] == n)[:, 0]
        adj_nodes.append(adj_els)
    return adj_nodes


def center_els(
    nodes: NDArray[np.float64],
    els: NDArray[np.float64]
) -> NDArray[np.float64]:
    """
    Calculate the center of each quadrilateral element in 2D.

    Parameters
    ----------
    nodes : NDArray[np.float64]
        Array with model nodes. Shape: (n_nodes, ?).
    els : NDArray[np.float64]
        Array with model elements. Shape: (n_els, ?).
        The last 4 columns typically contain the node indices of a quad element.

    Returns
    -------
    centers : NDArray[np.float64]
        Array of shape (n_els, 2) with the center of each element.
    """
    centers = []
    for el in els:
        n = nodes[el[-4:].astype(int), 1:3]
        # For a 4-node quad, pick the midpoint in x and y directions
        center = np.array([
            n[1, 0] + (n[0, 0] - n[1, 0]) / 2,
            n[2, 1] + (n[0, 1] - n[2, 1]) / 2
        ])
        centers.append(center)
    return np.array(centers)


def sensitivity_nodes(
    nodes: NDArray[np.float64],
    adj_nodes: List[NDArray[np.int_]],
    centers: NDArray[np.float64],
    sensi_els: NDArray[np.float64],
    dim_problem: int
) -> NDArray[np.float64]:
    """
    Calculate the sensitivity at each node.

    Parameters
    ----------
    nodes : NDArray[np.float64]
        Array with model nodes. Shape: (n_nodes, ?).
    adj_nodes : List[NDArray[np.int_]]
        A list where each entry corresponds to the element indices connected to a node.
    centers : NDArray[np.float64]
        Array of element centers. Shape: (n_els, 2) for 2D, (n_els, 3) for 3D.
    sensi_els : NDArray[np.float64]
        Sensitivity of each element without filtering. Shape: (n_els,).
    dim_problem : int
        Number of spatial dimensions (2 or 3).

    Returns
    -------
    sensi_nodes : NDArray[np.float64]
        Array of shape (n_nodes,) with the computed sensitivity for each node.
    """
    sensi_nodes = []
    for n in nodes:
        connected_els = np.array(adj_nodes[int(n[0])], dtype=int)
        if connected_els.shape[0] > 1:
            delta = centers[connected_els] - n[1:-dim_problem]
            r_ij = np.linalg.norm(delta, axis=1)
            w_i = 1.0 / (connected_els.shape[0] - 1) * (1 - r_ij / r_ij.sum())
            sensi = (w_i * sensi_els[connected_els]).sum(axis=0)
        else:
            sensi = sensi_els[connected_els[0]]
        sensi_nodes.append(sensi)
    return np.array(sensi_nodes)


def sensitivity_filter(
    nodes: NDArray[np.float64],
    centers: NDArray[np.float64],
    sensi_nodes: NDArray[np.float64],
    r_min: float,
    dim_problem: int
) -> NDArray[np.float64]:
    """
    Performs a simple nodal-based sensitivity filter and maps it back to elements.

    Parameters
    ----------
    nodes : NDArray[np.float64]
        Array with the model's nodes. Shape: (n_nodes, ?).
    centers : NDArray[np.float64]
        Array with the centers of the elements. Shape: (n_elements, dim_problem).
    sensi_nodes : NDArray[np.float64]
        Array with nodal sensitivities. Shape: (n_nodes,).
    r_min : float
        Minimum distance for the filter radius.
    dim_problem : int
        Number of spatial dimensions (2 or 3).

    Returns
    -------
    sensi_els : NDArray[np.float64]
        Filtered sensitivity of each element. Shape: (n_elements,).
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
            w = 1.0 / (n_omega - 1) * (1.0 - r_ij[omega_i] / r_ij[omega_i].sum())
            filtered_sensi = (w * sensi_nodes[omega_i]).sum() / w.sum()
            sensi_els.append(filtered_sensi)

    sensi_els = np.array(sensi_els)
    max_val = np.max(sensi_els) if sensi_els.size > 0 else 1.0
    if max_val > 0:
        sensi_els /= max_val
    return sensi_els


def sensitivity_elsESO(
    nodes: NDArray[np.float64],
    mats: NDArray[np.float64],
    els: NDArray[np.float64],
    UC: NDArray[np.float64],
    uel_func: Callable[[NDArray[np.float64], Tuple[float, ...]], Tuple[NDArray[np.float64], NDArray[np.float64]]],
    dim_problem: int,
    nnodes: int
) -> NDArray[np.float64]:
    """
    Calculate the sensitivity number for each element for the ESO method.

    Parameters
    ----------
    nodes : NDArray[np.float64]
        Array with model nodes. Shape: (n_nodes, ?).
    mats : NDArray[np.float64]
        Array with model materials. Shape: (n_materials, ?).
    els : NDArray[np.float64]
        Array with model elements. Shape: (n_els, ?).
    UC : NDArray[np.float64]
        Displacements at nodes. Shape: (n_nodes,).
    uel_func : Callable
        Function that returns the local stiffness matrix.
    dim_problem : int
        Number of spatial dimensions (2 or 3).
    nnodes : int
        Number of nodes per element.

    Returns
    -------
    sensi_number : NDArray[np.float64]
        Normalized sensitivity array of shape (n_els,).
    """
    sensi_number = []
    for el in els:
        params = tuple(mats[int(el[2]), :])
        node_el = el[-nnodes:].astype(int)
        elcoor = nodes[node_el, 1:-dim_problem]
        kloc, _ = uel_func(elcoor, params)

        U_el = UC[node_el]
        U_el = np.reshape(U_el, (nnodes * dim_problem, 1))
        a_i = 0.5 * U_el.T.dot(kloc.dot(U_el))[0, 0]
        sensi_number.append(a_i)
    sensi_number = np.array(sensi_number)
    max_val = sensi_number.max() if sensi_number.size > 0 else 1.0
    sensi_number /= max_val
    return sensi_number


def strain_els(
    els: NDArray[np.float64],
    E_nodes: NDArray[np.float64],
    S_nodes: NDArray[np.float64]
) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
    """
    Compute the average strains and stresses for each element by averaging over its nodes.

    Parameters
    ----------
    els : NDArray[np.float64]
        Array with model elements. Shape: (n_els, ?).
        Typically includes element ID and 4 node indices for 2D quads.
    E_nodes : NDArray[np.float64]
        Strains at nodes. Shape: (n_nodes, 3).
    S_nodes : NDArray[np.float64]
        Stresses at nodes. Shape: (n_nodes, 3).

    Returns
    -------
    E_els : NDArray[np.float64]
        Strains at elements. Shape: (n_els, 3).
    S_els : NDArray[np.float64]
        Stresses at elements. Shape: (n_els, 3).
    """
    E_els = []
    S_els = []
    for el in els:
        # el[3:7] -> the 4 node indices for a quad (if that is the format)
        strain_nodes = np.take(E_nodes, list(el[3:3 + 4].astype(int)), axis=0)
        stress_nodes = np.take(S_nodes, list(el[3:3 + 4].astype(int)), axis=0)
        strain_elemt = np.mean(strain_nodes, axis=0)
        stress_elemt = np.mean(stress_nodes, axis=0)
        E_els.append(strain_elemt)
        S_els.append(stress_elemt)
    return np.array(E_els), np.array(S_els)


def protect_elsESO(
    els: NDArray[np.float64],
    loads: NDArray[np.float64],
    BC: NDArray[np.float64],
    nnodes: int
) -> NDArray[np.bool_]:
    """
    Compute a mask array with the elements that must not be deleted (ESO variant).

    Parameters
    ----------
    els : NDArray[np.float64]
        Array with model elements. Shape: (n_els, ?).
    loads : NDArray[np.float64]
        Array with the model loads. The first column usually has node indices.
    BC : NDArray[np.float64]
        Array with the boundary condition node indices.
    nnodes : int
        Number of nodes per element.

    Returns
    -------
    mask_els : NDArray[np.bool_]
        Boolean array of shape (n_els,) indicating which elements must NOT be deleted.
    """
    mask_els = np.ones_like(els[:, 0], dtype=bool)
    protect_nodes = np.hstack((loads[:, 0], BC)).astype(int)
    for p in protect_nodes:
        protect_index = np.argwhere(els[:, -nnodes:] == p)[:, 0]
        mask_els[protect_index] = False
    return mask_els


def del_nodeESO(
    nodes: NDArray[np.float64],
    els: NDArray[np.float64],
    nnodes: int,
    dim_problem: int
) -> None:
    """
    Restricts nodes DOFs that are not used (ESO variant).

    Parameters
    ----------
    nodes : NDArray[np.float64]
        Array with model nodes. Shape: (n_nodes, ?).
        The last `dim_problem` entries typically store BC flags/dofs.
    els : NDArray[np.float64]
        Array with model elements. Shape: (n_els, ?).
    nnodes : int
        Number of nodes per element.
    dim_problem : int
        Number of spatial dimensions (2 or 3).

    Returns
    -------
    None
    """
    n_nodes = nodes.shape[0]
    for n in range(n_nodes):
        if n not in els[:, -nnodes:]:
            nodes[n, -dim_problem:] = -1


def sparse_assem(
    elements: NDArray[np.float64],
    nodes: NDArray[np.float64],
    mats: NDArray[np.float64],
    neq: int,
    assem_op: NDArray[np.int_],
    dim_problem: int,
    uel: Callable[[NDArray[np.float64], Tuple[float, ...]], Tuple[NDArray[np.float64], NDArray[np.float64]]]
) -> coo_matrix:
    """
    Assembles the global stiffness matrix using a sparse storing scheme.

    Parameters
    ----------
    elements : NDArray[np.float64]
        Array with the number for the nodes in each element (and other element info).
        Shape: (n_els, ?).
    nodes : NDArray[np.float64]
        Array with node coordinates and possibly BC data. Shape: (n_nodes, ?).
    mats : NDArray[np.float64]
        Array with the material properties. Shape: (n_materials, ?).
    neq : int
        Number of active equations in the system.
    assem_op : NDArray[np.int_]
        Assembly operator. Shape: (n_els, some_dof).
    dim_problem : int
        Number of spatial dimensions (2 or 3).
    uel : Callable
        Python function that returns the local stiffness matrix and other info.

    Returns
    -------
    stiff : coo_matrix
        Global stiffness matrix in a sparse (CSR) format. Shape: (neq, neq).
    """
    rows = []
    cols = []
    stiff_vals = []
    nels = elements.shape[0]
    for ele in range(nels):
        kloc, _ = ass.retriever(elements, mats, nodes[:, :-dim_problem], ele, uel=uel)
        # Example of scaling with a factor from mats: adjust if needed
        kloc_ = kloc * mats[elements[ele, 0].astype(int), 2]
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


def optimality_criteria(
    nel: int,
    rho: NDArray[np.float64],
    d_c: NDArray[np.float64],
    g: float
) -> Tuple[NDArray[np.float64], float]:
    """
    Optimality criteria method for topology optimization.

    Parameters
    ----------
    nel : int
        Number of elements.
    rho : NDArray[np.float64]
        Array with the density of each element. Shape: (nel,).
    d_c : NDArray[np.float64]
        Array with the derivative of the compliance. Shape: (nel,).
    g : float
        Volume constraint.

    Returns
    -------
    rho_new : NDArray[np.float64]
        Updated density array of shape (nel,).
    gt : float
        Updated volume constraint value.
    """
    l1 = 0.0
    l2 = 1e9
    move = 0.2
    rho_new = np.zeros(nel, dtype=np.float64)
    while (l2 - l1) / (l1 + l2) > 1e-3:
        lmid = 0.5 * (l2 + l1)
        rho_new[:] = np.maximum(
            0.0,
            np.maximum(
                rho - move,
                np.minimum(
                    1.0,
                    np.minimum(
                        rho + move,
                        rho * np.sqrt(-d_c / lmid)
                    )
                )
            )
        )
        gt = g + np.sum((rho_new - rho))
        if gt > 0:
            l1 = lmid
        else:
            l2 = lmid
    return rho_new, gt


def density_filter(
    centers: NDArray[np.float64],
    r_min: float,
    rho: NDArray[np.float64],
    d_rho: NDArray[np.float64]
) -> NDArray[np.float64]:
    """
    Perform a density-based filter for topology optimization.

    Parameters
    ----------
    centers : NDArray[np.float64]
        Array with the centers of each element. Shape: (n_els, dim).
    r_min : float
        Minimum radius of the filter.
    rho : NDArray[np.float64]
        Array with the density of each element. Shape: (n_els,).
    d_rho : NDArray[np.float64]
        Derivative of some measure (e.g., density or compliance). Shape: (n_els,).

    Returns
    -------
    densi_els : NDArray[np.float64]
        Filtered sensitivities of shape (n_els,).
    """
    dist = cdist(centers, centers, "euclidean")
    delta = r_min - dist
    H = np.maximum(0.0, delta)
    densi_els = (rho * H * d_rho).sum(axis=1) / (H.sum(axis=1) * np.maximum(0.001, rho))
    return densi_els


def calculate_element_centers(
    nodes: NDArray[np.float64],
    els: NDArray[np.float64],
    dim_problem: int = 2,
    nnodes: int = 4
) -> NDArray[np.float64]:
    """
    Calculate the center of each element for 2D and 3D meshes.

    Parameters
    ----------
    nodes : NDArray[np.float64]
        Array of node coordinates and possibly BC flags.
        Format: [node number, x, y, (z), BC...].
    els : NDArray[np.float64]
        Array of elements.
        Format: [element number, material, node1, node2, ...].
    dim_problem : int, optional
        Dimension of the problem (2 for 2D, 3 for 3D). Default is 2.
    nnodes : int, optional
        Number of nodes per element. For 2D: 3 or 4, for 3D: 4 or 8. Default is 4.

    Returns
    -------
    centers : NDArray[np.float64]
        Array of shape (n_elements, dim_problem) containing the center coordinates of each element.
        For 2D: shape (n_elements, 2)
        For 3D: shape (n_elements, 3)
    """
    if dim_problem not in [2, 3]:
        raise ValueError("dim_problem must be 2 or 3.")

    valid_nnodes = {2: [3, 4], 3: [4, 8]}
    if nnodes not in valid_nnodes[dim_problem]:
        raise ValueError(
            f"For {dim_problem}D problems, nnodes must be one of {valid_nnodes[dim_problem]}."
        )

    n_elements = els.shape[0]
    centers = np.zeros((n_elements, dim_problem))

    for el in els:
        el_num = int(el[0])
        node_indices = el[2 : 2 + nnodes].astype(int)
        node_coords = nodes[node_indices, 1 : 1 + dim_problem]

        if dim_problem == 2:
            # Triangles or quads
            center = np.mean(node_coords, axis=0)
        else:
            # Tetrahedrons or hexahedrons
            center = np.mean(node_coords, axis=0)

        centers[el_num] = center

    return centers
