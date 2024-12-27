import numpy as np
import solidspy.preprocesor as pre

def structures(
    L: float = 10,
    H: float = 10,
    E: float = 206.8e9,
    v: float = 0.28,
    nx: int = 20,
    ny: int = 20,
    dirs: np.ndarray = np.array([]),
    positions: np.ndarray = np.array([]),
    n: int = 1
):
    """
    This function selects the appropriate structure function to call based on n.

    Parameters
    ----------
    L : float, optional
        Structure's length, by default 10
    H : float, optional
        Structure's height, by default 10
    E : float, optional
        Young's modulus, by default 206.8e9
    v : float, optional
        Poisson's ratio, by default 0.28
    nx : int, optional
        Number of elements in the x direction, by default 20
    ny : int, optional
        Number of elements in the y direction, by default 20
    dirs : np.ndarray, optional
        Directions of the loads, default np.array([])
    positions : np.ndarray, optional
        Positions of the loads, default np.array([])
    n : int, optional
        Selector for the structure function to call, by default 1

    Returns
    -------
    nodes, mats, els, loads, idx_BC
        The outputs returned by the selected structure function.
    """
    structure_map = {
        1: structure_1,
        2: structure_2,
        # If you had more, you could add them here,
        # 3: structure_3, etc.
    }

    if n not in structure_map:
        raise ValueError(
            f"Invalid structure ID: {n}. "
            f"Available structures are: {list(structure_map.keys())}"
        )

    # Call the function corresponding to 'n'
    return structure_map[n](L, H, E, v, nx, ny, dirs, positions)



def structure_1(L=10, H=10, E=206.8e9, v=0.28, nx=20, ny=20, dirs=np.array([]), positions=np.array([])):
    """
    Make the mesh for a cuadrilateral model with cantilever structure's constrains.

    Parameters
    ----------
    L : float, optional
        Length of the structure, by default 10
    H : float, optional
        Height of the structure, by default 10
    E : float, optional
        Young's modulus, by default 206.8e9
    v : float, optional
        Poisson's ratio, by default 0.28
    nx : int, optional
        Number of elements in the x direction, by default 20
    ny : int, optional
        Number of elements in the y direction, by default 20
    dirs : ndarray, optional
        An array with the directions of the loads, by default empty array. [[0,1],[1,0],[0,-1]]
    positions : ndarray, optional
        An array with the positions of the loads, by default empty array. [[61,30], [1,30], [30, 1]]

    Returns
    -------
    nodes : ndarray
        Array of nodes
    mats : ndarray
        Array of material properties
    els : ndarray
        Array of elements
    loads : ndarray
        Array of loads
    """
    x, y, els = pre.rect_grid(L, H, nx, ny)
    mats = np.zeros((els.shape[0], 3))
    mats[:] = [E,v,1]
    nodes = np.zeros(((nx + 1)*(ny + 1), 5))
    nodes[:, 0] = range((nx + 1)*(ny + 1))
    nodes[:, 1] = x
    nodes[:, 2] = y
    mask = (x==-L/2)
    nodes[mask, 3:] = -1

    loads = np.zeros((dirs.shape[0], 3), dtype=int)
    node_index = nx*positions[:,0]+(positions[:,0]-positions[:,1])

    loads[:, 0] = node_index
    loads[:, 1] = dirs[:,0]
    loads[:, 2] = dirs[:,1]
    idx_BC = nodes[mask, 0]
    return nodes, mats, els, loads, idx_BC

def structure_2(L=10, H=10, E=206.8e9, v=0.28, nx=20, ny=20, dirs=np.array([]), positions=np.array([])):
    """
    Make the mesh for a cuadrilateral model with simply supported structure's constrains.

    Parameters
    ----------
    L : float, optional
        Length of the structure, by default 10
    H : float, optional
        Height of the structure, by default 10
    E : float, optional
        Young's modulus, by default 206.8e9
    v : float, optional
        Poisson's ratio, by default 0.28
    nx : int, optional
        Number of elements in the x direction, by default 20
    ny : int, optional
        Number of elements in the y direction, by default 20
    dirs : ndarray, optional
        An array with the directions of the loads, by default empty array. [[0,1],[1,0],[0,-1]]
    positions : ndarray, optional
        An array with the positions of the loads, by default empty array. [[61,30], [1,30], [30, 1]]

    Returns
    -------
    nodes : ndarray
        Array of nodes
    mats : ndarray
        Array of material properties
    els : ndarray
        Array of elements
    loads : ndarray
        Array of loads
    """
    x, y, els = pre.rect_grid(L, H, nx, ny)
    mats = np.zeros((els.shape[0], 3))
    mats[:] = [E,v,1]
    nodes = np.zeros(((nx + 1)*(ny + 1), 5))
    nodes[:, 0] = range((nx + 1)*(ny + 1))
    nodes[:, 1] = x
    nodes[:, 2] = y
    mask_1 = (x == L/2) & (y > H/4)
    mask_2 = (x == L/2) & (y < -H/4)
    mask = np.bitwise_or(mask_1, mask_2)
    nodes[mask, 3:] = -1

    loads = np.zeros((dirs.shape[0], 3), dtype=int)
    node_index = nx*positions[:,0]+(positions[:,0]-positions[:,1])

    loads[:, 0] = node_index
    loads[:, 1] = dirs[:,0]
    loads[:, 2] = dirs[:,1]
    idx_BC = nodes[mask, 0]
    return nodes, mats, els, loads, idx_BC

def structure_3d(L=10, H=10, W=10, E=206.8e9, v=0.28, nx=10, ny=10, nz=10, dirs=np.array([]), positions=np.array([])):
    """
    Make the mesh for a cubic model with cantilever structure constraints.

    Parameters
    ----------
    L : float, optional
        Length of the structure (x-direction), by default 10
    H : float, optional
        Height of the structure (y-direction), by default 10
    W : float, optional
        Width of the structure (z-direction), by default 10
    E : float, optional
        Young's modulus, by default 206.8e9
    v : float, optional
        Poisson's ratio, by default 0.28
    nx : int, optional
        Number of elements in the x direction, by default 10
    ny : int, optional
        Number of elements in the y direction, by default 10
    nz : int, optional
        Number of elements in the z direction, by default 10
    dirs : ndarray, optional
        An array with the directions of the loads, by default empty array. [[0,1,0],[1,0,0],[0,0,-1]]
    positions : ndarray, optional
        An array with the positions of the loads, by default empty array. [[5,5,9], [1,1,9], [8, 8, 9]]

    Returns
    -------
    nodes : ndarray
        Array of nodes
    mats : ndarray
        Array of material properties
    els : ndarray
        Array of elements
    loads : ndarray
        Array of loads
    idx_BC : ndarray
        Indices of nodes on the constrained face
    """
    # Generate 3D grid of nodes
    x = np.linspace(-L/2, L/2, nx + 1)
    y = np.linspace(-H/2, H/2, ny + 1)
    z = np.linspace(0, W, nz + 1)
    xv, yv, zv = np.meshgrid(x, y, z, indexing='ij')
    
    nodes = np.zeros(((nx + 1) * (ny + 1) * (nz + 1), 7))  # Added columns for BCs in X, Y, Z
    nodes[:, 0] = np.arange(nodes.shape[0])  # Node IDs
    nodes[:, 1] = xv.ravel()
    nodes[:, 2] = yv.ravel()
    nodes[:, 3] = zv.ravel()

    # Create elements (hexahedral)
    els = []
    count = 0
    for i in range(nx):
        for j in range(ny):
            for k in range(nz):
                n0 = k + (nz + 1) * (j + (ny + 1) * i)
                n1 = n0 + 1
                n2 = n0 + (nz + 1)
                n3 = n2 + 1
                n4 = n0 + (nz + 1) * (ny + 1)
                n5 = n4 + 1
                n6 = n4 + (nz + 1)
                n7 = n6 + 1
                els.append([count, 9, 0, n0, n2, n3, n1, n4, n6, n7, n5])
                count += 1
    els = np.array(els, dtype=int)

    # Assign material properties to elements
    mats = np.zeros((els.shape[0], 3))
    mats[:, 0] = E  # Young's modulus
    mats[:, 1] = v  # Poisson's ratio
    mats[:, 2] = 1  # Placeholder property

    # Apply constraints on one face (z = 0)
    mask_z0 = nodes[:, 3] == 0  # Nodes where z = 0
    nodes[mask_z0, 4] = -1  # BC in X direction
    nodes[mask_z0, 5] = -1  # BC in Y direction
    nodes[mask_z0, 6] = -1  # BC in Z direction

    idx_BC = nodes[mask_z0, 0].astype(int)

    # Define loads on the opposite face (z = W)
    loads = np.zeros((dirs.shape[0], 4))
    mask_load = nodes[:, 3] == W  # Nodes where z = W
    node_indices = np.where(mask_load)[0]
    node_indices_selected = node_indices[positions[:, 0] * (ny + 1) + positions[:, 1]]
    
    loads[:, 0] = node_indices_selected  # Node indices for loads
    loads[:, 1] = dirs[:, 0]             # X-direction load
    loads[:, 2] = dirs[:, 1]             # Y-direction load
    loads[:, 3] = dirs[:, 2]             # Z-direction load

    return nodes, mats, els, loads, idx_BC

