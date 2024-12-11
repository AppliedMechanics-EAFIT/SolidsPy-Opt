import numpy as np

def calculate_mesh_area(nodes: np.ndarray, elements: np.ndarray) -> float:
    """
    Calculate the total area of a 2D square mesh.

    Parameters
    ----------
    nodes : ndarray
        Array of node coordinates with shape (N, 3), where each row is [node_id, x, y].
    elements : ndarray
        Array of elements with shape (M, 5), where each row is [element_id, node1, node2, node3, node4].

    Returns
    -------
    total_area : float
        Total area of the mesh.
    """
    total_area = 0.0

    for element in elements:
        # Extract node indices
        node_ids = element[1:]  # Skip element ID
        # Get coordinates of the nodes
        coords = nodes[np.isin(nodes[:, 0], node_ids), 1:3]  # Select columns [x, y]
        
        # Calculate side length (assumes square element)
        side_length = np.linalg.norm(coords[1] - coords[0])  # Distance between two adjacent vertices
        
        # Area of the square element
        area = side_length ** 2
        total_area += area

    return total_area
