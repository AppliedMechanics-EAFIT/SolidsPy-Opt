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

def calculate_mesh_volume(nodes, els):
    """
    Calculate the total volume of a 3D mesh composed of hexahedral elements.

    Parameters:
    - nodes : ndarray
        Array of node coordinates with the format 
        [node number, x, y, z, BC], where BC indicates boundary conditions.
    - els : ndarray
        Array of elements with the format [element number, material, node1, node2, ...].

    Returns:
    - total_volume : float
        Total volume of the mesh.
    """
    def hexahedron_volume(node_coords):
        """
        Calculate the volume of a hexahedron given its node coordinates.
        
        Parameters:
        - node_coords : ndarray of shape (8, 3)
            Coordinates of the 8 nodes of the hexahedron.
        
        Returns:
        - volume : float
            Volume of the hexahedron.
        """
        # Split into two tetrahedra for a rough volume estimate
        p0, p1, p2, p3, p4, p5, p6, p7 = node_coords[:,]

        # First tetrahedron volume: p0, p1, p3, p4
        v1 = np.abs(np.dot(np.cross(p1 - p0, p3 - p0), p4 - p0)) / 6.0

        # Second tetrahedron volume: p1, p4, p5, p6
        v2 = np.abs(np.dot(np.cross(p4 - p1, p5 - p1), p6 - p1)) / 6.0

        # Return the sum of tetrahedral volumes as the hexahedron volume
        return v1 + v2

    total_volume = 0.0
    for el in els:
        # Extract node numbers for the element
        node_indices = el[3:]  # Skip element number and material
        # Get coordinates of the nodes for this element
        node_coords = nodes[node_indices, 1:4]  # Subtract 1 for zero-based indexing
        # Add the hexahedron volume to the total
        total_volume += hexahedron_volume(node_coords)
    
    return total_volume
