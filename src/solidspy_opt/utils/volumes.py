from typing import Tuple
import numpy as np
from numpy.typing import NDArray


def calculate_mesh_area(
    nodes: NDArray[np.float64],
    elements: NDArray[np.float64]
) -> float:
    """
    Calculate the total area of a 2D square mesh.

    Parameters
    ----------
    nodes : NDArray[np.float64]
        Array of node coordinates with shape (N, 3), where each row is:
        [node_id, x, y].
        - node_id : int
        - x       : float
        - y       : float
    elements : NDArray[np.float64]
        Array of elements with shape (M, 5), where each row is:
        [element_id, node1, node2, node3, node4].
        - element_id : int
        - node1      : int
        - node2      : int
        - node3      : int
        - node4      : int

    Returns
    -------
    total_area : float
        Total area of the mesh.
    """
    total_area = 0.0

    for element in elements:
        # Extract node indices (skip element ID)
        node_ids = element[1:].astype(int)
        # Get coordinates of the nodes (select columns [x, y])
        coords = nodes[np.isin(nodes[:, 0], node_ids), 1:3]

        # Calculate side length assuming a square element
        side_length = np.linalg.norm(coords[1] - coords[0])
        area = side_length ** 2
        total_area += area

    return total_area


def calculate_mesh_volume(
    nodes: NDArray[np.float64],
    els: NDArray[np.float64]
) -> float:
    """
    Calculate the total volume of a 3D mesh composed of hexahedral elements.

    Parameters
    ----------
    nodes : NDArray[np.float64]
        Array of node coordinates with the format:
        [node_id, x, y, z, BC].
        Shape: (N, 5)
        - node_id : int
        - x       : float
        - y       : float
        - z       : float
        - BC      : int or float (boundary condition info)
    els : NDArray[np.float64]
        Array of elements with the format:
        [element_id, material_id, node1, node2, ..., node8].
        Shape: (M, 10)
        - element_id   : int
        - material_id  : int (or float, depending on convention)
        - node1..node8 : int

    Returns
    -------
    total_volume : float
        Total volume of the mesh.
    """

    def hexahedron_volume(node_coords: NDArray[np.float64]) -> float:
        """
        Calculate the volume of a hexahedron given its node coordinates.

        Parameters
        ----------
        node_coords : NDArray[np.float64]
            Shape: (8, 3). Coordinates [x, y, z] of the 8 nodes.

        Returns
        -------
        volume : float
            Volume of the hexahedron (estimated by splitting into tetrahedra).
        """
        # Unpack node coordinates
        p0, p1, p2, p3, p4, p5, p6, p7 = node_coords

        # First tetrahedron volume: p0, p1, p3, p4
        v1 = abs(np.dot(np.cross(p1 - p0, p3 - p0), p4 - p0)) / 6.0

        # Second tetrahedron volume: p1, p4, p5, p6
        v2 = abs(np.dot(np.cross(p4 - p1, p5 - p1), p6 - p1)) / 6.0

        return v1 + v2

    total_volume = 0.0
    for el in els:
        # Extract node numbers for the element (skip element ID and material)
        node_indices = el[3:].astype(int)
        # Get coordinates of the 8 nodes
        node_coords = nodes[node_indices, 1:4]  # columns 1:4 -> x, y, z
        total_volume += hexahedron_volume(node_coords)

    return total_volume
