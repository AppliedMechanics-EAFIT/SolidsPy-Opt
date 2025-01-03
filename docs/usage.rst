Usage
=====

This page guides you through a basic **SolidsPy-Opt** workflow: creating or loading a mesh, defining materials and loads, and then running a chosen optimization algorithm.

Basics
------

**SolidsPy-Opt** is designed to build on top of the original SolidSpy approach, but integrates topology optimization methods. In general, you will:

1. **Import** the needed modules (e.g., from ``solidspy_opt.optimize``).
2. **Create or load** a mesh: by using the utility functions in ``solidspy_opt.utils`` or by reading external files (e.g., via meshio).
3. **Define material properties** and boundary/loading conditions.
4. **Call** an optimization function (e.g., **`ESO_stress`**, **`ESO_stiff`**, **`BESO`**, or **`SIMP`**).
5. **Process or visualize** the results.

A Simple 2D Example
-------------------

Below is an outline for a **2D** topology optimization using **`ESO_stress`**. This is a direct script version of what you might run in a Jupyter notebook.

.. code-block:: python

   import numpy as np
   import matplotlib.pyplot as plt

   from solidspy_opt.optimize import ESO_stress
   from solidspy_opt.utils import structures

   # 1. Set up geometry and loading
   L, H = 60, 60          # domain dimensions
   nx, ny = 60, 60        # number of divisions in X and Y
   dirs = np.array([[0, -1]])      # direction: downward load
   positions = np.array([[15, 1]]) # near the top-left corner
   n = 1  # single load

   # 2. Build the mesh and boundary conditions
   nodes, mats, els, loads, idx_BC = structures(
       L=L, H=H, nx=nx, ny=ny,
       dirs=dirs, positions=positions, n=n
   )

   # 3. Run the ESO_stress optimization
   ELS_opt, nodes_opt, UC, E_nodes, S_nodes = ESO_stress(
       nodes=nodes,
       els=els,
       mats=mats,
       loads=loads,
       idx_BC=idx_BC,
       niter=200,           # maximum iterations
       RR=0.001,           # initial removal ratio
       ER=0.01,            # increment for removal ratio each iteration
       volfrac=0.5,        # target volume fraction
       plot=True,          # enable plotting via SolidSpy
       dim_problem=2,      # 2D
       nnodes=4            # 4-node quadrilaterals
   )

   # 4. The optimization function returns:
   #    ELS_opt: the optimized elements array
   #    nodes_opt: the updated nodes array
   #    UC: full nodal displacements
   #    E_nodes, S_nodes: strain and stress at the nodes

   plt.show()

Interpreting Results
--------------------

When you enable ``plot=True``, **SolidsPy-Opt** uses *SolidSpy* postprocessing routines to visualize:

- **Deformed Mesh**: showing nodal displacements.
- **Stress Fields**: color contour of von Mises or principal stresses in the mesh.
- **Element Removal**: final shape after removing or adding elements (depending on the optimization).

You can retrieve numeric data directly from the returned arrays:

- **`ELS_opt`**: a 2D array with remaining elements and their connectivity.
- **`nodes_opt`**: the updated list of nodes and boundary flags after removal of orphaned nodes.
- **`UC`**: an array of nodal displacements that you can further post-process or plot.

Advanced 3D Usage
-----------------

For 3D, you can follow a similar pattern but use:

- **`structure_3d`** from ``solidspy_opt.utils`` to create a cubic or rectangular 3D domain.
- **`dim_problem=3`** and **`nnodes=4`** (tetra) or **`nnodes=8`** (brick), as supported by your chosen routine.

For example:

.. code-block:: python

   from solidspy_opt.utils import structure_3d
   from solidspy_opt.optimize import ESO_stress

   load_dirs = np.array([[0, 0, -1]])  # downward load in Z
   load_pos  = np.array([[5, 5, 9]])   # near top center (for a 10x10x10 domain)

   nodes_3d, mats_3d, els_3d, loads_3d, idx_BC_3d = structure_3d(
       L=10, H=10, W=10, E=210e9, v=0.3, nx=10, ny=10, nz=10,
       dirs=load_dirs, positions=load_pos
   )

   ELS_opt_3d, nodes_opt_3d, UC_3d, E_nodes_3d, S_nodes_3d = ESO_stress(
       nodes=nodes_3d,
       els=els_3d,
       mats=mats_3d,
       loads=loads_3d,
       idx_BC=idx_BC_3d,
       niter=100,
       RR=0.005,
       ER=0.02,
       volfrac=0.5,
       plot=False,   # disable plotting if you prefer
       dim_problem=3,
       nnodes=8
   )

Switching Between Optimization Methods
-------------------------------------

If you want to switch from **`ESO_stress`** to, for example, **`BESO`** or **`SIMP`**, you only need to change the import and function call:

.. code-block:: python

   # from solidspy_opt.optimize import BESO
   # ...
   # ELS_opt, nodes_opt, UC, E_nodes, S_nodes = BESO(
   #    nodes=..., els=..., ...
   # )

Be mindful that each method has slightly different parameters (e.g., `t` and `ER` for BESO, or `penal` for SIMP).

Integration with External Meshes
-------------------------------

Though you can rely on **`structures()`** or **`structure_3d()`** to create simple rectangular/square meshes, you can also load meshes from Gmsh or other external generators:

1. Install **meshio** (already in dependencies) to handle reading `.msh` or `.xml` formats, etc.
2. Convert that mesh into the format required by SolidSpy / SolidsPy-Opt (a combination of nodal coordinates, element connectivity, boundary conditions, etc.).

For advanced geometry or boundary definitions, see the official SolidSpy or meshio documentation.

Summary
-------

This **Usage** guide has walked you through:
- Building or loading a mesh via **`structures()`** or **`structure_3d()`**.
- Calling the various optimization routines in **SolidsPy-Opt** (e.g., **`ESO_stress`**).
- Retrieving and plotting the optimized shape and solution fields.

Explore the :ref:`tutorials <tutorials>` section for deeper examples and step-by-step instructions on each solver (ESO, BESO, SIMP, etc.). If you run into trouble, open an issue on `GitHub <https://github.com/AppliedMechanics-EAFIT/SolidsPy-Opt/issues>`_. 
