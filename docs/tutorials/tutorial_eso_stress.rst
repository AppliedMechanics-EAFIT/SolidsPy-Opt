ESO by Stress Tutorial
======================

In this tutorial, we demonstrate how to perform **Evolutionary Structural Optimization (ESO)** based on stress criteria using **SolidsPy-Opt**. This method iteratively removes elements with low relative stress until a target volume fraction is reached.

Overview
--------

The **`ESO_stress`** function uses nodal and element data to:

1. Assemble and solve the linear system.
2. Compute element von Mises stresses.
3. Remove elements below a certain stress threshold.
4. Repeat until the desired volume fraction or iteration limit is reached.

Example Code
------------

Below is a minimal **2D** usage example. You can adapt it for **3D** as well.

.. code-block:: python

   import numpy as np
   import matplotlib.pyplot as plt

   # Import the ESO_stress solver and utility functions
   from solidspy_opt.optimize import ESO_stress
   from solidspy_opt.utils import structures

   # Define geometry, mesh, and loading
   L, H = 60, 60
   nx, ny = 60, 60
   dirs = np.array([[0, -1]])           # single downward load
   positions = np.array([[15, 1]])      # near top edge
   nodes, mats, els, loads, idx_BC = structures(
       L=L, H=H,
       nx=nx, ny=ny,
       dirs=dirs,
       positions=positions,
       n=1
   )

   # Run ESO based on stress
   ELS_opt, nodes_opt, UC, E_nodes, S_nodes = ESO_stress(
       nodes=nodes,
       els=els,
       mats=mats,
       loads=loads,
       idx_BC=idx_BC,
       niter=200,     # number of iterations
       RR=0.001,      # initial removal ratio
       ER=0.01,       # removal ratio increment
       volfrac=0.5,   # target volume fraction
       plot=True,
       dim_problem=2,
       nnodes=4
   )

Description of Arguments
------------------------

- **`nodes`**: Numpy array with node coordinates and boundary conditions.
- **`els`**: Numpy array with element connectivity.
- **`mats`**: Numpy array with material properties.
- **`loads`**: Numpy array listing node loads (forces).
- **`idx_BC`**: Array of node indices for boundary conditions.
- **`niter`**: Maximum number of optimization iterations.
- **`RR`**, **`ER`**: Initial removal ratio and its increment.
- **`volfrac`**: Fraction of initial volume (area in 2D, volume in 3D) to remain after optimization.
- **`plot`**: Boolean to enable the SolidSpy-based plotting.
- **`dim_problem`**: 2 or 3 (2D or 3D).
- **`nnodes`**: Number of nodes per element (3 or 4 in 2D, 4 or 8 in 3D).

Tips & Notes
------------

- Ensure your mesh is fine enough to capture meaningful stress distributions.
- For 3D usage, call :code:`structure_3d(...)` from **`solidspy_opt.utils`**.
- Plotting relies on SolidSpy's postprocessing modules.


Further Reading
--------------

- **Xie and Steven**, 1993, on the original ESO approach.
- `ESO_stress` docstring or `examples/eso_stress.ipynb` in the **examples/** directory.
