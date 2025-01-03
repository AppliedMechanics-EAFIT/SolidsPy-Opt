ESO by Stiffness Tutorial
=========================

**Evolutionary Structural Optimization (ESO)** can also be driven by *stiffness-based* criteria, removing elements with the lowest “sensitivity” or “stiffness contribution.”

Function Overview
-----------------

**`ESO_stiff`** follows a similar flow to **`ESO_stress`**, but instead of the von Mises stress for removal, it evaluates an element stiffness (or strain energy) sensitivity criterion.

Quick Example
------------

.. code-block:: python

   import numpy as np
   from solidspy_opt.optimize import ESO_stiff
   from solidspy_opt.utils import structures

   # 2D Example
   L, H = 8, 8
   nx, ny = 8, 8
   dirs = np.array([[0, -1]])  # downward load
   positions = np.array([[4, 1]])
   nodes, mats, els, loads, idx_BC = structures(
       L=L, H=H, nx=nx, ny=ny, dirs=dirs, positions=positions, n=1
   )

   ELS_opt, nodes_opt, UC, E_nodes, S_nodes = ESO_stiff(
       nodes=nodes,
       els=els,
       mats=mats,
       loads=loads,
       idx_BC=idx_BC,
       niter=2,
       RR=0.1,
       ER=0.1,
       volfrac=0.5,
       plot=False,
       dim_problem=2,
       nnodes=4
   )

Explanation of Key Steps
------------------------

1. **System Assembly and Solution**: As with other approaches, it forms and solves :math:`K \, u = f`.
2. **Sensitivity Calculation**: Each element's stiffness contribution or strain-energy-based sensitivity is computed.
3. **Element Removal**: Elements with the lowest sensitivity are removed.
4. **Re-Meshing**: The algorithm may remove nodes if they become orphaned (no longer connected to any element).
5. **Iteration**: The process repeats until the desired volume fraction is reached or iteration count is exceeded.

Parameter Notes
--------------

- **`RR`** (Removal Ratio): Threshold below which elements are removed.
- **`ER`** (Removal Ratio Increment): Increases the threshold each iteration.
- The rest of the parameters are akin to **`ESO_stress`** (see :ref:`tutorial_eso_stress`).

See Also
--------

- `examples/eso_stiff.ipynb` for a more advanced demonstration.
- Xie and Steven, 1993, for original ESO references.
