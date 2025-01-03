SIMP Tutorial
=============

The **Solid Isotropic Material with Penalization (SIMP)** method is a classic topology optimization technique that penalizes intermediate densities to encourage “0 or 1” (void or solid) solutions.

Basic Workflow
--------------

1. **Initialize** element densities (:math:`\rho_i`) to the volume fraction :math:`V_f`.
2. **Assemble and Solve** for displacements.
3. **Compute Sensitivities**: Typically compliance is used (minimize :math:`\int \sigma \epsilon`).
4. **Update Densities**: Using an **Optimality Criteria** or **Method of Moving Asymptotes** approach.
5. **Repeat** until convergence or iteration limit.

Example
-------

Here is a 2D usage snippet of **`SIMP`**:

.. code-block:: python

   import numpy as np
   from solidspy_opt.optimize import SIMP
   from solidspy_opt.utils import structures

   # 2D domain
   L, H = 60, 60
   nx, ny = 60, 60
   dirs = np.array([[0, -1]])  
   positions = np.array([[15, 1]])  
   nodes, mats, els, loads, idx_BC = structures(
       L=L, H=H, nx=nx, ny=ny,
       dirs=dirs, positions=positions, n=1
   )

   rho, nodes_opt, UC, E_nodes, S_nodes = SIMP(
       nodes=nodes,
       els=els,
       mats=mats,
       loads=loads,
       idx_BC=idx_BC,
       niter=100,
       penal=3.0,       # penalization factor
       volfrac=0.5,
       dimensions=[nx, ny],  # needed for 2D plotting if you want to reshape densities
       plot=True,
       dim_problem=2,
       nnodes=4
   )

Key Parameters
--------------

- **`penal`**: Penalization exponent (typical range 2–5).
- **`volfrac`**: Target volume fraction.
- **`dimensions`**: For 2D, pass `[nx, ny]` so the final density array can be reshaped and plotted.
- **`niter`**: Maximum iterations.

Important Notes
---------------

- SIMP typically relies on a stable update scheme (Optimality Criteria or MMA).
- For 3D, pass `dim_problem=3` and optionally `nnodes=4` (tetra) or `nnodes=8` (brick).
- The code in `SIMP` may apply additional filtering or steps to ensure stable solutions.

References
----------

- Bendsøe and Sigmund, 1999, for the original SIMP approach.
- `examples/simp.ipynb` for a comprehensive notebook demonstration.
