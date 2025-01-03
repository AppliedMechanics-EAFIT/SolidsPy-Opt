Bi-directional ESO (BESO) Tutorial
==================================

**Bi-directional Evolutionary Structural Optimization (BESO)** refines the standard ESO concept by not only removing material in low-stress/sensitivity regions, but also allowing selective *adding* of material in some steps.

Process
-------

1. **Sensitivity analysis**: Evaluate element-level sensitivity (stress, strain energy, etc.).
2. **Element removal**: Remove the lowest-sensitivity elements (like ESO).
3. **Element addition**: Potentially restore some elements in high-sensitivity areas if the design is too coarse or if a certain criterion is triggered.
4. **Iterations**: Repeat until final volume fraction or iteration limit is achieved.

Minimal Example
--------------

Below is a 2D usage snippet of the **`BESO`** function:

.. code-block:: python

   import numpy as np
   from solidspy_opt.optimize import BESO
   from solidspy_opt.utils import structures

   # A small 2D example
   L, H = 8, 8
   nx, ny = 8, 8
   dirs = np.array([[0, -1]])  
   positions = np.array([[4, 1]])  
   nodes, mats, els, loads, idx_BC = structures(L, H, nx, ny, dirs, positions, n=1)

   ELS_opt, nodes_opt, UC, E_nodes, S_nodes = BESO(
       nodes=nodes,
       els=els,
       mats=mats,
       loads=loads,
       idx_BC=idx_BC,
       niter=2,
       t=0.0001,       # target stress ratio
       ER=0.001,       # ratio increment
       volfrac=0.5,
       plot=False,
       dim_problem=2,
       nnodes=4
   )

Explanation of Parameters
-------------------------

- **`t`** (Float): The *target stress ratio* for adding or removing elements. 
- **`ER`**: Increment of that ratio each iteration.
- **`volfrac`**: Target volume fraction. BESO attempts to keep or converge near this fraction, possibly toggling add/remove to refine the shape.
- **`niter`**: Maximum number of iterations.

Further Reading
--------------

- Huang, Xie (2009) for BESO fundamentals.
- `examples/beso.ipynb` or `docs/advanced/membrane_vibration.rst` references if you have advanced examples.
