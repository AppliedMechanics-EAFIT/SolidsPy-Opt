# SolidsPy: 2D/3D-Finite Element Analysis with Python

| ![Animation 1](assets/anim1.gif) | ![Animation 2](assets/anim2.gif) |
|----------------------------------|----------------------------------|

[![PyPI version](https://img.shields.io/pypi/v/solidspy.svg)](https://pypi.python.org/pypi/solidspy-opt)
[![Documentation Status](https://readthedocs.org/projects/solidspy-opt/badge/?version=latest)](https://solidspy-opt.readthedocs.io/en/latest/)
[![Downloads frequency](https://img.shields.io/pypi/dm/solidspy)](https://pypistats.org/packages/solidspy-opt)

A simple topology optimization code for 2D/3D elasticity problems. **SolidsPy-Opt** uses as input easy-to-create text files (or Python data structures) defining a model in terms of nodal, element, material, and load data. It extends or modifies functionality from the original [SolidsPy](https://github.com/AppliedMechanics-EAFIT/SolidsPy) to support topology optimization workflows.

- Documentation: http://solidspy-opt.readthedocs.io
- GitHub: https://github.com/AppliedMechanics-EAFIT/SolidsPy-Opt
- PyPI: https://pypi.org/project/solidspy-opt/
- Free and open source software: [MIT license](http://en.wikipedia.org/wiki/MIT_License)

## Features

* Built upon an open-source Python ecosystem.
* Easy to use and modify for **topology optimization** tasks.
* Extends SolidsPy features to include optimization of material layout (topology optimization).
* Created with academic and research goals in mind.
* Can be used to teach or illustrate:
  - Computational Modeling
  - Topology Optimization
  - Other advanced engineering topics

## Installation

The code is written in Python, depending on `numpy`, `scipy`, and `SolidsPy`. It has been tested under Windows, Mac and Linux.

To install *SolidsPy-Opt*, open a terminal and type:

```bash
pip install solidspy-opt
```

For generating mesh files from [Gmsh](http://gmsh.info/), install [meshio](https://github.com/nschloe/meshio):

```bash
pip install meshio
```

## How to run a simple model

Below is a minimal example showing how you might set up and run a 2D topology optimization analysis in *SolidsPy-Opt*.

```python
import numpy as np
import matplotlib.pyplot as plt

from solidspy_opt.optimize import ESO_stress
from solidspy_opt.utils import structure_3d, structures

# Define the load directions and positions on the top face
load_directions_3d = np.array([
    [0, 1, 0],    # Load in the Y direction
    [1, 0, 0],    # Load in the X direction
    [0, 0, -1]    # Load in the negative Z direction
])
load_positions_3d = np.array([
    [5, 5, 9],    # Position near the center of the top face
    [1, 1, 9],    # Position near one corner of the top face
    [8, 8, 9]     # Position near another corner of the top face
])

# Generate the nodes, materials, elements, loads, and BC indexes
nodes_3d, mats_3d, els_3d, loads_3d, idx_BC_3d = structure_3d(
    L=10,       # length in X
    H=10,       # length in Y
    W=10,       # length in Z
    E=206.8e9,  # Young's modulus
    v=0.28,     # Poisson's ratio
    nx=10,      # number of divisions in X
    ny=10,      # number of divisions in Y
    nz=10,      # number of divisions in Z
    dirs=load_directions_3d,
    positions=load_positions_3d
)

# Run the ESO optimization
els_opt_3d, nodes_opt_3d, UC_3d, E_nodes_3d, S_nodes_3d = ESO_stress(
    nodes=nodes_3d,
    els=els_3d,
    mats=mats_3d,
    loads=loads_3d,
    idx_BC=idx_BC_3d,
    niter=200,
    RR=0.005,      # Initial removal ratio
    ER=0.05,       # Removal ratio increment
    volfrac=0.5,   # Target volume fraction
    plot=True,     # Whether to plot with solidspy's 3D plot function
    dim_problem=3,
    nnodes=8       # 8-node hexahedron
)
```

Save the script (for example, as `example_solidspy_opt.py`) and run it:

```bash
python example_solidspy_opt.py
```

## License

This project is licensed under the [MIT license](http://en.wikipedia.org/wiki/MIT_License). All documentation is licensed under the [Creative Commons Attribution License](http://creativecommons.org/licenses/by/4.0/).

## Citation

If you use **SolidsPy-Opt** in your research or publications, please cite it. A BibTeX entry for LaTeX users might look like:

```bibtex
@software{solidspy_opt,
  title     = {SolidsPy-Opt: 2D/3D-Finite Element and Topology Optimization Analysis with Python},
  author    = {Sepúlveda-García, Kevin and Guarin-Zapata, Nicolas},
  year      = 2025,
  version   = {0.1.0},
  keywords  = {finite-elements, scientific-computing, deep learning, topology, optimization},
  license   = {MIT License},
  url       = {https://github.com/AppliedMechanics-EAFIT/SolidsPy-Opt},
  abstract  = {SolidsPy-Opt is a Python package designed to perform
               topology optimization of 2D/3D solids by leveraging SolidsPy
               finite-element package and advanced computational tools.}
}
```