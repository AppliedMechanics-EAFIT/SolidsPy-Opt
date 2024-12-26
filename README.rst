SolidsPy-Opt: 2D Topology Optimization with Python
==================================================

.. figure:: https://raw.githubusercontent.com/AppliedMechanics-EAFIT/SolidsPy/master/docs/img/wrench.png
   :alt: Wrench under bending.

.. image:: https://img.shields.io/pypi/v/solidspy.svg
   :target: https://pypi.python.org/pypi/solidspy-opt
   :alt: PyPI download

.. image:: https://readthedocs.org/projects/solidspy-opt/badge/?version=latest
   :target: https://solidspy-opt.readthedocs.io/en/latest/
   :alt: Documentation Status

.. image:: https://img.shields.io/pypi/dm/solidspy
   :target: https://pypistats.org/packages/solidspy-opt
   :alt: Downloads frequency

A simple finite element analysis (FEA) and topology optimization code for
2D elasticity problems. **SolidsPy-Opt** uses as input easy-to-create text
files (or Python data structures) defining a model in terms of nodal,
element, material, and load data. It extends or modifies functionality
from the original `SolidsPy <https://github.com/AppliedMechanics-EAFIT/SolidsPy>`__ 
to support topology optimization workflows.

- Documentation: http://solidspy-opt.readthedocs.io
- GitHub: https://github.com/AppliedMechanics-EAFIT/SolidsPy-Opt
- PyPI: https://pypi.org/project/solidspy-opt/
- Free and open source software: `MIT license <http://en.wikipedia.org/wiki/MIT_License>`__


Features
--------

* Built upon an open-source Python ecosystem.

* Easy to use and modify for **topology optimization** tasks.

* Handles 2D elasticity, displacement, strain, and stress solutions, and 
  extends these workflows to include optimization of material layout
  (topology optimization).

* Organized in independent modules for pre-processing, assembly, optimization,
  and post-processing, allowing the user to easily modify or add new 
  features (e.g., new elements or custom optimization strategies).

* Created with academic and research goals in mind.

* Can be used to teach or illustrate:
  
  - Computational Modeling
  - Finite Element Methods
  - Topology Optimization
  - Other advanced engineering topics


Installation
------------

The code is written in Python, depending on ``numpy``, ``scipy``, and
``sympy`` (and possibly other libraries if your optimization approach
requires them). It has been tested under Windows, Mac, Linux, and Android 
environments.

To install *SolidsPy-Opt*, open a terminal and type:

::

    pip install solidspy-opt

If you also need an interactive GUI for file selection, you can install
`easygui <http://easygui.readthedocs.org/en/master/>`__:

::

    pip install easygui

For generating mesh files from
`Gmsh <http://gmsh.info/>`__, install
`meshio <https://github.com/nschloe/meshio>`__:

::

    pip install meshio


How to run a simple model
-------------------------

Below is a minimal example showing how you might set up and run a
2D finite element + topology optimization analysis in *SolidsPy-Opt*.
(Adapt it to your actual code structure and function names.)

.. code:: python

    import numpy as np
    import matplotlib.pyplot as plt
    # Hypothetical module import for SolidsPy-Opt:
    from solidspy_opt.solids_GUI import solids_opt_auto

    # Define the data (nodes, constraints, elements, materials, loads, etc.)
    nodes = np.array([
        [0, 0.00, 0.00],
        [1, 2.00, 0.00],
        [2, 2.00, 2.00],
        [3, 0.00, 2.00],
        [4, 1.00, 0.00],
        [5, 2.00, 1.00],
        [6, 1.00, 2.00],
        [7, 0.00, 1.00],
        [8, 1.00, 1.00]])

    cons = np.array([
        [0, -1],
        [0, -1],
        [0,  0],
        [0,  0],
        [-1, -1],
        [0,  0],
        [0,  0],
        [0,  0],
        [0,  0]])

    elements = np.array([
        [0, 1, 0, 0, 4, 8, 7],
        [1, 1, 0, 4, 1, 5, 8],
        [2, 1, 0, 7, 8, 6, 3],
        [3, 1, 0, 8, 5, 2, 6]])

    mats = np.array([[1.0, 0.3]])

    loads = np.array([
        [2, 0.0, 1.0],
        [3, 0.0, 1.0],
        [6, 0.0, 2.0]])

    data = {
        "nodes": nodes,
        "cons": cons,
        "elements": elements,
        "mats": mats,
        "loads": loads
        # Potentially additional data for optimization:
        # "vol_frac": 0.5,
        # "penal": 3.0,
        # "filter_radius": 1.2,
        # etc.
    }

    # Run the simulation + topology optimization
    disp, topo_density = solids_opt_auto(data)

    # Plot results
    plt.figure()
    # Hypothetical function that plots the density distribution
    plt.imshow(topo_density.reshape(2,2))  
    plt.title("Topology Density")
    plt.colorbar()
    plt.show()


Save the script (for example, as ``example_solidspy_opt.py``) and run it:

.. code:: bash

    python example_solidspy_opt.py


License
-------

This project is licensed under the `MIT
license <http://en.wikipedia.org/wiki/MIT_License>`__. All documentation
is licensed under the `Creative Commons Attribution
License <http://creativecommons.org/licenses/by/4.0/>`__.


Citation
--------

If you use **SolidsPy-Opt** in your research or publications, please cite it.
A BibTeX entry for LaTeX users might look like:

.. code:: bibtex

    @software{solidspy_opt,
      title     = {SolidsPy-Opt: 2D-Finite Element and Topology Optimization Analysis with Python},
      author    = {Sepúlveda-García, Kevin and Guarin-Zapata, Nicolas},
      year      = 2024,
      version   = {0.1.0},
      keywords  = {finite-elements, scientific-computing, deep learning, topology, optimization},
      license   = {MIT License},
      url       = {https://github.com/AppliedMechanics-EAFIT/SolidsPy-Opt},
      abstract  = {SolidsPy-Opt is a Python package designed to perform
                   topology optimization of 2D solids by leveraging
                   finite-element methods and advanced computational tools.}
    }
