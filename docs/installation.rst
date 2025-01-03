Installation
============

This page describes how to install **SolidsPy-Opt** and its dependencies.


Basic Installation
------------------

If you already have Python 3.8+ and the standard scientific stack (Numpy, Scipy, etc.) installed, you can install **SolidsPy-Opt** from the Python Package Index (PyPI) with:

.. code-block:: bash

   pip install solidspy-opt

This will download and install the package along with any missing dependencies.

Alternatively, you can install **SolidsPy-Opt** into a **virtual environment** (via `venv` or `conda` environments) to keep dependencies isolated:

.. code-block:: bash

   python -m venv venv
   source venv/bin/activate
   pip install --upgrade pip
   pip install solidspy-opt


Optional Dependencies
---------------------

**SolidsPy-Opt** relies on some key packages for full functionality:

- **`numpy`** (Array computations)
- **`scipy`** (Sparse linear algebra, solver utilities)
- **`matplotlib`** (Plotting, visualizations)
- **`meshio`** (Reading/writing Gmsh and other mesh formats, pinned at version 5.3.5)
- **`tensorflow`** (Potential usage in advanced or neural-based workflows)
- **`easygui`** (GUI usage in some interactive features)

Some of these are optional, but recommended for a complete experience. By default, installing from PyPI attempts to install them if they are not present.

If you need to generate complex geometries or 3D meshes, ensure that **`meshio`** is installed:

.. code-block:: bash

   pip install meshio==5.3.5


Development Installation
------------------------

If you want the **latest** code or to contribute, clone the GitHub repository and install in editable mode:

.. code-block:: bash

   git clone https://github.com/AppliedMechanics-EAFIT/SolidsPy-Opt.git
   cd SolidsPy-Opt
   pip install --upgrade pip setuptools wheel
   pip install -e .

This will link your local copy so that any changes you make are immediately reflected in the installed package.


Installing from Test PyPI
-------------------------

If you have a **test** version of **SolidsPy-Opt** or a custom fork (e.g., ``SolidsPyKevin``) published on Test PyPI, you can install it with:

.. code-block:: bash

   pip install --index-url https://test.pypi.org/simple/ \
               --extra-index-url https://pypi.org/simple \
               SolidsPyKevin

This ensures that non-Test-PyPI dependencies can still be pulled from the main PyPI index.


Verifying Installation
----------------------

After installing, open a Python shell or Jupyter notebook and try:

.. code-block:: python

   import solidspy_opt
   from solidspy_opt.optimize import ESO_stress

   print("SolidsPy-Opt installed successfully!")

If no error is raised, your installation is ready.


Troubleshooting
---------------

- **Missing Dependencies**: Ensure that you have the correct versions of `numpy`, `scipy`, and other core libraries installed.  
- **Path Issues**: If you are using a virtual environment, confirm that you have **activated** it before installing.  
- **Conflicts**: Old versions of packages can lead to conflicts. Try `pip install --upgrade <package>` to update them.  

If you encounter any issue, please check the `GitHub Issues <https://github.com/AppliedMechanics-EAFIT/SolidsPy-Opt/issues>`_ for existing reports or open a new issue describing your problem. 
