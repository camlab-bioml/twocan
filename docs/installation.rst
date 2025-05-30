Installation
============

Requirements
------------

Twocan requires Python 3.8 or later and has the following dependencies:

* numpy
* pandas
* opencv-python
* scikit-image
* scikit-learn
* spatialdata
* optuna
* tifffile

Installing from Source
----------------------

.. code-block:: bash

   # Clone the repository
   git clone https://github.com/camlab-bioml/twocan.git
   cd twocan

   # Create and activate conda environment
   conda env create -f environment.yml
   conda activate twocan

   # Install the package
   pip install .

Installing from PyPI
---------------------

.. note::
   PyPI installation will be available in future releases.

.. code-block:: bash

   pip install twocan

Development Installation
------------------------

For development, install in editable mode with development dependencies:

.. code-block:: bash

   # Clone and enter directory
   git clone https://github.com/camlab-bioml/twocan.git
   cd twocan

   # Create conda environment
   conda env create -f environment.yml
   conda activate twocan

   # Install in development mode
   pip install -e .

   # Install additional development dependencies
   pip install pytest sphinx nbsphinx myst-nb sphinx-rtd-theme

Verification
------------

To verify your installation, run:

.. code-block:: python

   import twocan
   print(twocan.__version__)

   # Test basic functionality
   from twocan import RegEstimator, IFProcessor, IMCProcessor
   print("Twocan installed successfully!")
