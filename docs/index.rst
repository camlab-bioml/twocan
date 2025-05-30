Twocan Documentation
====================

*A Bayesian optimization framework for multimodal registration of highly multiplexed single-cell spatial proteomics data*

.. image:: https://github.com/user-attachments/assets/1cad2a1e-ca87-474e-96de-fd6b02560771
   :width: 100px
   :align: center

|

Twocan is a Python package that uses Bayesian optimization (via Optuna) to automatically find optimal parameters for registering images from different spatial proteomics technologies like IF (Immunofluorescence), IMC (Imaging Mass Cytometry), FISH, and IMS.

.. toctree::
   :maxdepth: 2
   :caption: Getting Started

   installation
   quickstart
   concepts

.. toctree::
   :maxdepth: 2
   :caption: Tutorials

   notebooks/01_basic_registration

.. toctree::
   :maxdepth: 2
   :caption: API Reference

   api/index

Key Features
------------

* **Automated parameter optimization** using Bayesian optimization
* **Multiple imaging modalities** support (IF, IMC, FISH, IMS)
* **Scikit-learn compatible** API for easy integration
* **Flexible preprocessing** with modality-specific processors
* **Quality metrics** for registration assessment
* **Extensible design** for custom objectives and preprocessors

Quick Example
-------------

.. code-block:: python

   import optuna
   from twocan import IFProcessor, IMCProcessor, iou_corr_single_objective
   from spatialdata import read_zarr

   # Load your spatial data
   images = read_zarr('path/to/your/data.zarr')
   
   # Setup optimization
   study = optuna.create_study(direction='maximize')
   
   # Register images
   study.optimize(
       lambda trial: iou_corr_single_objective(
           trial, images, 
           registration_channels=['DAPI', 'DNA1', 'DNA2'],
           moving_image='IMC', 
           static_image='IF'
       ),
       n_trials=50
   )

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

