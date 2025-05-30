Quickstart Guide
================

This guide will get you up and running with Twocan in minutes.

What is Twocan?
---------------

Twocan automatically finds the best parameters for registering images from different spatial proteomics technologies. Instead of manually tuning preprocessing and registration parameters, Twocan uses Bayesian optimization to explore the parameter space efficiently.

Basic Workflow
--------------

The typical Twocan workflow involves:

1. **Load your spatial data** (IF, IMC, FISH, etc.)
2. **Define registration channels** (usually nuclear markers)
3. **Setup Bayesian optimization** with Optuna
4. **Run optimization** to find best registration parameters
5. **Apply results** to register your full dataset

Your First Registration
-----------------------

Here's a complete example using IF and IMC data:

.. code-block:: python

   import optuna
   from twocan import IFProcessor, IMCProcessor, single_objective
   from spatialdata import read_zarr

   # 1. Load your data
   images = read_zarr('path/to/your/data.zarr')
   
   # 2. Define channels for registration (nuclear markers work best)
   registration_channels = ['DAPI', 'DNA1', 'DNA2']
   
   # 3. Setup Bayesian optimization
   study = optuna.create_study(direction='maximize', sampler=optuna.samplers.TPESampler(seed=42))
   
   # 4. Run optimization
   study.optimize(
       lambda trial: single_objective(
           trial, 
           images, 
           registration_channels,
           moving_image='IMC',    # Image to transform
           static_image='IF',     # Reference image
           moving_preprocessor=IMCProcessor(),
           static_preprocessor=IFProcessor()
       ),
       n_trials=50  # More trials = better results but slower
   )
   
   # 5. Get the best registration
   best_trial = study.best_trial
   print(f"Best registration score: {best_trial.value}")
   print(f"Best parameters: {best_trial.params}")

Understanding the Results
-------------------------

Twocan optimizes for registration quality using metrics like:

* **Intersection over Union (IoU)**: Overlap between registered images
* **Correlation**: Similarity of intensity patterns
* **Logical operations**: AND, OR, XOR of binary masks

The optimization automatically balances these metrics to find the best overall registration.

Sample Use Cases
----------------

**Same-slide co-stained registration (IF + IMC)**
   Register images from the same tissue section acquired with different technologies.

**Same-slide serial-stained registration (FISH + IMC)**
   Register images from the same tissue section acquired with different technologies.

**Serial section registration (IMS + IMC)**
   Register adjacent tissue sections with similar but not identical features.


Next Steps
----------

* Check out the :doc:`tutorials/index` for detailed walkthroughs
* Read about :doc:`concepts` to understand the methodology
* See the :doc:`api/index` for complete function documentation

Getting Help
------------

* **GitHub Issues**: Report bugs or request features

Tips for Success
----------------

1. **Choose good registration channels**: Nuclear markers (DAPI, DNA) usually work best
2. **Start with more trials**: 50-100 trials often give good results
3. **Check your data quality**: Poor image quality leads to poor registration
4. **Validate results visually**: Inspect the final registration
5. **Use appropriate preprocessing**: Different modalities need different preprocessing 