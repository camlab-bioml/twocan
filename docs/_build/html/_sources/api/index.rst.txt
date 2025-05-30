API Reference
=============

This section provides detailed documentation for all Twocan classes and functions.

Core Classes
------------

.. autosummary::
   :toctree: generated/
   :template: class.rst

   twocan.RegEstimator

Preprocessors
-------------

.. autosummary::
   :toctree: generated/
   :template: class.rst

   twocan.IFProcessor
   twocan.IMCProcessor

Optimization Functions
----------------------

.. autosummary::
   :toctree: generated/
   :template: function.rst

   twocan.iou_corr_single_objective
   twocan.iou_corr_multi_objective
   twocan.registration_trial

Callbacks
---------

.. autosummary::
   :toctree: generated/
   :template: class.rst

   twocan.SaveTrialsDFCallback
   twocan.ThresholdReachedCallback
   twocan.MatrixConvergenceCallback

Utilities
---------

.. autosummary::
   :toctree: generated/
   :template: function.rst

   twocan.stretch_255
   twocan.read_M
   twocan.multi_channel_corr
   twocan.get_aligned_coordinates
   twocan.prep_zarr
   twocan.pick_best_registration

Plotting
--------

.. autosummary::
   :toctree: generated/
   :template: function.rst

   twocan.plot_cartoon_affine
   twocan.get_merge

Complete API
------------

.. automodule:: twocan
   :members:
   :undoc-members:
   :show-inheritance: 