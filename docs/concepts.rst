Core Concepts
=============

This section explains the key concepts behind Twocan's approach to multimodal image registration.

What is Multimodal Image Registration?
---------------------------------------

Multimodal image registration is the process of aligning images acquired using different technologies or at different times. In spatial proteomics, this might involve:

* **IF (Immunofluorescence)** + **IMC (Imaging Mass Cytometry)** from the same tissue section
* **FISH (Fluorescence in situ hybridization)** cycles from sequential rounds
* **IMS (Ion Mobility Spectrometry)** + **IMC** from serial sections

The challenge is that different technologies have different:

* **Resolution**: Pixel sizes and image dimensions
* **Signal characteristics**: Intensity distributions, noise patterns
* **Channel availability**: Different markers and detection methods

Bayesian Optimization Approach
-------------------------------

Traditional registration methods require manual parameter tuning, which is:

* **Time-consuming**: Many parameters to optimize
* **Subjective**: Hard to define "good enough" registration
* **Technology-specific**: Different modalities need different approaches

Twocan uses **Bayesian optimization** (via Optuna) to automatically:

1. **Explore parameter space** efficiently using probabilistic models
2. **Balance multiple objectives** (overlap, correlation, feature matching)
3. **Converge quickly** to optimal solutions
4. **Provide uncertainty estimates** for registration quality

The Registration Pipeline
-------------------------

Twocan's registration pipeline consists of several stages:

Preprocessing
~~~~~~~~~~~~~

Each imaging modality requires specific preprocessing:

**IF Images:**
   * Scaling for resolution matching
   * Gaussian blurring for noise reduction
   * Binarization for feature extraction

**IMC Images:**
   * Arcsinh transformation for variance stabilization
   * Winsorization for outlier handling
   * Gaussian blurring and binarization

**Custom Processors:**
   You can define custom preprocessing for other modalities.

Feature Detection and Matching
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Twocan uses OpenCV's ORB (Oriented FAST and Rotated BRIEF) detector to:

1. **Detect keypoints** in both preprocessed images
2. **Extract descriptors** that are invariant to rotation and scale
3. **Match features** between images using Hamming distance
4. **Filter matches** keeping only the most confident ones

Transformation Estimation
~~~~~~~~~~~~~~~~~~~~~~~~~

From the matched features, Twocan estimates an **affine transformation**:

.. math::

   \begin{bmatrix} x' \\ y' \end{bmatrix} = \begin{bmatrix} a & b \\ c & d \end{bmatrix} \begin{bmatrix} x \\ y \end{bmatrix} + \begin{bmatrix} t_x \\ t_y \end{bmatrix}

This transformation handles:

* **Translation**: Moving the image
* **Rotation**: Rotating the image
* **Scaling**: Uniform scaling
* **Shearing**: Limited non-uniform deformation

Quality Assessment
~~~~~~~~~~~~~~~~~~

Registration quality can be assessed using multiple metrics:

**Geometric Overlap:**
   * IoU (Intersection over Union)
   * Logical AND, OR, XOR operations
   * Coverage percentages

**Intensity Correlation:**
   * Pearson correlation of registered channels
   * Multi-channel correlation matrices
   * Channel-specific correlations

**Spatial Consistency:**
   * Feature distribution similarity
   * Registration matrix properties

Parameter Optimization (Defaults)
----------------------

Twocan optimizes parameters across several categories:

Preprocessing Parameters
~~~~~~~~~~~~~~~~~~~~~~~~

**IF Processing:**
   * ``binarization_threshold``: Threshold for creating binary masks
   * ``gaussian_sigma``: Amount of blurring for noise reduction

**IMC Processing:**
   * ``arcsinh_cofactor``: Scaling factor for arcsinh transformation
   * ``winsorization_limits``: Percentiles for outlier clipping
   * ``binarization_threshold``: Threshold for binary masks
   * ``gaussian_sigma``: Blurring parameter

Registration Parameters
~~~~~~~~~~~~~~~~~~~~~~~

* ``max_features``: Maximum ORB features to detect
* ``percentile``: Fraction of best matches to keep
* ``registration_target``: Which image serves as the reference

Objective Functions
-------------------

Twocan provides different objective functions for optimization:

Multi Objective
~~~~~~~~~~~~~~~

Treats different metrics as separate objectives for optimization:

* **Objective 1**: Geometric overlap (IoU) of thresholded pixels
* **Objective 2**: Cross-modality correlation in the intersection of thresholded pixels

Single Objective
~~~~~~~~~~~~~~~~

Optimizes the product of the two metrics:

.. math::

   \text{objective} =  \text{Cell IoU} \cdot \text{Cell correlation}



Custom Objectives
~~~~~~~~~~~~~~~~~

You can define custom objective functions for specific use cases:

.. code-block:: python

   def custom_objective(trial, images, channels, **kwargs):
       # Your custom registration logic
       # Return single value or list of values
       return registration_score

When to Use Twocan
------------------

Twocan is particularly useful when:

* **Multiple modalities** need registration, especially from highly multiplexed omics technologies
* **Manual parameter tuning** is impractical
* **Registration quality** is critical for downstream analysis
* **Reproducible results** are required

Twocan may not be the best choice when:

* **Very large images** exceed memory constraints


Limitations and Considerations
------------------------------

**Computational Cost:**
   Bayesian optimization requires multiple registration attempts, making it slower than single-shot methods.

**Memory Requirements:**
   Large images and multiple channels can require substantial memory.

**Feature-based Approach:**
   Requires detectable features; may struggle with very smooth or homogeneous images.

**Affine Transformation Model:**
   Cannot handle complex non-linear deformations that might occur in some biological samples. 

**Channel Selection:**
   Registration quality depends heavily on choosing appropriate channels that are present in both modalities. 