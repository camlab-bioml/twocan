"""
Twocan: A Bayesian optimization framework for multimodal registration of highly multiplexed single-cell spatial proteomics data.
"""

# Local imports - these define the public API
from .base import RegEstimator
from .callbacks import SaveTrialsDFCallback, ThresholdReachedCallback, MatrixConvergenceCallback
from .plotting import plot_cartoon_affine, get_merge
from .utils import (
    stretch_255, read_M, multi_channel_corr, 
    preprocess_if, preprocess_imc, get_aligned_coordinates, prep_zarr
)


__version__ = "0.1.0" 