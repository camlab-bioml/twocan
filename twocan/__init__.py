"""
Twocan: Bayesian optimization for multimodal registration of spatial proteomics.

Twocan is a Python package that uses Bayesian optimization (via Optuna) to 
automatically find optimal parameters for registering images from 
spatial proteomics technologies such as IF (Immunofluorescence), IMC (Imaging 
Mass Cytometry), FISH, IMS, etc.

The package provides:
- Automated parameter optimization for preprocessing and registration
- Modality-specific preprocessing functions  
- Quality metrics for registration assessment
- Visualization tools for registration results
- Extensible design for custom objectives and preprocessors

"""

# Local imports - these define the public API
from .base import RegEstimator
from .callbacks import SaveTrialsDFCallback, ThresholdReachedCallback, MatrixConvergenceCallback
from .plotting import plot_cartoon_affine, get_merge, get_rectangle_area, AsinhNorm
from .utils import (
    stretch_255, read_M, multi_channel_corr, 
    IFProcessor, IMCProcessor, get_aligned_coordinates, prep_zarr, pick_best_registration
)
from .optimize import registration_trial, iou_corr_single_objective, iou_corr_multi_objective

__version__ = "0.1.2"

__all__ = [
    "RegEstimator",
    "IFProcessor", 
    "IMCProcessor",
    "multi_channel_corr",
    "prep_zarr",
    "pick_best_registration",
    "registration_trial",
    "iou_corr_single_objective", 
    "iou_corr_multi_objective",
    "SaveTrialsDFCallback",
    "ThresholdReachedCallback",
    "MatrixConvergenceCallback", 
    "plot_cartoon_affine",
    "AsinhNorm",
] 