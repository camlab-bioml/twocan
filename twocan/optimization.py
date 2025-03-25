"""Optimization utilities for multimodal image registration.

This module provides utilities and examples for optimizing image registration using Optuna.
The main pattern consists of two components:

1. registration_trial: Handles the registration process and records metrics
2. objective functions: Define how to score registrations using the recorded metrics

Examples
--------
>>> import optuna
>>> from twocan.optimization import registration_trial, iou_single_objective
>>> 
>>> # Create an Optuna study
>>> study = optuna.create_study(direction='maximize')
>>> 
>>> # Run optimization with default IoU objective
>>> study.optimize(lambda trial: iou_single_objective(
...     trial, images, if_scale=1.0,
...     registration_channels=['DAPI'],
...     correlation_channels=['CD3', 'CD20']
... ), n_trials=100)
"""
from typing import Dict, Any, List, Optional, Callable, Union
import warnings
import numpy as np
import optuna
from skimage.filters import threshold_otsu
from .utils import preprocess_if, preprocess_imc
from .base import RegEstimator


def registration_trial(trial: optuna.trial.Trial,
                      images: Dict[str, Any],
                      if_scale: float,
                      registration_channels: List[str],
                      correlation_channels: List[str]) -> None:
    """Run a single registration trial and record metrics.
    
    This function:
    1. Sets up trial parameters with reasonable ranges
    2. Preprocesses both IF and IMC images
    3. Performs registration
    4. Records metrics as trial attributes
    
    Parameters
    ----------
    trial : optuna.trial.Trial
        The Optuna trial object to update with parameters and metrics.
    images : Dict[str, Any]
        Dictionary containing 'IF' and 'IMC' image data.
    if_scale : float
        Scale factor for the IF image.
    registration_channels : List[str]
        Channel names to use for registration.
    correlation_channels : List[str]
        Channel names to use for correlation metrics.
        
    Notes
    -----
    The following metrics are recorded as trial attributes:
    - registration_matrix: 2x3 affine transformation matrix
    - logical_and: Overlap between binary masks
    - logical_or: Union of binary masks
    - logical_xor: XOR of binary masks
    - logical_iou: Intersection over Union
    - reg_image_max_corr: Maximum correlation between registered channels
    - corr_image_max_corr: Maximum correlation between correlation channels
    
    Examples
    --------
    >>> def custom_objective(trial, images, if_scale, reg_channels, corr_channels):
    ...     registration_trial(trial, images, if_scale, reg_channels, corr_channels)
    ...     # Get metrics from trial attributes
    ...     corr = trial.user_attrs['reg_image_max_corr']
    ...     iou = trial.user_attrs['logical_iou']
    ...     # Define custom scoring
    ...     return 0.6 * corr + 0.4 * iou if not np.isnan(corr) else 0
    """
    # Set up trial parameters
    trial.suggest_float("IF_binarization_threshold", 0, 1)
    trial.suggest_float("IF_gaussian_sigma", 0, 5)
    trial.suggest_categorical("IMC_arcsinh_normalize", [True, False])
    trial.suggest_float("IMC_arcsinh_cofactor", 1, 100)
    trial.suggest_float("IMC_winsorization_lower_limit", 0, 0.2)
    trial.suggest_float("IMC_winsorization_upper_limit", 0, 0.2)
    trial.suggest_float("IMC_binarization_threshold", 0, 1)
    trial.suggest_float("IMC_gaussian_sigma", 0, 5)
    trial.suggest_categorical("binarize_images", [True])
    trial.suggest_categorical("registration_max_features", [int(1e5)])
    trial.suggest_categorical("registration_percentile", [0.9])
    trial.suggest_categorical("registration_target", ['IF'])
    
    # Extract images and channels
    IF = images['IF'].to_numpy()
    IMC = images['IMC'].to_numpy()
    IF_reg = IF[images['IF'].c.to_index().isin(registration_channels)]
    IMC_reg = IMC[images['IMC'].c.to_index().isin(registration_channels)]
    IF_corr = IF[images['IF'].c.to_index().isin(correlation_channels)]
    IMC_corr = IMC[images['IMC'].c.to_index().isin(correlation_channels)]
    
    # Preprocess images
    IF_processed = preprocess_if(
        IF_reg, if_scale,
        binarize=trial.params["binarize_images"],
        binarization_threshold=trial.params["IF_binarization_threshold"],
        sigma=trial.params["IF_gaussian_sigma"]
    )
    IMC_processed = preprocess_imc(
        IMC_reg,
        arcsinh_normalize=trial.params["IMC_arcsinh_normalize"],
        arcsinh_cofactor=trial.params["IMC_arcsinh_cofactor"],
        winsorize_limits=[
            trial.params["IMC_winsorization_lower_limit"],
            trial.params["IMC_winsorization_upper_limit"]
        ],
        binarize=trial.params["binarize_images"],
        binarization_threshold=trial.params["IMC_binarization_threshold"],
        sigma=trial.params["IMC_gaussian_sigma"]
    )
    
    # Check for invalid preprocessing results
    if (IF_processed).all() or (~IF_processed).all():
        _set_nan_metrics(trial)
        return
    if (IMC_processed).all() or (~IMC_processed).all():
        _set_nan_metrics(trial)
        return
        
    # Perform registration
    reg = RegEstimator(
        trial.params["registration_max_features"],
        trial.params["registration_percentile"]
    )
    try:
        reg.fit(IMC_processed, IF_processed)
    except:
        _set_nan_metrics(trial)
        return
        
    # Record metrics
    score = reg.score(IMC_processed, IF_processed)
    trial.set_user_attr('registration_matrix', reg.M_)
    trial.set_user_attr('logical_and', score['and'])
    trial.set_user_attr('logical_or', score['or'])
    trial.set_user_attr('logical_xor', score['xor'])
    trial.set_user_attr('logical_iou', score['iou'])
    
    # Calculate correlations
    stack = reg.transform(IMC, IF)
    stack_mask = reg.transform(np.ones(IMC_processed.shape),
                             np.ones(IF_processed.shape)).sum(0) > 1
    
    if stack_mask.any():
        reg_stack = stack[np.concatenate([
            images['IMC'].c.to_index().isin(registration_channels),
            images['IF'].c.to_index().isin(registration_channels)
        ])]
        corr_stack = stack[np.concatenate([
            images['IMC'].c.to_index().isin(correlation_channels),
            images['IF'].c.to_index().isin(correlation_channels)
        ])]
        
        trial.set_user_attr('reg_image_max_corr',
            np.nanmax(np.corrcoef(
                reg_stack[:,stack_mask][0:IMC_reg.shape[0]],
                reg_stack[:,stack_mask][IMC_reg.shape[0]:]
            ))
        )
        trial.set_user_attr('corr_image_max_corr',
            np.nanmax(np.corrcoef(
                corr_stack[:,stack_mask][0:IMC_corr.shape[0]],
                corr_stack[:,stack_mask][IMC_corr.shape[0]:]
            ))
        )
    else:
        trial.set_user_attr('reg_image_max_corr', np.nan)
        trial.set_user_attr('corr_image_max_corr', np.nan)


def _set_nan_metrics(trial: optuna.trial.Trial) -> None:
    """Set all metrics to NaN for failed trials."""
    for metric in ['registration_matrix', 'logical_and', 'logical_or',
                  'logical_xor', 'logical_iou', 'reg_image_max_corr',
                  'corr_image_max_corr']:
        trial.set_user_attr(metric, np.nan)


def iou_single_objective(trial: optuna.trial.Trial,
                        images: Dict[str, Any],
                        if_scale: float,
                        registration_channels: List[str],
                        correlation_channels: List[str]) -> float:
    """Default objective function using IoU and correlation.
    
    Combines Intersection over Union with maximum channel correlation
    to create a single objective value.
    
    Parameters
    ----------
    trial : optuna.trial.Trial
        The Optuna trial object.
    images : Dict[str, Any]
        Dictionary containing 'IF' and 'IMC' image data.
    if_scale : float
        Scale factor for the IF image.
    registration_channels : List[str]
        Channel names to use for registration.
    correlation_channels : List[str]
        Channel names to use for correlation metrics.
        
    Returns
    -------
    float
        The objective value to maximize.
    """
    registration_trial(trial, images, if_scale, registration_channels, correlation_channels)
    if np.isnan(trial.user_attrs['reg_image_max_corr']):
        return 0
    else:
        return trial.user_attrs['reg_image_max_corr'] * trial.user_attrs['logical_iou']


def xor_single_objective(trial: optuna.trial.Trial,
                        images: Dict[str, Any],
                        if_scale: float,
                        registration_channels: List[str],
                        correlation_channels: List[str]) -> float:
    """Alternative objective function using XOR and correlation.
    
    Combines logical XOR with maximum channel correlation to create
    a single objective value that penalizes misalignment.
    
    Parameters
    ----------
    trial : optuna.trial.Trial
        The Optuna trial object.
    images : Dict[str, Any]
        Dictionary containing 'IF' and 'IMC' image data.
    if_scale : float
        Scale factor for the IF image.
    registration_channels : List[str]
        Channel names to use for registration.
    correlation_channels : List[str]
        Channel names to use for correlation metrics.
        
    Returns
    -------
    float
        The objective value to maximize.
    """
    registration_trial(trial, images, if_scale, registration_channels, correlation_channels)
    if np.isnan(trial.user_attrs['reg_image_max_corr']):
        return 0
    else:
        return trial.user_attrs['reg_image_max_corr'] * (
            trial.user_attrs['logical_and'] - trial.user_attrs['logical_xor']
        ) 