import cv2
import numpy as np
from .utils import IFProcessor, IMCProcessor, multi_channel_corr
from .base import RegEstimator

def registration_trial(
    trial, images, registration_channels, correlation_channels,
    moving_image='IMC', static_image='IF', 
    moving_preprocessor=IMCProcessor(), static_preprocessor=IFProcessor()
):
    assert moving_image in images.keys() and static_image in images.keys()
    
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
    trial.suggest_categorical("moving_image", [moving_image])
    trial.suggest_categorical("static_image", [static_image])

    # Extract arrays and channels
    source = images[moving_image].to_numpy()
    target = images[static_image].to_numpy()
    source_reg = source[images[moving_image].c.to_index().isin(registration_channels)]
    target_reg = target[images[static_image].c.to_index().isin(registration_channels)]
    source_corr = source[images[moving_image].c.to_index().isin(correlation_channels)]
    
    # Preprocess images
    moving_preprocessor.configure(trial.params)
    static_preprocessor.configure(trial.params)
    source_processed = moving_preprocessor(source_reg)
    target_processed = static_preprocessor(target_reg)

    # list of attributes to set as NaN when trial fails 
    df_na_list = [
        'registration_matrix','prop_source_covered', 'prop_target_covered', 
        'logical_and', 'logical_xor','logical_iou',
        'stack_image_max_corr','reg_image_max_corr','corr_image_max_corr',
        'stack_cell_max_corr','reg_cell_max_corr','corr_cell_max_corr'
    ]

    # Check for invalid preprocessing results
    if (target_processed).all() or (~target_processed).all(): 
        [trial.set_user_attr(k, np.NaN) for k in df_na_list]
        return 
    if (source_processed).all() or (~source_processed).all(): 
        [trial.set_user_attr(k, np.NaN) for k in df_na_list]
        return 
    
    # Register images
    reg = RegEstimator(trial.params["registration_max_features"], trial.params["registration_percentile"])
    try:
        reg.fit(source_processed, target_processed)
    except cv2.error:
        [trial.set_user_attr(k, np.NaN) for k in df_na_list]
        return  
    
    # Check for invalid registration results
    if (reg.M_ is None) or (np.linalg.det(reg.M_[0:2,0:2]) == 0):
        [trial.set_user_attr(k, np.NaN) for k in df_na_list]
        return
    if np.allclose(reg.transform(source_reg), 0):
        [trial.set_user_attr(k, np.NaN) for k in df_na_list]
        return
        
    # Compute registration metrics
    score = reg.score(source_processed, target_processed)
    
    # Transform and stack images
    stack = reg.transform(source, target)

    # Extract channel-specific stacks
    reg_stack = stack[np.concatenate([
        images[moving_image].c.to_index().isin(registration_channels),
        images[static_image].c.to_index().isin(registration_channels)
    ])]
    corr_stack = stack[np.concatenate([
        images[moving_image].c.to_index().isin(correlation_channels),
        images[static_image].c.to_index().isin(correlation_channels)
    ])]

    # Check for invalid registration results
    if (reg.M_ is None) or (np.linalg.det(reg.M_[0:2,0:2]) == 0):
        [trial.set_user_attr(k, np.NaN) for k in df_na_list]
        return 
    if np.allclose(reg.transform(source_reg), 0):
        [trial.set_user_attr(k, np.NaN) for k in df_na_list]
        return 
    
    def get_max_corr(stack, mask, n_channels):
        return np.nanmax(multi_channel_corr(
            stack[:,mask][:n_channels], 
            stack[:,mask][n_channels:]
        )) if mask.any() else np.nan

    # Image intersection correlations
    mask = reg.transform(np.ones(source_processed.shape), np.ones(target_processed.shape)).sum(0) > 1
    stack_image_max_corr = get_max_corr(stack, mask, source.shape[0])
    reg_image_max_corr = get_max_corr(reg_stack, mask, source_reg.shape[0]) 
    corr_image_max_corr = get_max_corr(corr_stack, mask, source_corr.shape[0])

    # Pixel intersection correlations  
    mask = reg.transform(source_processed, target_processed).sum(0) > 1
    stack_cell_max_corr = get_max_corr(stack, mask, source.shape[0])
    reg_cell_max_corr = get_max_corr(reg_stack, mask, source_reg.shape[0])
    corr_cell_max_corr = get_max_corr(corr_stack, mask, source_corr.shape[0])

    # Compute registration metrics
    score = reg.score(source_processed, target_processed)
    trial.set_user_attr('registration_matrix', reg.M_)
    trial.set_user_attr('source_sum', score['source_sum'])
    trial.set_user_attr('target_sum', score['target_sum'])
    trial.set_user_attr('logical_and', score['and'])
    trial.set_user_attr('logical_or', score['or'])
    trial.set_user_attr('logical_xor', score['xor'])
    trial.set_user_attr('logical_iou', score['iou'])


def single_objective(trial, images, registration_channels, correlation_channels,
    moving_image='IMC', static_image='IF', 
    moving_preprocesser=IMCProcessor(), static_preprocesser=IFProcessor()

):
    raise NotImplementedError("single_objective not implemented")

def multi_objective(trial, images, registration_channels, correlation_channels,
    moving_image='IMC', static_image='IF', 
    moving_preprocesser=IMCProcessor(), static_preprocesser=IFProcessor()
):
    raise NotImplementedError("multi_objective not implemented")