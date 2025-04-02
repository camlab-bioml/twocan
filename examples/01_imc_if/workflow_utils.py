import cv2
import numpy as np
from twocan import preprocess_if, preprocess_imc, RegEstimator, multi_channel_corr, read_M, plot_cartoon_affine
from twocan.utils import pick_best_registration
import matplotlib.pyplot as plt
from scipy.stats.mstats import winsorize





def iou_single_objective(trial, images, if_scale, registration_channels, correlation_channels):
    """Objective function that optimizes for IoU (Intersection over Union)."""
    registration_trial(trial, images, if_scale, registration_channels, correlation_channels)
    if np.isnan(trial.user_attrs['reg_image_max_corr']):
        return 0
    return trial.user_attrs['reg_image_max_corr'] * trial.user_attrs['logical_iou']

def iou_multi_objective(trial, images, if_scale, registration_channels, correlation_channels):
    """Multi-objective function that optimizes for both correlation and IoU."""
    registration_trial(trial, images, if_scale, registration_channels, correlation_channels)
    if np.isnan(trial.user_attrs['reg_image_max_corr']):
        return 0, 0
    return trial.user_attrs['reg_image_max_corr'], trial.user_attrs['logical_iou']




def registration_trial(trial, images, if_scale, registration_channels, correlation_channels):
    """Run a single registration trial with the given parameters.
    
    This function handles the optimization of registration parameters and computes
    various metrics for evaluating the registration quality.
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
    
    # Extract arrays and channels
    IF = images['IF'].to_numpy()
    IMC = images['IMC'].to_numpy()
    IF_reg = IF[images['IF'].c.to_index().isin(registration_channels)]
    IMC_reg = IMC[images['IMC'].c.to_index().isin(registration_channels)]
    IF_corr = IF[images['IF'].c.to_index().isin(correlation_channels)]
    IMC_corr = IMC[images['IMC'].c.to_index().isin(correlation_channels)]   
    
    # Preprocess images
    IF_processed = preprocess_if(IF_reg, if_scale, trial.params["binarize_images"], 
                               trial.params["IF_binarization_threshold"], 
                               trial.params["IF_gaussian_sigma"])
    IMC_processed = preprocess_imc(IMC_reg, trial.params["IMC_arcsinh_normalize"], 
                                 trial.params["IMC_arcsinh_cofactor"],
                                 [trial.params["IMC_winsorization_lower_limit"], 
                                  trial.params["IMC_winsorization_upper_limit"]], 
                                 trial.params["binarize_images"],
                                 trial.params["IMC_binarization_threshold"], 
                                 trial.params["IMC_gaussian_sigma"])
    
    # List of attributes to set as NaN when trial fails
    df_na_list = [
        'registration_matrix','prop_source_covered', 'prop_target_covered', 
        'logical_and', 'logical_xor','logical_iou',
        'stack_image_max_corr','reg_image_max_corr','corr_image_max_corr',
        'stack_cell_max_corr','reg_cell_max_corr','corr_cell_max_corr'
    ]

    # Check for invalid preprocessing results
    if (IF_processed).all() or (~IF_processed).all(): 
        [trial.set_user_attr(k, np.NaN) for k in df_na_list]
        return
    if (IMC_processed).all() or (~IMC_processed).all(): 
        [trial.set_user_attr(k, np.NaN) for k in df_na_list]
        return
    
    # Register images
    assert trial.params["registration_target"] == 'IF'
    reg = RegEstimator(trial.params["registration_max_features"], trial.params["registration_percentile"])
    try:
        reg.fit(IMC_processed, IF_processed)
    except cv2.error:
        [trial.set_user_attr(k, np.NaN) for k in df_na_list]
        return
        
    # Check for invalid registration results
    if (reg.M_ is None) or (np.linalg.det(reg.M_[0:2,0:2]) == 0):
        [trial.set_user_attr(k, np.NaN) for k in df_na_list]
        return
    if np.allclose(reg.transform(IMC_reg), 0):
        [trial.set_user_attr(k, np.NaN) for k in df_na_list]
        return
        
    # Compute registration metrics
    score = reg.score(IMC_processed, IF_processed)
    
    # Transform and stack images
    if_rescale_shape = (int(IF.shape[2]*if_scale), int(IF.shape[1]*if_scale))
    stack = reg.transform(IMC, np.array([cv2.resize(IF[i], if_rescale_shape, 
                        interpolation=cv2.INTER_LANCZOS4) for i in range(IF.shape[0])]))
    
    # Extract channel-specific stacks
    reg_stack = stack[np.concatenate([
        images['IMC'].c.to_index().isin(registration_channels),
        images['IF'].c.to_index().isin(registration_channels)
    ])]
    corr_stack = stack[np.concatenate([
        images['IMC'].c.to_index().isin(correlation_channels),
        images['IF'].c.to_index().isin(correlation_channels)
    ])]
    
    # Compute correlations over image intersection
    stack_mask = reg.transform(np.ones(IMC_processed.shape), 
                             np.ones(IF_processed.shape)).sum(0) > 1
    if stack_mask.any():
        stack_image_max_corr = np.nanmax(multi_channel_corr(
            stack[:,stack_mask][0:IMC.shape[0]], 
            stack[:,stack_mask][IMC.shape[0]:]
        ))
        reg_image_max_corr = np.nanmax(multi_channel_corr(
            reg_stack[:,stack_mask][0:IMC_reg.shape[0]], 
            reg_stack[:,stack_mask][IMC_reg.shape[0]:]
        ))
        corr_image_max_corr = np.nanmax(multi_channel_corr(
            corr_stack[:,stack_mask][0:IMC_corr.shape[0]], 
            corr_stack[:,stack_mask][IMC_corr.shape[0]:]
        ))
    else:
        stack_image_max_corr = reg_image_max_corr = corr_image_max_corr = np.nan
        
    # Compute correlations over pixel intersection
    stack_mask = reg.transform(IMC_processed, IF_processed).sum(0) > 1
    if stack_mask.any():
        stack_cell_max_corr = np.nanmax(multi_channel_corr(
            stack[:,stack_mask][0:IMC.shape[0]], 
            stack[:,stack_mask][IMC.shape[0]:]
        ))
        reg_cell_max_corr = np.nanmax(multi_channel_corr(
            reg_stack[:,stack_mask][0:IMC_reg.shape[0]], 
            reg_stack[:,stack_mask][IMC_reg.shape[0]:]
        ))
        corr_cell_max_corr = np.nanmax(multi_channel_corr(
            corr_stack[:,stack_mask][0:IMC_corr.shape[0]], 
            corr_stack[:,stack_mask][IMC_corr.shape[0]:]
        ))
    else:
        stack_cell_max_corr = reg_cell_max_corr = corr_cell_max_corr = np.nan
        
    # Set trial attributes
    trial.set_user_attr('registration_matrix', reg.M_)
    trial.set_user_attr('source_sum', score['source_sum'])
    trial.set_user_attr('target_sum', score['target_sum'])
    trial.set_user_attr('logical_and', score['and'])
    trial.set_user_attr('logical_or', score['or'])
    trial.set_user_attr('logical_xor', score['xor'])
    trial.set_user_attr('logical_iou', score['iou'])
    trial.set_user_attr('stack_image_max_corr', stack_image_max_corr)
    trial.set_user_attr('reg_image_max_corr', reg_image_max_corr)
    trial.set_user_attr('corr_image_max_corr', corr_image_max_corr)
    trial.set_user_attr('stack_cell_max_corr', stack_cell_max_corr)
    trial.set_user_attr('reg_cell_max_corr', reg_cell_max_corr)
    trial.set_user_attr('corr_cell_max_corr', corr_cell_max_corr)


def plot_registration_results(images, best_trial, metadata_row, output_path):
    """Create visualization of registration results.
    
    Args:
        images: SpatialData object containing IF and IMC images
        best_trial: DataFrame row containing best trial parameters and results
        metadata_row: DataFrame row containing metadata for this pair
        output_path: Path to save the visualization
    """
    registration_channels = metadata_row['registration_channels'].split(' ')
    if_scale = metadata_row['if_scale']
    
    # Get registration channels
    IF_reg = images['IF'][images['IF'].c.to_index().isin(registration_channels)].to_numpy()
    IMC_reg = images['IMC'][images['IMC'].c.to_index().isin(registration_channels)].to_numpy()
    
    # Process images using best trial parameters
    IF_processed = preprocess_if(IF_reg, if_scale, 
                               best_trial['params_binarize_images'],
                               best_trial['params_IF_binarization_threshold'],
                               best_trial['params_IF_gaussian_sigma'])
    
    IMC_processed = preprocess_imc(IMC_reg,
                                 best_trial['params_IMC_arcsinh_normalize'],
                                 best_trial['params_IMC_arcsinh_cofactor'],
                                 [best_trial['params_IMC_winsorization_lower_limit'],
                                  best_trial['params_IMC_winsorization_upper_limit']],
                                 best_trial['params_binarize_images'],
                                 best_trial['params_IMC_binarization_threshold'],
                                 best_trial['params_IMC_gaussian_sigma'])
    
    # Get registration matrix
    M = read_M(best_trial['user_attrs_registration_matrix'])
    
    # Create figure
    fig = plt.figure(figsize=(5, 10))
    grid = plt.GridSpec(3, 2, figure=fig)
    
    # Plot original images
    ax1 = fig.add_subplot(grid[0,0])
    ax1.imshow(winsorize(IMC_reg.sum(0), limits=[0,0.05]))
    ax1.set_title('IMC', fontsize=12)
    ax1.axis('off')
    
    ax2 = fig.add_subplot(grid[0,1])
    ax2.imshow(winsorize(IF_reg.sum(0), limits=[0,0.05]))
    ax2.set_title('IF', fontsize=12)
    ax2.axis('off')
    
    # Plot processed images
    ax3 = fig.add_subplot(grid[1,0])
    ax3.imshow(IMC_processed)
    ax3.set_title('IMC processed', fontsize=12)
    ax3.axis('off')
    
    ax4 = fig.add_subplot(grid[1,1])
    ax4.imshow(IF_processed)
    ax4.set_title('IF processed', fontsize=12)
    ax4.axis('off')
    
    # Plot metadata and registration visualization
    ax5 = fig.add_subplot(grid[2,0])
    ax5.axis('off')
    info_text = '\n'.join([
        f"Zarr ID: {metadata_row.name}",
        f"Sampler: {best_trial['sampler']}",
        f"Objective: {best_trial['objective']}",
        f"Triangle score: {best_trial['triangle_score']:.3f}",
        f"Transformation matrix:\n{M.round(2)}"
    ])
    props = dict(boxstyle='round,pad=1', facecolor='white', alpha=0.8, ec='gray')
    ax5.text(0.5, 0.5, info_text,
            transform=ax5.transAxes,
            fontsize='small',
            verticalalignment='center',
            horizontalalignment='center',
            bbox=props)
    
    ax6 = fig.add_subplot(grid[2,1])
    source_shape = IMC_processed.T.shape
    target_shape = IF_processed.T.shape
    ax6, lines = plot_cartoon_affine(*source_shape, M, *target_shape, 
                                   show_source=False, 
                                   source_color='#37c100', 
                                   target_color='#cc008b')
    ax6.invert_yaxis()
    ax6.axis('off')
    ax6.set_title('Registration', fontsize=12)
    ax6.legend(bbox_to_anchor=(0.5, -0.5), loc='lower center', fontsize='small')
    lines[0].set_linewidth(2)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=600)
    plt.close()
