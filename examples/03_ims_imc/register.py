import twocan
import os
import cv2
import optuna
import numpy as np
import pandas as pd
import spatialdata as sd
from tifffile import imread
from skimage import transform
import matplotlib.pyplot as plt
from scipy.stats.mstats import winsorize
from twocan import preprocess_if, preprocess_imc, multi_channel_corr, RegEstimator, get_merge, plot_cartoon_affine
from twocan.callbacks import SaveTrialsDFCallback
from twocan.utils import pick_best_registration

OUT_DIR = 'results/'
os.makedirs(OUT_DIR, exist_ok=True)

# define the preprocessing parameters to run baysean optimizaiton over

def preprocess_ims_imc(trial, IMS, IMC, registration_channels):
    # IMS parameters
    IMS_binarization_threshold = trial.suggest_float("IMS_binarization_threshold", 0, 1)
    IMS_gaussian_sigma = trial.suggest_float("IMS_gaussian_sigma", 0, 5)
    # IMC parameters
    IMC_arcsinh_normalize = trial.suggest_categorical("IMC_arcsinh_normalize", [True, False])
    IMC_arcsinh_cofactor = trial.suggest_float("IMC_arcsinh_cofactor", 1, 100) 
    IMC_winsorization_lower_limit = trial.suggest_float("IMC_winsorization_lower_limit", 0, 0.2)
    IMC_winsorization_upper_limit = trial.suggest_float("IMC_winsorization_upper_limit", 0, 0.2)
    IMC_binarization_threshold = trial.suggest_float("IMC_binarization_threshold", 0, 1)
    IMC_gaussian_sigma = trial.suggest_float("IMC_gaussian_sigma", 0, 5)
    # ===============================================
    # do preprocessing for registration
    IMS_reg = IMS[IMS.c.to_index().isin(registration_channels)]
    IMC_reg = IMC[IMC.c.to_index().isin(registration_channels)]
    IMS_processed = preprocess_if(IMS_reg.to_numpy(), 1, True, IMS_binarization_threshold, IMS_gaussian_sigma)
    IMC_processed = preprocess_imc(IMC_reg.to_numpy(), IMC_arcsinh_normalize, IMC_arcsinh_cofactor, [IMC_winsorization_lower_limit, IMC_winsorization_upper_limit], True, IMC_binarization_threshold, IMC_gaussian_sigma)
    return [IMS_processed, IMC_processed]


# specify trial
def registration_trial(trial, source_processed, target_processed, source, target, registration_channels):
    # source and target should be sdata images
    # processed should be binary np arrays shape 1xWxH
    # chanel list should match the channels in sdata
    source_reg = source[source.c.to_index().isin(registration_channels)]
    target_reg = target[target.c.to_index().isin(registration_channels)]
    df_na_list = [
        'registration_matrix','prop_source_covered', 'prop_target_covered', 
        'logical_and', 'logical_xor','logical_iou',
        'stack_image_max_corr','reg_image_max_corr',
        'stack_cell_max_corr','reg_cell_max_corr',
    ]
    if (source_processed).all() or (~source_processed).all() or (target_processed).all() or (~target_processed).all(): 
        [trial.set_user_attr(k, np.nan) for k in df_na_list]
        return
    # register
    reg = RegEstimator(int(1e5), 0.9)
    try: reg.fit(source_processed, target_processed)
    except cv2.error: 
        [trial.set_user_attr(k, np.nan) for k in df_na_list]
        return
    # fail if singular matrix
    if (reg.M_ is None) or (np.linalg.det(reg.M_[0:2,0:2]) == 0):
        [trial.set_user_attr(k, np.nan) for k in df_na_list]
        return
    # fail if transformation eliminates signal
    if np.allclose(reg.transform(source_reg),0): 
        [trial.set_user_attr(k, np.nan) for k in df_na_list]
        return
    # record metrics
    score = reg.score(source_processed, target_processed)
    stack = reg.transform(source, target)
    # filter to registration channels in reg_stack
    reg_stack = stack[np.concatenate([source.c.to_index().isin(registration_channels),target.c.to_index().isin(registration_channels)])]
    # correlation over image intersection 
    stack_mask = reg.transform(np.ones(source_processed.shape), np.ones(target_processed.shape)).sum(0) >1
    if stack_mask.any():
        # can be nan for channels with no signal
        stack_image_max_corr = np.nanmax(multi_channel_corr(stack[:,stack_mask][0:source.shape[0]], stack[:,stack_mask][target.shape[0]:]))
        reg_image_max_corr = np.nanmax(multi_channel_corr(reg_stack[:,stack_mask][0:source_reg.shape[0]], reg_stack[:,stack_mask][target_reg.shape[0]:]))
    else:
        stack_image_max_corr = np.nan
        reg_image_max_corr = np.nan
    # correlation over pixel intersection
    stack_mask = reg.transform(source_processed, target_processed).sum(0) >1
    if stack_mask.any():
        # can be nan for channels with no signal
        stack_cell_max_corr = np.nanmax(multi_channel_corr(stack[:,stack_mask][0:source.shape[0]], stack[:,stack_mask][target.shape[0]:]))
        reg_cell_max_corr = np.nanmax(multi_channel_corr(reg_stack[:,stack_mask][0:source_reg.shape[0]], reg_stack[:,stack_mask][target_reg.shape[0]:]))
    else:
        stack_cell_max_corr = np.nan
        reg_cell_max_corr = np.nan
    # ===============================================
    # set user attributes
    # ===============================================   
    trial.set_user_attr('registration_matrix', reg.M_)
    trial.set_user_attr('source_sum', score['source_sum'])
    trial.set_user_attr('target_sum', score['target_sum'])
    trial.set_user_attr('logical_and', score['and'])
    trial.set_user_attr('logical_or', score['or'])
    trial.set_user_attr('logical_xor', score['xor'])
    trial.set_user_attr('logical_iou', score['iou'])
    trial.set_user_attr('stack_image_max_corr', stack_image_max_corr)
    trial.set_user_attr('reg_image_max_corr', reg_image_max_corr)
    trial.set_user_attr('stack_cell_max_corr', stack_cell_max_corr)
    trial.set_user_attr('reg_cell_max_corr', reg_cell_max_corr)

# specify objective
def iou_single_objective(trial, source, target, registration_channels, preprocesser):
    source_processed, target_processed = preprocesser(trial, source, target, registration_channels)
    registration_trial(trial, source_processed, target_processed, source, target, registration_channels)
    if np.isnan(trial.user_attrs['reg_cell_max_corr']):
        return 0
    else: 
        return trial.user_attrs['reg_cell_max_corr'] * trial.user_attrs['logical_iou']



if __name__ == '__main__':

    # read in channel names
    ims_panel = pd.read_csv('data/ims_channels.csv', index_col=1).index
    imc_panel = pd.read_csv('data/prostate_panel.csv', index_col=3).index

    registration_channels = list(ims_panel[[57]]) + list(imc_panel[[49, 50]])

    # read in data
    IMS = sd.models.Image2DModel.parse(data=imread('data/155-A25-2_ims.tiff'), c_coords=ims_panel)
    
    # downsample IMC to get close to IMS resolution
    imc_rescale = np.stack([transform.rescale(x, 0.1, preserve_range=True, anti_aliasing=True) for x in imread('data/155-A25_ROI_001.tiff')])
    IMC = sd.models.Image2DModel.parse(data=imc_rescale, c_coords=imc_panel)

    # prep the study and run trials
    study_name = 'rep1'
    cbs = [SaveTrialsDFCallback(f'{OUT_DIR}/{study_name}.csv', anno_dict={})]
    study = optuna.create_study(direction='maximize', study_name = study_name, sampler = optuna.samplers.TPESampler(seed=int(42)))
    study.optimize(lambda trial: iou_single_objective(trial, IMS, IMC, registration_channels=registration_channels, preprocesser=preprocess_ims_imc), callbacks=cbs, n_trials=50)


    # pick the best registration
    study_df = study.trials_dataframe()
    best_trial = pick_best_registration(study_df)
    reg = RegEstimator()
    reg.M_ = best_trial['user_attrs_registration_matrix']
    stack = reg.transform(IMS.to_numpy(), IMC.to_numpy())

    # visualize
    ims_processed, imc_processed = preprocess_ims_imc(study.best_trial, IMS, IMC, registration_channels)
    viz_stack = np.array([winsorize(c, limits=[0,0.05]) for c in stack[[56, len(ims_panel) + 49],:,:]])
    green, magenta, comb = get_merge((viz_stack[0]), viz_stack[1])


    fig, ((ax1, ax2), (ax3, ax4), (ax5, ax6)) = plt.subplots(3, 2, figsize=(12, 8))
    ax1.imshow(ims_processed)
    ax1.axis('off')
    ax1.set_title('IMS processed')   
    ax2.imshow(imc_processed)
    ax2.axis('off')
    ax2.set_title('IMC processed')
    ax3.imshow(green)
    ax3.axis('off')
    ax3.set_title('IMS 2100.7347')   
    ax4.imshow(magenta)
    ax4.axis('off')
    ax4.set_title('IMC DNA1')
    ax5.imshow(comb)
    ax5.axis('off')
    ax5.set_title('Merge')
    ax6, lines = plot_cartoon_affine(*IMS.T.shape[:2], reg.M_, *IMC.T.shape[:2], show_source=False, source_color='#37c100', target_color='#cc008b')
    ax6.axis('off')
    ax6.invert_yaxis()
    ax6.set_title('Registration')
    plt.tight_layout()
    plt.savefig(f'{OUT_DIR}/{study_name}.png', dpi = 600)
    plt.close()
