import os
import cv2
import optuna
import numpy as np
import spatialdata as sd
from tifffile import imread
from skimage import transform
import matplotlib.pyplot as plt
from scipy.stats.mstats import winsorize
from twocan import preprocess_if, preprocess_imc, multi_channel_corr, RegEstimator, get_merge, plot_cartoon_affine
from twocan.utils import pick_best_registration
from twocan.callbacks import SaveTrialsDFCallback

FISH_scale = 0.2

OUT_DIR = 'results/'

os.makedirs(OUT_DIR, exist_ok=True)


def preprocess_fish_imc(trial, IMC, FISH, registration_channels):
    # source and target are sdata
    # ===============================================
    # set up trial with ranges of parameters to try
    # ===============================================
    # FISH parameters
    FISH_binarization_threshold = trial.suggest_float("FISH_binarization_threshold", 0, 1)
    FISH_gaussian_sigma = trial.suggest_float("FISH_gaussian_sigma", 0, 5)
    # IMC parameters
    IMC_arcsinh_normalize = trial.suggest_categorical("IMC_arcsinh_normalize", [True, False])
    IMC_arcsinh_cofactor = trial.suggest_float("IMC_arcsinh_cofactor", 1, 100) 
    IMC_winsorization_lower_limit = trial.suggest_float("IMC_winsorization_lower_limit", 0, 0.2)
    IMC_winsorization_upper_limit = trial.suggest_float("IMC_winsorization_upper_limit", 0, 0.2)
    IMC_binarization_threshold = trial.suggest_float("IMC_binarization_threshold", 0, 1)
    IMC_gaussian_sigma = trial.suggest_float("IMC_gaussian_sigma", 0, 5)
    # ===============================================
    # do preprocessing for registration
    FISH_reg = FISH[FISH.c.to_index().isin(registration_channels)]
    IMC_reg = IMC[IMC.c.to_index().isin(registration_channels)]
    FISH_processed = preprocess_if(FISH_reg.to_numpy(), 1, True, FISH_binarization_threshold, FISH_gaussian_sigma)
    IMC_processed = preprocess_imc(IMC_reg.to_numpy(), IMC_arcsinh_normalize, IMC_arcsinh_cofactor, [IMC_winsorization_lower_limit, IMC_winsorization_upper_limit], True, IMC_binarization_threshold, IMC_gaussian_sigma)
    return [IMC_processed, FISH_processed]

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
    if np.isnan(trial.user_attrs['reg_image_max_corr']):
        return 0
    else: 
        return trial.user_attrs['reg_image_max_corr'] * trial.user_attrs['logical_iou']


if __name__ == '__main__':


    ## technical replicate 2
    # =================================
    # load data 
    print('registering technical replicate 2')
    fish_rescaled = np.flipud(transform.rescale(imread('data/PPIB_2/registered/PPIB_RNA_IF.tif'), FISH_scale, preserve_range=True, anti_aliasing=True))
    FISH = sd.models.Image2DModel.parse(data=fish_rescaled[None,:,:], c_coords=['PPIB'])
    IMC = sd.models.Image2DModel.parse(data=np.vstack([imread('data/PPIB_2/DNA2(Ir193Di).tiff')[None,:,:],imread('data/PPIB_2/C2_IMC_PPIB_10nM.tiff')[None,:,:]]), c_coords=['DNA2', 'PPIB'])

    # register IMC to FISH
    study_name = 'rep2'
    cbs = [SaveTrialsDFCallback(f'{OUT_DIR}/{study_name}.csv', anno_dict={})]
    study_2 = optuna.create_study(direction='maximize', study_name = study_name, sampler = optuna.samplers.TPESampler(seed=int(42)))
    study_2.optimize(lambda trial: iou_single_objective(trial, IMC, FISH, registration_channels=['PPIB'], preprocesser=preprocess_fish_imc), callbacks=cbs, n_trials=50)

    # pick the best registration
    study_df_2 = study_2.trials_dataframe()
    best_trial = pick_best_registration(study_df_2)
    reg = RegEstimator()
    reg.M_ = best_trial['user_attrs_registration_matrix']
    stack = reg.transform(IMC.to_numpy(), FISH.to_numpy())

    # visualize
    stack = np.array([winsorize(c, limits=[0,0.05]) for c in stack])
    green, magenta, comb = get_merge((stack[1]), stack[2])

    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 5))
    ax1.imshow(green)
    ax1.axis('off')
    ax1.set_title('PPIB (IMC)')   
    ax2.imshow(magenta)
    ax2.axis('off')
    ax2.set_title('PPIB (FISH)')
    ax3.imshow(comb)
    ax3.axis('off')
    ax3.set_title('Merge')
    ax4, lines = plot_cartoon_affine(*IMC.T.shape[:2], reg.M_, *FISH.T.shape[:2], show_source=False, source_color='#37c100', target_color='#cc008b')
    ax4.axis('off')
    ax4.invert_yaxis()
    ax4.set_title('Registration')
    plt.savefig(f'{OUT_DIR}/{study_name}.png', dpi = 600)
    plt.close()


    ## technical replicate 3
    # =================================
    # load data 
    print('registering technical replicate 3')
    fish_rescaled = np.flipud(transform.rescale(imread('data/PPIB_3/registered/PPIB_IF_RNA.tif'), FISH_scale, preserve_range=True, anti_aliasing=True))
    FISH = sd.models.Image2DModel.parse(data=fish_rescaled[None,:,:], c_coords=['PPIB'])
    IMC = sd.models.Image2DModel.parse(data=np.vstack([imread('data/PPIB_3/DNA2(Ir193Di)_rotated.tif')[None,:,:],imread('data/PPIB_3/C2_PPIB(Ho165Di)_rotated.tiff')[None,:,:]]), c_coords=['DNA2', 'PPIB'])

    # register IMC to FISH
    study_name = 'rep3'
    cbs = [SaveTrialsDFCallback(f'{OUT_DIR}/{study_name}.csv', anno_dict={})]
    study_3 = optuna.create_study(direction='maximize', study_name = study_name, sampler = optuna.samplers.TPESampler(seed=int(42)))
    study_3.optimize(lambda trial: iou_single_objective(trial, IMC, FISH, registration_channels=['PPIB'], preprocesser=preprocess_fish_imc), callbacks=cbs, n_trials=50)


    # pick the best registration
    study_df_3 = study_3.trials_dataframe()
    best_trial = pick_best_registration(study_df_3)
    reg = RegEstimator()
    reg.M_ = best_trial['user_attrs_registration_matrix']
    stack = reg.transform(IMC.to_numpy(), FISH.to_numpy())

    # visualize
    stack = np.array([winsorize(c, limits=[0,0.05]) for c in stack])
    green, magenta, comb = get_merge((stack[1]), stack[2])

    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 5))
    ax1.imshow(green)
    ax1.axis('off')
    ax1.set_title('PPIB (IMC)')   
    ax2.imshow(magenta)
    ax2.axis('off')
    ax2.set_title('PPIB (FISH)')
    ax3.imshow(comb)
    ax3.axis('off')
    ax3.set_title('Merge')
    ax4, lines = plot_cartoon_affine(*IMC.T.shape[:2], reg.M_, *FISH.T.shape[:2], show_source=False, source_color='#37c100', target_color='#cc008b')
    ax4.axis('off')
    ax4.invert_yaxis()
    ax4.set_title('Registration')
    plt.savefig(f'{OUT_DIR}/{study_name}.png', dpi = 600)
    plt.close()



