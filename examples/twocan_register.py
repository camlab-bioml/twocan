import os
import cv2
import twocan
import optuna
import argparse
import numpy as np
import pandas as pd
import spatialdata as sd
from itertools import chain
from pyometiff import OMETIFFReader, OMETIFFWriter
from skimage.filters import threshold_otsu

parser = argparse.ArgumentParser()
parser.add_argument('--if_path', type=str,  default = 'data/cell-line-0028/0c77abf.ome.tiff')
parser.add_argument('--imc_path', type=str, default = 'data/cell-line-0028/8205eb6.ome.tiff')
parser.add_argument('--registration_channels', nargs='+', type=str, default = ['DAPI', 'DNA1', 'DNA2'], help="List of registration channel names (e.g., DAPI CD45)")
parser.add_argument('--sampler', default = 'TPESampler', type=str)
parser.add_argument('--objective', default = 'iou_single_objective', type=str)
parser.add_argument('--seed', default = 42, type=int)
parser.add_argument('--n_trials', default = 200, type=int)
parser.add_argument('--study_out_dir', default = 'results/twocan/out/', type=str)
parser.add_argument('--registration_csv', default = 'results/twocan/affine_matrix.csv', type=str)
args = parser.parse_args()

os.makedirs(args.study_out_dir, exist_ok=True)
os.makedirs(os.path.dirname(args.registration_csv), exist_ok=True)
os.makedirs(os.path.join(args.study_out_dir, "preprocessed"), exist_ok=True)

study_id = f"{os.path.basename(args.if_path).split('.')[0]}-{os.path.basename(args.imc_path).split('.')[0]}-{args.objective}-{args.sampler}-{args.seed}"

sampler = {
    'TPESampler': optuna.samplers.TPESampler, 
    'RandomSampler': optuna.samplers.RandomSampler, 
    'GPSampler': optuna.samplers.GPSampler
}[args.sampler]

objective = {
    'iou_single_objective': twocan.iou_corr_single_objective,
    'iou_multi_objective': twocan.iou_corr_multi_objective
}[args.objective]

objective_direction = {
    'iou_single_objective': 'maximize',
    'iou_multi_objective': ['maximize','maximize']
}[args.objective]

# Load images
reader_imc = OMETIFFReader(fpath=args.imc_path)
imc_img_array, imc_metadata, _ = reader_imc.read()

reader_if = OMETIFFReader(fpath=args.if_path)
if_img_array, if_metadata, _ = reader_if.read()


# scale IF to 1um/pixel
if_scale = if_metadata['PhysicalSizeX']
#if_img_rescaled = np.array([cv2.resize(if_img_array[i], (int(if_img_array.shape[2]*if_scale), int(if_img_array.shape[1]*if_scale)), interpolation=cv2.INTER_LANCZOS4) for i in range(if_img_array.shape[0])])
if_img_rescaled = if_img_array # TODO note rescale in ometiff output if changing this

# Setup callbacks
cbs = [twocan.SaveTrialsDFCallback(f"{args.study_out_dir}/{study_id}.csv")]

# registration trial
study = optuna.create_study(
    direction=objective_direction, 
    study_name = study_id,
    sampler = sampler(seed=int(args.seed))
)

if_otsu = twocan.IFProcessor(binarize=False, binarization_threshold=0, sigma=0)
imc_otsu = twocan.IMCProcessor(arcsinh_normalize=False, winsorize_limits = [0.01,0.01], binarize=False, sigma=0)

# enqueue kim baseline as first trial
study.enqueue_trial({
    'IF_binarization_threshold': threshold_otsu(
        if_otsu(if_img_rescaled[pd.Series(list(if_metadata['Channels'])).isin(args.registration_channels)])
    ),
    'IF_gaussian_sigma': 0, 'IMC_gaussian_sigma': 0,
    'IMC_binarization_threshold': threshold_otsu(
        imc_otsu(imc_img_array[pd.Series(list(imc_metadata['Channels'])).isin(args.registration_channels)])
    ),
    'IMC_arcsinh_normalize': False,
    'IMC_winsorization_lower_limit': 0.01,
    'IMC_winsorization_upper_limit': 0.01, 
    'registration_target' : 'IF'
})

# prep images for twocan
IF = sd.models.Image2DModel.parse(data=if_img_rescaled, c_coords=list(if_metadata['Channels']))
IMC = sd.models.Image2DModel.parse(data=imc_img_array, c_coords=list(imc_metadata['Channels']))
images = sd.SpatialData({'IF': IF, 'IMC': IMC})

# run optuna study
study.optimize(
    lambda trial: objective(
        trial, images, args.registration_channels, 
        moving_image='IMC', moving_preprocesser=twocan.IMCProcessor(), 
        static_image='IF', static_preprocesser=twocan.IFProcessor()
    ), 
    n_trials=args.n_trials, 
    callbacks=cbs
)

# save affine transform for benchmark
best_trial = twocan.pick_best_registration(study.trials_dataframe())
A = np.array(best_trial['user_attrs_registration_matrix'])

# adjust for rescale 
#A = A / if_scale

record = pd.Series({
    'study_id': study_id, 
    'if_path': args.if_path, 'imc_path': args.imc_path, 
    'objective': args.objective, 'sampler': args.sampler,
    'seed': args.seed, 'n_trials': args.n_trials,
    'registration_matrix': A
})

pd.DataFrame([record]).to_csv(args.registration_csv, index=False, mode='a', header=not os.path.exists(args.registration_csv))

# save best trial preprocessed images for registration with other methods
if_preprocessed = twocan.IFProcessor(
    binarize=best_trial['params_binarize_images'],
    binarization_threshold=best_trial['params_IF_binarization_threshold'],
    sigma=best_trial['params_IF_gaussian_sigma']
)(if_img_rescaled[pd.Series(list(if_metadata['Channels'])).isin(args.registration_channels)])


writer = OMETIFFWriter(
    fpath=f"{args.study_out_dir}/preprocessed/{os.path.basename(args.if_path).split('.')[0]}.tiff",
    dimension_order="CYX",
    array=if_preprocessed[None,:,:].astype(np.uint16),
    metadata={
        "PhysicalSizeX": if_metadata['PhysicalSizeX'],
        "PhysicalSizeXUnit": if_metadata['PhysicalSizeXUnit'],
        "PhysicalSizeY": if_metadata['PhysicalSizeY'],
        "PhysicalSizeYUnit": if_metadata['PhysicalSizeYUnit'],
        "Channels": {
            '0': {
                "Name": 'twocan_preprocessed',
                "SamplesPerPixel": 1,
                "ID": "Channel:000"
            } 
        }
    }
)
writer.write()

imc_preprocessed = twocan.IMCProcessor(
    arcsinh_normalize=best_trial['params_IMC_arcsinh_normalize'],
    winsorize_limits=[best_trial['params_IMC_winsorization_lower_limit'], best_trial['params_IMC_winsorization_upper_limit']],
    binarize=best_trial['params_binarize_images'],
    binarization_threshold=best_trial['params_IMC_binarization_threshold'],
    sigma=best_trial['params_IMC_gaussian_sigma']
)(imc_img_array[pd.Series(list(imc_metadata['Channels'])).isin(args.registration_channels)])

writer = OMETIFFWriter(
    fpath=f"{args.study_out_dir}/preprocessed/{os.path.basename(args.imc_path).split('.')[0]}.tiff",
    dimension_order="CYX",
    array=imc_preprocessed[None,:,:].astype(np.uint16),
    metadata={
        "PhysicalSizeX": imc_metadata['PhysicalSizeX'],
        "PhysicalSizeXUnit": imc_metadata['PhysicalSizeXUnit'],
        "PhysicalSizeY": imc_metadata['PhysicalSizeY'],
        "PhysicalSizeYUnit": imc_metadata['PhysicalSizeYUnit'],
        "Channels": {
            '0': {
                "Name": 'twocan_preprocessed',
                "SamplesPerPixel": 1,
                "ID": "Channel:000"
            } 
        }
    }
)
writer.write()