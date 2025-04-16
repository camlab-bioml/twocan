import numpy as np
import pandas as pd
from tifffile import imread
from register import FISH_scale
from twocan import RegEstimator, read_M
from twocan.utils import pick_best_registration
from scipy.stats import spearmanr, pearsonr
from scipy.io import loadmat


###############
# rep 1 skip - has no mask


###############
# rep 2

best_trial = pick_best_registration(pd.read_csv('results/rep2.csv'))

# load data
fish = np.stack([
   np.flipud(imread('data/PPIB_2/registered/Dapi_IF.tif')),
   np.flipud(imread('data/PPIB_2/registered/PPIB_RNA_IF.tif'))
])
imc = np.stack([imread('data/PPIB_2/DNA2(Ir193Di).tiff'),imread('data/PPIB_2/C2_IMC_PPIB_10nM.tiff')])
mask = np.flipud(imread('data/PPIB_2/registered/Dapi_IF_Mask.tiff'))
schultz_imc = np.stack([np.flipud(imread('data/PPIB_2/registered/DNA2_IMC.tif')),np.flipud(imread('data/PPIB_2/registered/PPIB_IMC.tif'))])


# get transform
M = read_M(best_trial['user_attrs_registration_matrix'])
M = M / FISH_scale
reg = RegEstimator()
reg.M_ = M
stack = reg.transform(imc, fish)
sum_imc_0 = np.bincount(mask.ravel(), weights=stack[0,:,:].ravel()) 
sum_imc_1 = np.bincount(mask.ravel(), weights=stack[1,:,:].ravel()) 
sum_schultz_0 = np.bincount(mask.ravel(), weights=schultz_imc[0,:,:].ravel()) 
sum_schultz_1 = np.bincount(mask.ravel(), weights=schultz_imc[1,:,:].ravel()) 
sum_fish_0 = np.bincount(mask.ravel(), weights=stack[2,:,:].ravel()) 
sum_fish_1 = np.bincount(mask.ravel(), weights=stack[3,:,:].ravel()) 

# exclude bin 0 (background)

# correlation of total signal
spearmanr(sum_imc_1[1:], sum_fish_1[1:])
spearmanr(sum_schultz_1[1:], sum_fish_1[1:])
pearsonr(sum_imc_1[1:], sum_fish_1[1:])
pearsonr(sum_schultz_1[1:], sum_fish_1[1:])

# correlation of mean signal
counts = np.bincount(mask.ravel())
spearmanr(sum_imc_1[1:]/counts[1:], sum_fish_1[1:]/counts[1:])
spearmanr(sum_schultz_1[1:]/counts[1:], sum_fish_1[1:]/counts[1:])
pearsonr(sum_imc_1[1:]/counts[1:], sum_fish_1[1:]/counts[1:])
pearsonr(sum_schultz_1[1:]/counts[1:], sum_fish_1[1:]/counts[1:])

# read in spots
spots = pd.read_csv('results/rep2_deepcell_spots.csv')

# for each cell in mask, count the number of spots 
spot_locations = np.zeros(mask.shape)
spot_locations[spots['x'].round().astype(int), spots['y'].round().astype(int)] = 1

# count the number of spots in each cell
cell_counts = np.bincount(mask.ravel(), weights=spot_locations.ravel())

cell_counts[1:].mean()
np.median(cell_counts[1:])

# correlation of RNA spots withh total ion counts
spearmanr(sum_imc_1[1:], cell_counts[1:])
spearmanr(sum_schultz_1[1:], cell_counts[1:])
pearsonr(sum_imc_1[1:], cell_counts[1:])
pearsonr(sum_schultz_1[1:], cell_counts[1:])


###############
# rep 3


best_trial = pick_best_registration(pd.read_csv('results/rep3.csv'))

# load data
fish = np.stack([
   np.flipud(imread('data/PPIB_3/registered/Dapi.tif')),
   np.flipud(imread('data/PPIB_3/registered/PPIB_IF_RNA.tif'))
])
imc = np.stack([imread('data/PPIB_3/DNA2(Ir193Di)_rotated.tif'), imread('data/PPIB_3/C2_PPIB(Ho165Di)_rotated.tiff')])
mask = np.flipud(loadmat('data/PPIB_3/registered/Dapi_Mask.mat')['Image'])
schultz_imc = np.stack([np.flipud(imread('data/PPIB_3/registered/PPIB_IMC.tif'))])


# get transform
M = read_M(best_trial['user_attrs_registration_matrix'])
M = M / FISH_scale
reg = RegEstimator()
reg.M_ = M
stack = reg.transform(imc, fish)
sum_imc_0 = np.bincount(mask.ravel(), weights=stack[0,:,:].ravel()) 
sum_imc_1 = np.bincount(mask.ravel(), weights=stack[1,:,:].ravel()) 
sum_schultz_0 = np.bincount(mask.ravel(), weights=schultz_imc[0,:,:].ravel()) 
sum_fish_0 = np.bincount(mask.ravel(), weights=stack[2,:,:].ravel()) 
sum_fish_1 = np.bincount(mask.ravel(), weights=stack[3,:,:].ravel()) 

# correlation of total signal
spearmanr(sum_imc_1[1:], sum_fish_1[1:])
spearmanr(sum_schultz_0[1:], sum_fish_1[1:])
pearsonr(sum_imc_1[1:], sum_fish_1[1:])
pearsonr(sum_schultz_0[1:], sum_fish_1[1:])

# correlation of mean signal
counts = np.bincount(mask.ravel())
spearmanr(sum_imc_1[1:]/counts[1:], sum_fish_1[1:]/counts[1:])
spearmanr(sum_schultz_0[1:]/counts[1:], sum_fish_1[1:]/counts[1:])
pearsonr(sum_imc_1[1:]/counts[1:], sum_fish_1[1:]/counts[1:])
pearsonr(sum_schultz_0[1:]/counts[1:], sum_fish_1[1:]/counts[1:])

# read in spots
spots = pd.read_csv('results/rep3_deepcell_spots.csv')

# for each cell in mask, count the number of spots 
spot_locations = np.zeros(mask.shape)
spot_locations[spots['x'].round().astype(int), spots['y'].round().astype(int)] = 1

# count the number of spots in each cell
cell_counts = np.bincount(mask.ravel(), weights=spot_locations.ravel())
cell_counts[1:].mean()
np.median(cell_counts[1:])

# correlation of RNA spots withh total ion counts
spearmanr(sum_imc_1[1:], cell_counts[1:])
spearmanr(sum_schultz_0[1:], cell_counts[1:])
pearsonr(sum_imc_1[1:], cell_counts[1:])
pearsonr(sum_schultz_0[1:], cell_counts[1:])