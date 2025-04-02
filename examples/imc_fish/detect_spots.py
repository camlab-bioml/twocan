import numpy as np
import pandas as pd
from tifffile import imread
import matplotlib.pyplot as plt
from deepcell_spots.applications import Polaris
from deepcell_spots.singleplex import process_spot_dict


###############
# rep 2

# Load image
fish = np.stack([
   np.flipud(imread('data/PPIB_2/registered/Dapi_IF.tif')),
   np.flipud(imread('data/PPIB_2/registered/PPIB_RNA_IF.tif'))
])

# If image has multiple channels, select the ones you need
# For example, if channel 0 is nuclear and channel 1 has spots
nuc_image = fish[0:1]
spots_image = fish[1:2]


# Detect nuclei and spots
app = Polaris(segmentation_type='nucleus')
pred = app.predict(
    spots_image=spots_image[:,:,:,None], 
    segmentation_image=nuc_image[:,:,:,None], 
    spots_threshold=0.95
)

# Extract results
spot_dict_nuc = pred[0]['spots_assignment']
labeled_im_nuc = pred[0]['cell_segmentation']
coords, cmap_list = process_spot_dict(spot_dict_nuc)

# Visualize results
fig, ax = plt.subplots(2, 2, figsize=(15, 15))
ax[0, 0].imshow(nuc_image[0], cmap='gray', vmax=np.percentile(nuc_image, 99.5))
ax[0, 0].set_title('Nuclear')
ax[0, 1].imshow(labeled_im_nuc[0,...,0]%13, cmap='jet')
ax[0, 1].set_title('Nuclear segmentation')
ax[1, 0].imshow(spots_image[0], cmap='gray')
ax[1, 0].set_title('Spots')
ax[1, 1].imshow(spots_image[0], cmap='gray')
ax[1, 1].scatter(coords[:,1], coords[:,0], c=np.array(cmap_list)%13, cmap='jet', s=0.1, edgecolors='none')
ax[1, 1].set_title('Spot assignment to cells')
plt.tight_layout()
plt.savefig('results/rep2_spots.png', dpi=600)

# save coords
coords = pd.DataFrame(coords, columns=['x', 'y'])
coords.to_csv('results/rep2_deepcell_spots.csv', index=False)


###############
# rep 3

# Load image
fish = np.stack([
   np.flipud(imread('data/PPIB_3/registered/Dapi.tif')),
   np.flipud(imread('data/PPIB_3/registered/PPIB_IF_RNA.tif'))
])

# If image has multiple channels, select the ones you need
# For example, if channel 0 is nuclear and channel 1 has spots
nuc_image = fish[0:1]
spots_image = fish[1:2]



# If image has multiple channels, select the ones you need
# For example, if channel 0 is nuclear and channel 1 has spots
nuc_image = fish[0:1]
spots_image = fish[1:2]


# Detect nuclei and spots
app = Polaris(segmentation_type='nucleus')
pred = app.predict(
    spots_image=spots_image[:,:,:,None], 
    segmentation_image=nuc_image[:,:,:,None], 
    spots_threshold=0.95
)

# Extract results
spot_dict_nuc = pred[0]['spots_assignment']
labeled_im_nuc = pred[0]['cell_segmentation']
coords, cmap_list = process_spot_dict(spot_dict_nuc)

# Visualize results
fig, ax = plt.subplots(2, 2, figsize=(15, 15))
ax[0, 0].imshow(nuc_image[0], cmap='gray', vmax=np.percentile(nuc_image, 99.5))
ax[0, 0].set_title('Nuclear')
ax[0, 1].imshow(labeled_im_nuc[0,...,0]%13, cmap='jet')
ax[0, 1].set_title('Nuclear segmentation')
ax[1, 0].imshow(spots_image[0], cmap='gray')
ax[1, 0].set_title('Spots')
ax[1, 1].imshow(spots_image[0], cmap='gray')
ax[1, 1].scatter(coords[:,1], coords[:,0], c=np.array(cmap_list)%13, cmap='jet', s=0.1, edgecolors='none')
ax[1, 1].set_title('Spot assignment to cells')
plt.tight_layout()
plt.savefig('results/rep3_spots.png', dpi=600)

# save coords
coords = pd.DataFrame(coords, columns=['x', 'y'])
coords.to_csv('results/rep3_deepcell_spots.csv', index=False)