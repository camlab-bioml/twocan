from typing import List, Optional, Tuple, Union
import re
import numpy as np
import cv2
#from skimage import transform
from skimage.filters import gaussian
from scipy.stats.mstats import winsorize
import numpy as np
from typing import List
import spatialdata as sd
from spatialdata.transformations import get_transformation, set_transformation
from spatialdata.transformations.transformations import BaseTransformation, Sequence
import pandas as pd


def stretch_255(image: np.ndarray) -> np.ndarray:
    """Convert an image to 8-bit grayscale by stretching its range to [0, 255].
    
    Parameters
    ----------
    image : np.ndarray
        Input image array of any shape.
        
    Returns
    -------
    np.ndarray
        8-bit grayscale image with values in [0, 255].
        Returns original image if max value is 0.
    """
    if image.max() == 0: return image
    return (image * (255/image.max())).astype('uint8')




def prep_zarr(IF_arr: np.ndarray, 
              IMC_arr: np.ndarray, 
              IF_panel: List[str], 
              IMC_panel: List[str]) -> sd.SpatialData:
    """Create a SpatialData object from IF and IMC arrays with their channel panels.
    
    Parameters
    ----------
    IF_arr : np.ndarray
        Immunofluorescence array of shape (H, W) or (C, H, W).
    IMC_arr : np.ndarray
        Imaging mass cytometry array of shape (H, W) or (C, H, W).
    IF_panel : List[str]
        List of channel names for IF data.
    IMC_panel : List[str]
        List of channel names for IMC data.
        
    Returns
    -------
    sd.SpatialData
        SpatialData object containing both modalities with channel information.
    """
    # prep zarr
    if IF_arr.ndim == 2:
        IF_arr = IF_arr[None, :, :]
    if IMC_arr.ndim == 2:
        IMC_arr = IMC_arr[None, :, :]
    IF = sd.models.Image2DModel.parse(data=IF_arr, c_coords=IF_panel)
    IMC = sd.models.Image2DModel.parse(data=IMC_arr, c_coords=IMC_panel)
    return sd.SpatialData({'IF': IF, 'IMC': IMC})



def get_aligned_coordinates(
    moving_element: sd.models.SpatialElement,
    reference_element: sd.models.SpatialElement,
    transformation: BaseTransformation,
    reference_coordinate_system: str = "global",
    moving_coordinate_system: str = "global",
    new_coordinate_system: str = 'aligned',
    write_to_sdata: sd.SpatialData = None,
) -> None:
    """Apply a transformation to align two spatial elements in a new coordinate system.
    
    Parameters
    ----------
    moving_element : sd.models.SpatialElement
        The element to be transformed.
    reference_element : sd.models.SpatialElement
        The reference element that defines the target space.
    transformation : BaseTransformation
        The transformation to apply to the moving element.
    reference_coordinate_system : str, default="global"
        Coordinate system of the reference element.
    moving_coordinate_system : str, default="global"
        Coordinate system of the moving element.
    new_coordinate_system : str, default='aligned'
        Name of the new coordinate system after alignment.
    write_to_sdata : sd.SpatialData, optional
        If provided, write the transformation to this SpatialData object.
    """
    old_moving_transformation = get_transformation(moving_element, moving_coordinate_system)
    old_reference_transformation = get_transformation(reference_element, reference_coordinate_system)
    assert isinstance(old_moving_transformation, BaseTransformation)
    assert isinstance(old_reference_transformation, BaseTransformation)
    #
    new_moving_transformation = Sequence([old_moving_transformation, transformation])
    new_reference_transformation = old_reference_transformation
    #
    set_transformation(moving_element, new_moving_transformation, new_coordinate_system, write_to_sdata=write_to_sdata)



def read_M(M: str) -> np.ndarray:
    """Parse a string representation of an affine transformation matrix.
    
    Parameters
    ----------
    M : str
        String representation of a 2x3 affine transformation matrix.
        
    Returns
    -------
    np.ndarray
        2x3 affine transformation matrix.
    """
    mstring = re.sub(r"\s+", ",", M)
    mstring = re.sub(r"\[\[,", "[[", mstring)
    mstring = re.sub(r"\],\[,", "],[", mstring)
    m = eval('np.array('+ mstring +')')
    return m


def multi_channel_corr(source: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Calculate correlation coefficients between all pairs of channels from two images.
    
    Parameters
    ----------
    source : np.ndarray
        Source image array of shape (n_channels_1, n_pixels).
    target : np.ndarray
        Target image array of shape (n_channels_2, n_pixels).
        
    Returns
    -------
    np.ndarray
        Correlation matrix of shape (n_channels_1, n_channels_2).
    """
    # Normalize the data
    source = (source - source.mean(axis=1)[:, None]) / source.std(axis=1)[:, None]
    target = (target - target.mean(axis=1)[:, None]) / target.std(axis=1)[:, None]
    # Calculate correlation matrix using matrix multiplication
    return np.dot(source, target.T) / source.shape[1]


def preprocess_if(source_image: np.ndarray, 
                 if_scale: float = 1, 
                 binarize: bool = True, 
                 binarization_threshold: float = 0.1, 
                 sigma: float = 1) -> np.ndarray:
    """Preprocess an immunofluorescence (IF) image for registration.
    
    Parameters
    ----------
    source_image : np.ndarray
        Input IF image of shape (C, H, W) where C is number of channels.
    if_scale : float, default=1
        Scale factor to resize the image.
    binarize : bool, default=True
        Whether to binarize the image.
    binarization_threshold : float, default=0.1
        Threshold for binarization.
    sigma : float, default=1
        Sigma for Gaussian smoothing.
        
    Returns
    -------
    np.ndarray
        Preprocessed image of shape (H', W') where H' and W' are scaled dimensions.
    """
    # cv2 resize is significantly faster than skimage rescale
    #source_image = transform.rescale(source_image, if_scale, preserve_range=True, anti_aliasing=True)
    source_image = np.array([cv2.resize(source_image[i], 
                                      (int(source_image.shape[2]*if_scale), 
                                       int(source_image.shape[1]*if_scale)), 
                                      interpolation=cv2.INTER_LANCZOS4) 
                            for i in range(source_image.shape[0])])
    source_image = source_image.sum(0)
    source_image = source_image / source_image.max()
    source_image = gaussian(source_image, sigma=sigma)
    if binarize:
        source_image = source_image > binarization_threshold      
    #source_image = stretch_255(source_image)
    return source_image


def preprocess_imc(target_image: np.ndarray, 
                  arcsinh_normalize: bool = True, 
                  arcsinh_cofactor: float = 5, 
                  winsorize_limits: List[Optional[float]] = [None, None], 
                  binarize: bool = True, 
                  binarization_threshold: float = 2, 
                  sigma: float = 1) -> np.ndarray:
    """Preprocess an imaging mass cytometry (IMC) image for registration.
    
    Parameters
    ----------
    target_image : np.ndarray
        Input IMC image of shape (C, H, W) where C is number of channels.
    arcsinh_normalize : bool, default=True
        Whether to apply arcsinh normalization.
    arcsinh_cofactor : float, default=5
        Cofactor for arcsinh normalization.
    winsorize_limits : List[Optional[float]], default=[None, None]
        Lower and upper limits for winsorization.
    binarize : bool, default=True
        Whether to binarize the image.
    binarization_threshold : float, default=2
        Threshold for binarization.
    sigma : float, default=1
        Sigma for Gaussian smoothing.
        
    Returns
    -------
    np.ndarray
        Preprocessed image of shape (H, W).
    """
    if arcsinh_normalize: 
        target_image = np.arcsinh(target_image/arcsinh_cofactor)
    target_image = target_image.sum(0) 
    target_image = winsorize(target_image, limits=winsorize_limits)
    target_image = target_image / target_image.max()
    target_image = gaussian(target_image, sigma=sigma)
    if binarize:
        target_image = target_image > binarization_threshold      
    #target_image = stretch_255(target_image)
    return target_image


def pick_best_registration(study_df):
    """Calculate triangle score and return best trial.
    
    Triangle score is calculated as:
    0.5 * |norm_and * norm_corr + norm_corr * norm_iou + norm_iou * norm_and|
    where each metric is normalized to [0,1] within the group.
    
    Args:
        study_df: DataFrame containing trial results with columns:
            - user_attrs_logical_and
            - user_attrs_logical_iou
            - user_attrs_reg_image_max_corr
            
    Returns:
        DataFrame row containing the best trial based on triangle score
    """
    study_df['norm_and'] = (np.log10(study_df['user_attrs_logical_and']+1)) / (np.log10(study_df['user_attrs_logical_and']+1).max())
    study_df['norm_iou'] = study_df['user_attrs_logical_iou'] / study_df['user_attrs_logical_iou'].max()
    study_df['norm_corr'] = study_df['user_attrs_reg_image_max_corr'] / study_df['user_attrs_reg_image_max_corr'].max()
    study_df['triangle_score'] = 0.5 * abs(study_df['norm_and'] * study_df['norm_corr'] + 
                                        study_df['norm_corr'] * study_df['norm_iou'] + 
                                        study_df['norm_iou'] * study_df['norm_and'])
    # Get the row with maximum triangle score
    best_row = study_df.loc[study_df['triangle_score'].idxmax()]
    return best_row
