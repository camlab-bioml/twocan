"""
Utility functions and preprocessing classes for Twocan image registration.

This module provides preprocessing classes for different imaging modalities,
utility functions for data manipulation, and helper functions for working with
SpatialData objects and transformation matrices.
"""

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
    
    This function linearly stretches the intensity values of an image to span
    the full 8-bit range [0, 255]. If the input image has zero maximum value,
    the original image is returned unchanged.
    
    Parameters
    ----------
    image : np.ndarray
        Input image array of any shape and dtype.
        
    Returns
    -------
    np.ndarray
        8-bit grayscale image with values in [0, 255] and dtype uint8.
        Returns original image unchanged if max value is 0.
    """
    if image.max() == 0: return image
    return (image * (255/image.max())).astype('uint8')




def prep_zarr(IF_arr: np.ndarray, 
              IMC_arr: np.ndarray, 
              IF_panel: List[str], 
              IMC_panel: List[str]) -> sd.SpatialData:
    """Create a SpatialData object from IF and IMC arrays with their channel panels.
    
    This function takes raw image arrays and their corresponding channel names
    to create a properly formatted SpatialData object containing both modalities.
    The function handles both 2D and 3D input arrays, automatically adding a
    channel dimension if needed.
    
    Parameters
    ----------
    IF_arr : np.ndarray
        Immunofluorescence array of shape (H, W) or (C, H, W).
        If 2D, will be expanded to (1, H, W).
    IMC_arr : np.ndarray
        Imaging mass cytometry array of shape (H, W) or (C, H, W).
        If 2D, will be expanded to (1, H, W).
    IF_panel : List[str]
        List of channel names for IF data. Must match the number of channels
        in IF_arr.
    IMC_panel : List[str]
        List of channel names for IMC data. Must match the number of channels
        in IMC_arr.
        
    Returns
    -------
    sd.SpatialData
        SpatialData object containing both modalities with proper channel
        information and coordinate systems.
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
    
    This function applies a spatial transformation to align a moving element with
    a reference element, creating a new coordinate system that contains both
    elements in aligned space. The transformation is applied on top of any
    existing transformations.
    
    Parameters
    ----------
    moving_element : sd.models.SpatialElement
        The spatial element to be transformed (e.g., image, points, shapes).
    reference_element : sd.models.SpatialElement
        The reference element that defines the target space.
    transformation : BaseTransformation
        The transformation to apply to the moving element.
    reference_coordinate_system : str, default="global"
        Name of the coordinate system for the reference element.
    moving_coordinate_system : str, default="global"
        Name of the coordinate system for the moving element.
    new_coordinate_system : str, default='aligned'
        Name of the new coordinate system after alignment.
    write_to_sdata : sd.SpatialData, optional
        If provided, write the transformation to this SpatialData object.
        
    Raises
    ------
    AssertionError
        If the existing transformations are not BaseTransformation instances.
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
    
    This function parses string representations of 2x3 affine transformation
    matrices, handling various formatting inconsistencies by normalizing
    whitespace and bracket placement. Useful when reading from csv.
    
    Parameters
    ----------
    M : str
        String representation of a 2x3 affine transformation matrix.
        Can contain various whitespace and formatting variations.
        
    Returns
    -------
    np.ndarray
        2x3 affine transformation matrix of shape (2, 3).
           
    Notes
    -----
    This function uses eval() which can be unsafe with untrusted input.
    It should only be used with trusted transformation matrix strings.
    """
    mstring = re.sub(r"\s+", ",", M)
    mstring = re.sub(r"\[\[,", "[[", mstring)
    mstring = re.sub(r"\],\[,", "],[", mstring)
    m = eval('np.array('+ mstring +')')
    return m


def multi_channel_corr(source: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Calculate correlation coefficients between all pairs of channels from two images.
    
    This function computes the Pearson correlation coefficient between every
    channel in the source image and every channel in the target image. The
    images are first normalized (z-scored) channel-wise before correlation
    calculation.
    
    Parameters
    ----------
    source : np.ndarray
        Source image array of shape (n_channels_1, n_pixels).
        Each row represents a flattened channel.
    target : np.ndarray
        Target image array of shape (n_channels_2, n_pixels).
        Each row represents a flattened channel.
        
    Returns
    -------
    np.ndarray
        Correlation matrix of shape (n_channels_1, n_channels_2).
        Element (i, j) contains the correlation between source channel i
        and target channel j.
    
    Notes
    -----
    The function assumes the input arrays have the same number of pixels
    (same second dimension). Channels are normalized to have zero mean
    and unit standard deviation before correlation calculation.
    """
    if (source.shape[1] == 0) or (target.shape[1] == 0): return np.nan
    if any(source.std(axis=1) == 0) or any(target.std(axis=1) == 0): return np.nan
    # Normalize the data
    source = (source - source.mean(axis=1)[:, None]) / source.std(axis=1)[:, None]
    target = (target - target.mean(axis=1)[:, None]) / target.std(axis=1)[:, None]
    # Calculate correlation matrix using matrix multiplication
    return np.dot(source, target.T) / source.shape[1]

class IFProcessor:
    """Preprocessing pipeline for Immunofluorescence (IF) images.
    
    This class provides a standardized preprocessing pipeline for IF images
    including channel summation, normalization, Gaussian blurring, and optional
    binarization. The processor can be configured with trial parameters from
    Optuna optimization.
    
    Parameters
    ----------
    binarize : bool, default=True
        Whether to apply binarization after preprocessing.
    binarization_threshold : float, default=0.1
        Threshold value for binarization (0-1 range after normalization).
    sigma : float, default=1
        Standard deviation for Gaussian blur kernel.
        
    Attributes
    ----------
    binarize : bool
        Whether binarization is enabled.
    binarization_threshold : float
        Current binarization threshold.
    sigma : float
        Current Gaussian blur sigma value.
    """
    
    def __init__(self, binarize=True, binarization_threshold=0.1, sigma=1):
        self.binarize = binarize
        self.binarization_threshold = binarization_threshold
        self.sigma = sigma
        
    def configure(self, trial_params):
        """Configure processor parameters from Optuna trial parameters.
        
        This method updates the processor parameters based on values suggested
        by an Optuna trial. It looks for specific parameter names in the trial
        params dictionary and updates the corresponding attributes.
        
        Parameters
        ----------
        trial_params : dict
            Dictionary of trial parameters from Optuna optimization.
            Expected keys: 'IF_binarization_threshold', 'IF_gaussian_sigma',
            'binarize_images'.
            
        Returns
        -------
        self : IFProcessor
            Returns self for method chaining.
        """
        if trial_params:
            # Extract IF-specific parameters from trial params
            if "IF_binarization_threshold" in trial_params:
                self.binarization_threshold = trial_params["IF_binarization_threshold"]
            if "IF_gaussian_sigma" in trial_params:
                self.sigma = trial_params["IF_gaussian_sigma"]
            if "binarize_images" in trial_params:
                self.binarize = trial_params["binarize_images"]            
        return self 
        
    def __call__(self, source_image):
        """Apply the preprocessing pipeline to an IF image.
        
        The preprocessing pipeline consists of:
        1. Sum all channels to create a single composite image
        2. Normalize to [0, 1] range by dividing by maximum value
        3. Apply Gaussian blur with specified sigma
        4. Optionally binarize using the threshold
        
        Parameters
        ----------
        source_image : np.ndarray
            Input IF image of shape (C, H, W) where C is number of channels.
            
        Returns
        -------
        np.ndarray
            Processed image of shape (H, W). If binarize=True, returns boolean
            array. Otherwise returns float array in [0, 1] range.
        """
        source_image = source_image.sum(0)
        source_image = source_image / source_image.max()
        source_image = gaussian(source_image, sigma=self.sigma)
        if self.binarize:
            source_image = source_image > self.binarization_threshold
        return source_image


class IMCProcessor:
    """Preprocessing pipeline for Imaging Mass Cytometry (IMC) images.
    
    This class provides a comprehensive preprocessing pipeline specifically
    designed for IMC images, including arcsinh transformation for variance
    stabilization, winsorization for outlier handling, normalization,
    Gaussian blurring, and optional binarization.
    
    Parameters
    ----------
    arcsinh_normalize : bool, default=True
        Whether to apply arcsinh transformation for variance stabilization.
    arcsinh_cofactor : float, default=5
        Cofactor for arcsinh transformation. Lower values increase the
        transformation strength.
    winsorize_limits : list of float, default=[None, None]
        Lower and upper percentile limits for winsorization.
        [0.01, 0.01] means clip bottom 1% and top 1% of values.
    binarize : bool, default=True
        Whether to apply binarization after preprocessing.
    binarization_threshold : float, default=2
        Threshold value for binarization.
    sigma : float, default=1
        Standard deviation for Gaussian blur kernel.
        
    Attributes
    ----------
    arcsinh_normalize : bool
        Whether arcsinh transformation is enabled.
    arcsinh_cofactor : float
        Current arcsinh cofactor value.
    winsorize_limits : list
        Current winsorization limits.
    binarize : bool
        Whether binarization is enabled.
    binarization_threshold : float
        Current binarization threshold.
    sigma : float
        Current Gaussian blur sigma value.
    
    Notes
    -----
    The arcsinh transformation is particularly useful for IMC data because it
    stabilizes variance across the intensity range, which is important for
    count-based mass spectrometry data.
    """
    
    def __init__(self, arcsinh_normalize=True, arcsinh_cofactor=5, winsorize_limits=[None, None], binarize=True, binarization_threshold=2, sigma=1):
        self.arcsinh_normalize = arcsinh_normalize
        self.arcsinh_cofactor = arcsinh_cofactor
        self.winsorize_limits = winsorize_limits
        self.binarize = binarize
        self.binarization_threshold = binarization_threshold
        self.sigma = sigma
        
    def configure(self, trial_params):
        """Configure processor parameters from Optuna trial parameters.
        
        This method updates the processor parameters based on values suggested
        by an Optuna trial. It looks for specific IMC parameter names in the
        trial params dictionary.
        
        Parameters
        ----------
        trial_params : dict
            Dictionary of trial parameters from Optuna optimization.
            Expected keys: 'IMC_arcsinh_normalize', 'IMC_arcsinh_cofactor',
            'IMC_winsorization_lower_limit', 'IMC_winsorization_upper_limit',
            'IMC_binarization_threshold', 'IMC_gaussian_sigma', 'binarize_images'.
            
        Returns
        -------
        self : IMCProcessor
            Returns self for method chaining.
        """
        if "IMC_arcsinh_normalize" in trial_params:
            self.arcsinh_normalize = trial_params["IMC_arcsinh_normalize"]
        if "IMC_arcsinh_cofactor" in trial_params:
            self.arcsinh_cofactor = trial_params["IMC_arcsinh_cofactor"]
        if "IMC_winsorization_lower_limit" in trial_params and "IMC_winsorization_upper_limit" in trial_params:
            self.winsorize_limits = [trial_params["IMC_winsorization_lower_limit"], trial_params["IMC_winsorization_upper_limit"]]
        if "IMC_binarization_threshold" in trial_params:
            self.binarization_threshold = trial_params["IMC_binarization_threshold"]
        if "IMC_gaussian_sigma" in trial_params:
            self.sigma = trial_params["IMC_gaussian_sigma"]
        if "binarize_images" in trial_params:
            self.binarize = trial_params["binarize_images"]
        return self
        
    def __call__(self, target_image):
        """Apply the IMC preprocessing pipeline to an image.
        
        The preprocessing pipeline consists of:
        1. Optional arcsinh transformation for variance stabilization
        2. Sum all channels to create composite image
        3. Winsorization to clip outlier intensities
        4. Normalize to [0, 1] range
        5. Apply Gaussian blur
        6. Optional binarization
        
        Parameters
        ----------
        target_image : np.ndarray
            Input IMC image of shape (C, H, W) where C is number of channels.
            
        Returns
        -------
        np.ndarray
            Processed image of shape (H, W). If binarize=True, returns boolean
            array. Otherwise returns float array in [0, 1] range.
        """
        if self.arcsinh_normalize:
            target_image = np.arcsinh(target_image/self.arcsinh_cofactor)
        target_image = target_image.sum(0)
        target_image = winsorize(target_image, limits=self.winsorize_limits)
        target_image = target_image / target_image.max()
        target_image = gaussian(target_image, sigma=self.sigma)
        if self.binarize:
            target_image = target_image > self.binarization_threshold
        return target_image


def pick_best_registration(study_df):
    """Calculate triangle score and return best trial from optimization results.
    
    This function implements a balanced scoring approach for selecting the best
    registration trial from a set of optimization results. It combines three
    key metrics (logical AND, IoU, and correlation) using a triangular scoring
    scheme that balances all three aspects of registration quality.
    
    The triangle score is calculated as:
    
    .. code-block:: text
    
        (1/3) * |norm_and * norm_corr + norm_corr * norm_iou + norm_iou * norm_and|
    
    where each metric is normalized to [0,1] within the group.
    
    Parameters
    ----------
    study_df : pd.DataFrame
        DataFrame containing trial results with required columns:
        - 'user_attrs_logical_and': Logical AND overlap between images
        - 'user_attrs_logical_iou': Intersection over Union score
        - 'user_attrs_reg_image_max_corr': Maximum correlation between channels
        
    Returns
    -------
    pd.Series
        DataFrame row containing the trial with the highest balanced score.
        The returned series includes all original columns plus computed
        normalization columns and the final 'balanced_score'.
        
    Notes
    -----
    The logical AND values are log-transformed before normalization because
    they typically span several orders of magnitude. The triangle score
    approach ensures that no single metric dominates the selection, leading
    to more robust registration quality assessment.
    
    Missing or NaN values in any of the required columns will result in
    NaN normalized scores, which may affect the final ranking.
    """
    study_df['norm_and'] = (np.log10(study_df['user_attrs_logical_and']+1)) / (np.log10(study_df['user_attrs_logical_and']+1).max())
    study_df['norm_iou'] = study_df['user_attrs_logical_iou'] / study_df['user_attrs_logical_iou'].max()
    study_df['norm_corr'] = study_df['user_attrs_reg_image_max_corr'] / study_df['user_attrs_reg_image_max_corr'].max()
    study_df['balanced_score'] = 1/3 * abs(study_df['norm_and'] * study_df['norm_corr'] + 
                                        study_df['norm_corr'] * study_df['norm_iou'] + 
                                        study_df['norm_iou'] * study_df['norm_and'])
    # Get the row with maximum triangle score
    best_row = study_df.loc[study_df['balanced_score'].idxmax()]
    return best_row
