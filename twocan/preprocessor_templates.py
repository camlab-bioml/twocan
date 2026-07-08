import numpy as np
from skimage.filters import gaussian
from scipy.stats.mstats import winsorize


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


class XEProcessor:
    """Preprocessing pipeline for Xenium IF images.
    
    This class provides a preprocessing pipeline for XE images.
    """
    
    def __init__(self):
        pass
        
    def __call__(self, source_image):
        return source_image

class IMSProcessor:
    """Preprocessing pipeline for Imaging Mass Spectrometry (IMS) images.
    
    This class provides a standardized preprocessing pipeline for IMS images
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
        by an Optuna trial. It looks for specific IMS parameter names in the 
        trial params dictionary and updates the corresponding attributes.
        
        Parameters
        ----------
        trial_params : dict
            Dictionary of trial parameters from Optuna optimization.
            Expected keys: 'IMS_binarization_threshold', 'IMS_gaussian_sigma',
            'binarize_images'.
            
        Returns
        -------
        self : IMSProcessor
            Returns self for method chaining.
        """
        if trial_params:
            # Extract IMS-specific parameters from trial params
            if "IMS_binarization_threshold" in trial_params:
                self.binarization_threshold = trial_params["IMS_binarization_threshold"]
            if "IMS_gaussian_sigma" in trial_params:
                self.sigma = trial_params["IMS_gaussian_sigma"]
            if "binarize_images" in trial_params:
                self.binarize = trial_params["binarize_images"]            
        return self 
        
    def __call__(self, source_image):
        """Apply the preprocessing pipeline to an IMS image.
        
        The preprocessing pipeline consists of:
        1. Sum all channels to create a single composite image
        2. Normalize to [0, 1] range by dividing by maximum value
        3. Apply Gaussian blur with specified sigma
        4. Optionally binarize using the threshold
        
        Parameters
        ----------
        source_image : np.ndarray
            Input IMS image of shape (C, H, W) where C is number of channels.
            
        Returns
        -------
        np.ndarray
            Processed image of shape (H, W). If binarize=True, returns boolean
            array. Otherwise returns float array in [0, 1] range.
        """
        source_image = source_image.sum(0)
        # Prevent division by zero if an empty patch/channel is passed
        max_val = source_image.max()
        if max_val > 0:
            source_image = source_image / max_val
        source_image = gaussian(source_image, sigma=self.sigma)
        if self.binarize:
            source_image = source_image > self.binarization_threshold
            
        return source_image