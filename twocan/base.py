"""
Core registration estimator for Twocan multimodal image registration.

This module implements the main RegEstimator class, which provides a 
scikit-learn compatible interface for feature-based image registration 
using OpenCV's ORB detector and affine transformation estimation.
"""

from typing import Optional, Tuple, Dict, Union, Any
import cv2
import numpy as np
from abc import ABC
from skimage import transform
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted
from .utils import stretch_255

class RegEstimator(TransformerMixin, BaseEstimator, ABC):
    """A scikit-learn compatible estimator for multimodal image registration.
    
    This class implements feature-based image registration using OpenCV's ORB 
    (Oriented FAST and Rotated BRIEF) detector and a partial affine transformation 
    model. It follows scikit-learn's estimator API with fit, transform, and 
    fit_transform methods, making it easy to integrate into machine learning pipelines.
    
    The registration process consists of:
    1. Feature detection using ORB on both images
    2. Feature matching using brute-force Hamming distance
    3. Affine transformation estimation using RANSAC
    4. Image transformation using the estimated parameters
    
    Parameters
    ----------
    registration_max_features : int, default=10000
        Maximum number of features to detect in each image using ORB.
        Higher values can improve registration accuracy but increase computation time.
    registration_percentile : float, default=0.9
        Percentile of features to keep after sorting by match quality (0-1).
        Only the top percentile of matches by distance are used for transformation
        estimation, which helps remove outliers.
        
    Attributes
    ----------
    M_ : np.ndarray
        The estimated 2x3 affine transformation matrix after fitting.
        Shape is (2, 3) representing the transformation [R|t] where R is
        rotation/scaling and t is translation.
    y_shape_ : Tuple[int, int]
        Shape (height, width) of the target image used during fitting.
        Used as the default output shape for transformations.
    
    Notes
    -----
    The estimator automatically converts input images to 8-bit grayscale for
    feature detection using the stretch_255 utility function. This ensures
    consistent feature detection regardless of input image dynamic range.
    
    The partial affine transformation model allows rotation, scaling, and 
    translation but not shearing, which is appropriate for most microscopy
    registration tasks where imaging geometry is approximately preserved.
    
    For best results:
    - Ensure sufficient overlap between images
    - Use images with distinct features (not uniform regions)
    - Consider preprocessing to enhance relevant structures
    - Adjust max_features based on image complexity and computational budget
    """
    
    def __init__(self, registration_max_features: int = 10000, registration_percentile: float = 0.9):
        self.registration_max_features = registration_max_features
        self.registration_percentile = registration_percentile
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> 'RegEstimator':
        """Estimate the affine transformation matrix between source (X) and target (y) images.
        
        This method detects features in both images using ORB, matches them, and
        estimates the best affine transformation that maps source features to
        target features using OpenCV's robust estimation.
        
        Parameters
        ----------
        X : np.ndarray
            Source image to be registered. Shape can be (H, W) for single-channel
            or (C, H, W) for multi-channel. If multi-channel, all channels are
            summed for feature detection.
        y : np.ndarray
            Target (reference) image to register to. Shape can be (H, W) for 
            single-channel or (C, H, W) for multi-channel. If multi-channel,
            all channels are summed for feature detection.
            
        Returns
        -------
        self : RegEstimator
            The fitted estimator with estimated transformation matrix in ``self.M_``.
            
        Raises
        ------
        cv2.error
            If affine transformation cannot be estimated, typically due to
            insufficient or poorly matched features.
            
        Notes
        -----
        The fitting process:
        1. Convert images to 8-bit for ORB compatibility
        2. Detect up to max_features keypoints in each image
        3. Compute ORB descriptors for each keypoint
        4. Match descriptors using brute-force Hamming distance
        5. Keep top percentile of matches by distance
        6. Estimate partial affine transformation using RANSAC
        
        The method uses OpenCV's estimateAffinePartial2D which finds the optimal
        similarity transformation (rotation, scaling, translation) rather than
        a full affine transformation. This is more robust for most registration
        scenarios.
        """
        X = stretch_255(X.copy())
        y = stretch_255(y.copy())
        # orb detector
        orb = cv2.ORB_create(self.registration_max_features, fastThreshold=0, edgeThreshold=0)
        (kpsA, descsA) = orb.detectAndCompute(X, None)
        (kpsB, descsB) = orb.detectAndCompute(y, None)
        # match the features
        method = cv2.DESCRIPTOR_MATCHER_BRUTEFORCE_HAMMING
        matcher = cv2.DescriptorMatcher_create(method)
        matches = matcher.match(descsA, descsB, None)
        # sort the matches by their distance (the smaller the distance, the "more similar" the features are)
        matches = sorted(matches, key=lambda x:x.distance)
        # keep only the top percentile matches
        keep = int(len(matches) * self.registration_percentile)
        matches = matches[:keep]
        ptsA = np.array([kpsA[m.queryIdx].pt for m in matches])
        ptsB = np.array([kpsB[m.trainIdx].pt for m in matches])
        # register
        M, _mask = cv2.estimateAffinePartial2D(ptsA, ptsB)
        self.M_ = M
        self.y_shape_ = y.shape
        return self
    
    def transform(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> np.ndarray:
        """Apply the estimated transformation to the source image(s).
        
        This method transforms the source image(s) using the affine transformation
        estimated during fitting. Optionally, a target image can be provided which
        will be stacked with the transformed source without transformation.
        
        Parameters
        ----------
        X : np.ndarray
            Source image(s) to transform. Shape can be (H, W) for single-channel
            or (C, H, W) for multi-channel. All channels are transformed using
            the same transformation matrix.
        y : Optional[np.ndarray], default=None
            Target image to stack with transformed source. If provided, this image
            is NOT transformed but is included in the output for direct comparison.
            Shape should be (H, W) or (C, H, W).
            
        Returns
        -------
        np.ndarray
            Transformed image(s). If y is None, returns transformed X with shape
            (C, H_out, W_out) where H_out, W_out match the target image from fitting.
            If y is provided, returns stacked array with transformed X channels
            followed by untransformed y channels.
            
        Raises
        ------
        NotFittedError
            If transform is called before fitting the estimator.
        AssertionError
            If the stored transformation matrix has invalid shape.
            
        Notes
        -----
        The transformation uses scikit-image's warp function with the inverse
        transformation matrix. This ensures proper interpolation and handles
        edge cases automatically.
        
        Output image dimensions match the target image used during fitting
        unless a different y image is provided during transformation.
        """
        check_is_fitted(self)
        assert self.M_.shape == (2,3)
        if X.ndim==2: X = X[None,:,:]
        y_shape = None if y is None else y.shape[-2:]
        t = transform.AffineTransform(matrix=np.vstack([self.M_, np.array([0,0,1])]))
        X_mv = np.stack([transform.warp(x, inverse_map=t.inverse, output_shape=(y_shape or self.y_shape_)) for x in X])
        if X_mv.ndim==2: X_mv = X_mv[None,:,:]
        if y is not None:
            if y.ndim==2: y = y[None,:,:]
            return np.vstack([X_mv,y])
        return X_mv

    def fit_transform(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Fit to data, then transform it.
        
        This convenience method combines fitting and transformation in a single
        call. It's equivalent to calling fit(X, y).transform(X, y) but slightly
        more efficient.
        
        Parameters
        ----------
        X : np.ndarray
            Source image to fit the transformation to and then transform.
        y : np.ndarray
            Target image to fit the transformation against. This image will
            also be included in the output stack.
            
        Returns
        -------
        np.ndarray
            Stacked array containing the transformed X channels followed by
            the original y channels. Shape is (C_x + C_y, H_y, W_y) where
            C_x, C_y are the channel counts and H_y, W_y are target dimensions.
        """
        self.fit(X,y)
        return self.transform(X,y)
    
    def score(self, source: np.ndarray, target: np.ndarray) -> Dict[str, float]:
        """Calculate registration quality metrics between source and target images.
        
        This method computes various metrics to assess the quality of registration
        between binary or continuous-valued images. Metrics are calculated only
        in regions where both images have valid data after transformation.
        
        Parameters
        ----------
        source : np.ndarray
            Source image, shape (H, W). Should be the same image used for fitting
            or a similar image from the same modality.
        target : np.ndarray
            Target image, shape (H, W). Should be the same image used for fitting
            or a similar image from the same modality.
            
        Returns
        -------
        Dict[str, float]
            Dictionary containing registration quality metrics:
            
            - 'and' : float
                Count of pixels where both source and target are positive
                (logical AND operation). Higher values indicate better overlap.
            - 'or' : float
                Count of pixels where either source or target is positive
                (logical OR operation). 
            - 'xor' : float
                Count of pixels where source and target disagree
                (logical XOR operation). Lower values indicate better agreement.
            - 'iou' : float
                Intersection over Union ratio (and/or). Values range from 0-1
                with 1 indicating perfect overlap. Returns 0.0 if no positive pixels exist.
            - 'source_sum' : float
                Sum of all source pixel intensities in the overlap region.
            - 'target_sum' : float
                Sum of all target pixel intensities in the overlap region.
                
        Notes
        -----
        Metrics are computed only in the intersection region where both images
        have valid data after transformation. This ensures fair comparison and
        avoids edge effects from the transformation.
        
        For binary images, the metrics have intuitive interpretations:
        - IoU is the standard Jaccard index
        - 'and' counts overlapping positive pixels
        - 'xor' counts disagreement pixels
        
        For continuous images, the logical operations are applied after
        implicit conversion to boolean (non-zero values are True).
        """
        assert source.ndim==2
        assert target.ndim==2
        stack = self.transform(source,target)
        # restrict to shared area of union 
        source_mask = np.ones(source.shape)
        target_mask = np.ones(target.shape)
        stack_mask = self.transform(source_mask,target_mask).sum(0) >1
        stack = stack[:,stack_mask]
        
        # Calculate logical operations
        logical_and = np.logical_and(stack[0], stack[1])
        logical_or = np.logical_or(stack[0], stack[1])
        logical_xor = np.logical_xor(stack[0], stack[1])
        
        and_sum = logical_and.sum()
        or_sum = logical_or.sum()
        xor_sum = logical_xor.sum()
        
        # Handle divide by zero case for IoU
        iou = (and_sum / or_sum) if or_sum > 0 else 0.0
        
        return({
            'and': and_sum,
            'or': or_sum,
            'xor': xor_sum,
            'iou': iou,
            'source_sum': stack[0].sum(),
            'target_sum': stack[1].sum()
        })
    

