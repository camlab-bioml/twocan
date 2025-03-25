from typing import Optional, Tuple, Dict, Union, Any
import cv2
import numpy as np
from abc import ABC
from skimage import transform
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted
from .utils import stretch_255

class RegEstimator(TransformerMixin, BaseEstimator, ABC):
    '''A scikit-learn compatible estimator for multimodal image registration.
    
    This class implements feature-based image registration using OpenCV's ORB detector
    and a partial affine transformation model. It follows scikit-learn's estimator API
    with fit, transform, and fit_transform methods.
    
    Parameters
    ----------
    registration_max_features : int, default=10000
        Maximum number of features to detect in each image using ORB.
    registration_percentile : float, default=0.9
        Percentile of features to keep after sorting by match quality (0-1).
        
    Attributes
    ----------
    M_ : np.ndarray
        The estimated 2x3 affine transformation matrix after fitting.
    y_shape_ : Tuple[int, int]
        Shape of the target image used during fitting.
        
    Examples
    --------
    >>> reg = RegEstimator(registration_max_features=10000)
    >>> reg.fit(source_image, target_image)
    >>> registered_image = reg.transform(source_image)
    '''
    def __init__(self, registration_max_features: int = 10000, registration_percentile: float = 0.9):
        self.registration_max_features = registration_max_features
        self.registration_percentile = registration_percentile
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> 'RegEstimator':
        '''Estimate the affine transformation matrix between source (X) and target (y) images.
        
        Parameters
        ----------
        X : np.ndarray
            Source image to be registered, shape (H, W) or (1, H, W).
        y : np.ndarray
            Target image to register to, shape (H, W) or (1, H, W).
            
        Returns
        -------
        self : RegEstimator
            The fitted estimator.
            
        Raises
        ------
        cv2.error
            If affine transformation cannot be estimated.
        '''
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
        """Apply the estimated transformation to the source image.
        
        Parameters
        ----------
        X : np.ndarray
            Source image(s) to transform, shape (H, W) or (C, H, W).
        y : Optional[np.ndarray], default=None
            If provided, will be stacked below the transformed X without transformation.
            
        Returns
        -------
        np.ndarray
            Transformed image(s), with shape matching input but potentially different H, W.
            If y is provided, output includes stacked y channels.
            
        Raises
        ------
        NotFittedError
            If transform is called before fitting.
        AssertionError
            If transformation matrix is invalid.
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
        
        Equivalent to fit(X, y).transform(X, y).
        
        Parameters
        ----------
        X : np.ndarray
            Source image to fit and transform.
        y : np.ndarray
            Target image to fit to.
            
        Returns
        -------
        np.ndarray
            The transformed source image stacked with the target.
        """
        self.fit(X,y)
        return self.transform(X,y)
    
    def score(self, source: np.ndarray, target: np.ndarray) -> Dict[str, float]:
        """Calculate registration quality metrics between source and target images.
        
        Parameters
        ----------
        source : np.ndarray
            Source image, shape (H, W).
        target : np.ndarray
            Target image, shape (H, W).
            
        Returns
        -------
        Dict[str, float]
            Dictionary containing registration metrics:
            - 'and': Logical AND of overlapping regions
            - 'or': Logical OR of overlapping regions
            - 'xor': Logical XOR of overlapping regions
            - 'iou': Intersection over Union
            - 'source_sum': Sum of source image pixels
            - 'target_sum': Sum of target image pixels
        """
        assert source.ndim==2
        assert target.ndim==2
        stack = self.transform(source,target)
        # restrict to shared area of union 
        source_mask = np.ones(source.shape)
        target_mask = np.ones(target.shape)
        stack_mask = self.transform(source_mask,target_mask).sum(0) >1
        stack = stack[:,stack_mask]
        return({
            'and': (np.logical_and(stack[0], stack[1])).sum(),
            'or': (np.logical_or(stack[0], stack[1])).sum(),
            'xor': (np.logical_xor(stack[0], stack[1])).sum(),
            'iou': (np.logical_and(stack[0], stack[1])).sum() / (np.logical_or(stack[0], stack[1])).sum(),
            'source_sum': stack[0].sum(),
            'target_sum': stack[1].sum()
        })
    

