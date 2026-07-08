"""
Core registration estimator for Twocan multimodal image registration.
"""

from typing import Optional, Tuple, Dict
import cv2
import numpy as np
from abc import ABC
from skimage import transform
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted
from .utils import stretch_255


class RegEstimator(TransformerMixin, BaseEstimator, ABC):
    def __init__(
        self,
        registration_max_features: int = 10000,
        registration_percentile: float = 0.9,
        feature_method: str = "orb",
        transform_model: str = "affine_partial",
        match_filter: str = "top_percentile",
        ratio_test_threshold: float = 0.75,
        ransac_reproj_threshold: float = 3.0,
        min_matches: int = 10,
    ):
        self.registration_max_features = registration_max_features
        self.registration_percentile = registration_percentile
        self.feature_method = feature_method
        self.transform_model = transform_model
        self.match_filter = match_filter
        self.ratio_test_threshold = ratio_test_threshold
        self.ransac_reproj_threshold = ransac_reproj_threshold
        self.min_matches = min_matches

    def set_M(self, M: np.ndarray) -> "RegEstimator":
        self.M_ = M
        self.warp_kind_ = "matrix"
        return self

    def _build_detector(self):
        method = self.feature_method.lower()
        if method == "orb":
            return cv2.ORB_create(self.registration_max_features, fastThreshold=0, edgeThreshold=0)
        if method == "sift":
            if not hasattr(cv2, "SIFT_create"):
                raise RuntimeError("SIFT is not available in this OpenCV build.")
            return cv2.SIFT_create(nfeatures=self.registration_max_features)
        raise ValueError("Unknown feature_method: {}".format(self.feature_method))

    def _build_matcher(self):
        method = self.feature_method.lower()
        if method == "orb":
            return cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=(self.match_filter == "top_percentile"))
        if method == "sift":
            return cv2.BFMatcher(cv2.NORM_L2, crossCheck=(self.match_filter == "top_percentile"))
        raise ValueError("Unknown feature_method: {}".format(self.feature_method))

    def _filter_matches(self, matcher, descsA, descsB):
        mode = self.match_filter.lower()
        if mode == "top_percentile":
            matches = matcher.match(descsA, descsB)
            matches = sorted(matches, key=lambda x: x.distance)
            keep = max(1, int(len(matches) * self.registration_percentile))
            return matches[:keep]
        if mode == "ratio_test":
            knn = matcher.knnMatch(descsA, descsB, k=2)
            return [m for m, n in knn if m.distance < self.ratio_test_threshold * n.distance]
        raise ValueError("Unknown match_filter: {}".format(self.match_filter))

    def fit(self, X: np.ndarray, y: np.ndarray) -> "RegEstimator":
        X = stretch_255(X.copy())
        y = stretch_255(y.copy())

        detector = self._build_detector()
        kpsA, descsA = detector.detectAndCompute(X, None)
        kpsB, descsB = detector.detectAndCompute(y, None)
        if descsA is None or descsB is None:
            raise RuntimeError("Feature detection failed: descriptors are missing.")

        matcher = self._build_matcher()
        matches = self._filter_matches(matcher, descsA, descsB)
        if len(matches) < self.min_matches:
            raise RuntimeError(
                "Insufficient matches after filtering: {} < {}".format(len(matches), self.min_matches)
            )

        ptsA = np.array([kpsA[m.queryIdx].pt for m in matches], dtype=np.float32)
        ptsB = np.array([kpsB[m.trainIdx].pt for m in matches], dtype=np.float32)

        model = self.transform_model.lower()
        self.inlier_mask_ = None
        if model == "affine_partial":
            M, mask = cv2.estimateAffinePartial2D(
                ptsA, ptsB, method=cv2.RANSAC, ransacReprojThreshold=self.ransac_reproj_threshold
            )
            if M is None:
                raise RuntimeError("estimateAffinePartial2D failed to estimate a transform.")
            self.M_ = M
            self.inlier_mask_ = mask
            self.warp_kind_ = "matrix"
        elif model == "affine":
            M, mask = cv2.estimateAffine2D(
                ptsA, ptsB, method=cv2.RANSAC, ransacReprojThreshold=self.ransac_reproj_threshold
            )
            if M is None:
                raise RuntimeError("estimateAffine2D failed to estimate a transform.")
            self.M_ = M
            self.inlier_mask_ = mask
            self.warp_kind_ = "matrix"
        elif model == "homography":
            M, mask = cv2.findHomography(ptsA, ptsB, cv2.RANSAC, self.ransac_reproj_threshold)
            if M is None:
                raise RuntimeError("findHomography failed to estimate a transform.")
            self.M_ = M
            self.inlier_mask_ = mask
            self.warp_kind_ = "matrix"
        elif model == "tps":
            if not hasattr(cv2, "createThinPlateSplineShapeTransformer"):
                raise RuntimeError("TPS is not available in this OpenCV build.")
            tps = cv2.createThinPlateSplineShapeTransformer()
            src_shape = ptsA.reshape(1, -1, 2)
            dst_shape = ptsB.reshape(1, -1, 2)
            tps_matches = [cv2.DMatch(i, i, 0) for i in range(ptsA.shape[0])]
            ok = tps.estimateTransformation(src_shape, dst_shape, tps_matches)
            if ok is False:
                raise RuntimeError("TPS estimation failed.")
            self.tps_ = tps
            self.warp_kind_ = "tps"
            self.M_ = np.eye(3, dtype=np.float32)
        else:
            raise ValueError("Unknown transform_model: {}".format(self.transform_model))

        self.y_shape_ = y.shape[-2:]
        return self

    def transform(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> np.ndarray:
        check_is_fitted(self)
        input_was_2d = X.ndim == 2
        if input_was_2d:
            X = X[None, :, :]
        y_shape = None if y is None else y.shape[-2:]
        output_shape = y_shape or self.y_shape_

        if self.warp_kind_ == "tps":
            X_mv = []
            for x in X:
                warped = self.tps_.warpImage(x.astype(np.float32))
                if warped.shape != tuple(output_shape):
                    warped = cv2.resize(warped, (output_shape[1], output_shape[0]))
                X_mv.append(warped)
            X_mv = np.stack(X_mv)
        else:
            if self.M_.shape == (2, 3):
                t = transform.AffineTransform(matrix=np.vstack([self.M_, np.array([0, 0, 1])]))
            elif self.M_.shape == (3, 3):
                t = transform.ProjectiveTransform(matrix=self.M_)
            else:
                raise RuntimeError("Unsupported matrix shape: {}".format(self.M_.shape))
            X_mv = np.stack([transform.warp(x, inverse_map=t.inverse, output_shape=output_shape) for x in X])

        if X_mv.ndim == 2:
            X_mv = X_mv[None, :, :]
        if input_was_2d:
            X_mv = X_mv[0]
        if y is not None:
            if y.ndim == 2:
                y = y[None, :, :]
            if input_was_2d:
                X_mv = X_mv[None, :, :]
            return np.vstack([X_mv, y])
        return X_mv

    def fit_transform(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        self.fit(X, y)
        return self.transform(X, y)

    def score(self, source: np.ndarray, target: np.ndarray) -> Dict[str, float]:
        assert source.ndim == 2
        assert target.ndim == 2
        stack = self.transform(source, target)
        source_mask = np.ones(source.shape)
        target_mask = np.ones(target.shape)
        stack_mask = self.transform(source_mask, target_mask).sum(0) > 1
        stack = stack[:, stack_mask]

        logical_and = np.logical_and(stack[0], stack[1])
        logical_or = np.logical_or(stack[0], stack[1])
        logical_xor = np.logical_xor(stack[0], stack[1])

        and_sum = logical_and.sum()
        or_sum = logical_or.sum()
        xor_sum = logical_xor.sum()
        iou = (and_sum / or_sum) if or_sum > 0 else 0.0

        return {
            "and": and_sum,
            "or": or_sum,
            "xor": xor_sum,
            "iou": iou,
            "source_sum": stack[0].sum(),
            "target_sum": stack[1].sum(),
        }
