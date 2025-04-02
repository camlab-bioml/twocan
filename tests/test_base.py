"""Tests for the base RegEstimator class."""
import pytest
import numpy as np
from twocan.base import RegEstimator

class TestRegEstimator:
    @pytest.fixture
    def estimator(self):
        return RegEstimator(registration_max_features=1000, registration_percentile=0.8)
    
    @pytest.fixture
    def sample_images(self):
        # Create simple test images with known features
        source = np.zeros((100, 100), dtype=np.float32)
        target = np.zeros((100, 100), dtype=np.float32)
        
        # Add larger, overlapping features
        source[20:50, 20:50] = 1  # 30x30 square
        source[50:80, 50:80] = 1  # 30x30 square
        target[30:60, 30:60] = 1  # 30x30 square, shifted right and down
        target[60:90, 60:90] = 1  # 30x30 square, shifted right and down
        
        return source, target
    
    @pytest.fixture
    def non_overlapping_images(self):
        # Create test images with no overlap
        source = np.zeros((100, 100), dtype=np.float32)
        target = np.zeros((100, 100), dtype=np.float32)
        
        # Add features in opposite corners
        source[10:30, 10:30] = 1  # Top left
        target[70:90, 70:90] = 1  # Bottom right
        
        return source, target
    
    def test_initialization(self, estimator):
        """Test estimator initialization."""
        assert estimator.registration_max_features == 1000
        assert estimator.registration_percentile == 0.8
        assert not hasattr(estimator, 'M_')
        assert not hasattr(estimator, 'y_shape_')
    
    def test_fit(self, estimator, sample_images):
        """Test fitting the estimator."""
        source, target = sample_images
        estimator.fit(source, target)
        
        assert hasattr(estimator, 'M_')
        assert hasattr(estimator, 'y_shape_')
        assert estimator.M_.shape == (2, 3)
        assert estimator.y_shape_ == target.shape
    
    def test_transform_before_fit(self, estimator, sample_images):
        """Test transform raises error before fitting."""
        source, _ = sample_images
        with pytest.raises(Exception):
            estimator.transform(source)
    
    def test_transform(self, estimator, sample_images):
        """Test image transformation."""
        source, target = sample_images
        estimator.fit(source, target)
        
        transformed = estimator.transform(source)
        assert transformed.shape[1:] == target.shape
    
    def test_fit_transform(self, estimator, sample_images):
        """Test fit_transform method."""
        source, target = sample_images
        transformed = estimator.fit_transform(source, target)
        
        assert transformed.shape[0] == 2  # Stacked source and target
        assert transformed.shape[1:] == target.shape
    
    def test_score(self, estimator, sample_images):
        """Test registration quality metrics."""
        source, target = sample_images
        estimator.fit(source, target)
    
        metrics = estimator.score(source, target)
        assert isinstance(metrics, dict)
        assert all(k in metrics for k in ['and', 'or', 'xor', 'iou', 'source_sum', 'target_sum'])
        assert 0 <= metrics['iou'] <= 1
    
    def test_score_no_overlap(self, estimator, non_overlapping_images):
        """Test registration quality metrics when images have no overlap."""
        source, target = non_overlapping_images
        estimator.M_ = np.array([[1, 0, 0], [0, 1, 0]]) # fake no overlap
        metrics = estimator.score(source, target)
        assert isinstance(metrics, dict)
        assert all(k in metrics for k in ['and', 'or', 'xor', 'iou', 'source_sum', 'target_sum'])
        assert metrics['iou'] == 0  # IoU should be 0 when there's no overlap
        assert metrics['and'] == 0  # No intersection
        assert metrics['or'] > 0  # Union should be non-zero
        assert metrics['xor'] > 0  # XOR should be non-zero
    
    def test_invalid_inputs(self, estimator):
        """Test handling of invalid inputs."""
        # Test with non-2D arrays
        with pytest.raises(AssertionError):
            estimator.score(np.zeros((1, 100, 100)), np.zeros((100, 100)))
        
        # Test with different shapes
        with pytest.raises(Exception):
            estimator.fit(np.zeros((100, 100)), np.zeros((200, 200)))
    
    def test_empty_images(self, estimator):
        """Test handling of empty or zero images."""
        empty = np.zeros((100, 100))
        with pytest.raises(Exception):
            estimator.fit(empty, empty)
    
    def test_extreme_values(self, estimator):
        """Test handling of extreme values."""
        source = np.random.rand(100, 100) * 1e6
        target = np.random.rand(100, 100) * 1e6
        estimator.fit(source, target)
        transformed = estimator.transform(source)
        assert not np.any(np.isnan(transformed))
        assert not np.any(np.isinf(transformed)) 