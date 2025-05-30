"""Tests for utility functions."""
import pytest
import numpy as np
import pandas as pd
import spatialdata as sd
from twocan.utils import (
    stretch_255,
    prep_zarr,
    read_M,
    multi_channel_corr,
    IFProcessor,
    IMCProcessor,
    pick_best_registration
)

class TestStretch255:
    def test_normal_case(self):
        """Test normal case with non-zero values."""
        img = np.array([[0, 1, 2], [3, 4, 5]], dtype=np.float32)
        result = stretch_255(img)
        assert result.dtype == np.uint8
        assert result.max() == 255
        assert result.min() == 0
    
    def test_zero_array(self):
        """Test array with all zeros."""
        img = np.zeros((2, 2))
        result = stretch_255(img)
        assert np.array_equal(result, img)
    
    def test_negative_values(self):
        """Test array with negative values."""
        img = np.array([[-1, 0, 1], [2, 3, 4]], dtype=np.float32)
        result = stretch_255(img)
        assert result.dtype == np.uint8
        assert result.max() == 255
        assert result.min() == 0

class TestPrepZarr:
    @pytest.fixture
    def sample_data(self):
        if_arr = np.random.rand(3, 100, 100)
        imc_arr = np.random.rand(3, 100, 100)
        if_panel = ['DAPI', 'CD3', 'CD20']
        imc_panel = ['DAPI', 'CD3', 'CD20']
        return if_arr, imc_arr, if_panel, imc_panel
    
    def test_prep_zarr(self, sample_data):
        """Test SpatialData creation."""
        if_arr, imc_arr, if_panel, imc_panel = sample_data
        sdata = prep_zarr(if_arr, imc_arr, if_panel, imc_panel)
        
        assert isinstance(sdata, sd.SpatialData)
        assert 'IF' in sdata.images
        assert 'IMC' in sdata.images
        assert sdata.images['IF'].shape == if_arr.shape
        assert sdata.images['IMC'].shape == imc_arr.shape


class TestReadM:
    def test_read_M(self):
        """Test matrix string parsing."""
        M_str = "[[ 1 0 0 ] [ 0 1 0 ]]"
        M = read_M(M_str)
        assert M.shape == (2, 3)
        assert np.array_equal(M, np.array([[1, 0, 0], [0, 1, 0]]))

class TestMultiChannelCorr:
    def test_multi_channel_corr(self):
        """Test correlation calculation."""
        source = np.random.rand(3, 100)
        target = np.random.rand(2, 100)
        corr = multi_channel_corr(source, target)
        
        assert corr.shape == (3, 2)
        assert np.all(np.abs(corr) <= 1)

class TestIFProcessor:
    def test_if_processor_init(self):
        """Test IFProcessor initialization."""
        processor = IFProcessor(binarize=True, binarization_threshold=0.2, sigma=2)
        assert processor.binarize == True
        assert processor.binarization_threshold == 0.2
        assert processor.sigma == 2
    
    def test_if_processor_configure(self):
        """Test IFProcessor parameter configuration."""
        processor = IFProcessor()
        trial_params = {
            'IF_binarization_threshold': 0.3,
            'IF_gaussian_sigma': 1.5,
            'binarize_images': False
        }
        processor.configure(trial_params)
        assert processor.binarization_threshold == 0.3
        assert processor.sigma == 1.5
        assert processor.binarize == False
    
    def test_if_processor_call(self):
        """Test IFProcessor image processing."""
        processor = IFProcessor(binarize=False, sigma=1)
        img = np.random.rand(3, 100, 100)
        result = processor(img)
        
        assert result.shape == (100, 100)
        assert result.max() <= 1
        assert result.min() >= 0

class TestIMCProcessor:
    def test_imc_processor_init(self):
        """Test IMCProcessor initialization."""
        processor = IMCProcessor(arcsinh_normalize=False, binarization_threshold=1.5)
        assert processor.arcsinh_normalize == False
        assert processor.binarization_threshold == 1.5
    
    def test_imc_processor_configure(self):
        """Test IMCProcessor parameter configuration."""
        processor = IMCProcessor()
        trial_params = {
            'IMC_arcsinh_cofactor': 10,
            'IMC_binarization_threshold': 3,
            'IMC_gaussian_sigma': 0.5
        }
        processor.configure(trial_params)
        assert processor.arcsinh_cofactor == 10
        assert processor.binarization_threshold == 3
        assert processor.sigma == 0.5
    
    def test_imc_processor_call(self):
        """Test IMCProcessor image processing."""
        processor = IMCProcessor(binarize=False, sigma=1)
        img = np.random.rand(3, 100, 100) + 1  # Add 1 to avoid zeros for arcsinh
        result = processor(img)
        
        assert result.shape == (100, 100)
        assert result.max() <= 1
        assert result.min() >= 0

class TestPickBestRegistration:
    def test_pick_best_registration(self):
        """Test best registration selection."""
        study_df = pd.DataFrame({
            'user_attrs_logical_and': [0.8, 0.7, 0.9],
            'user_attrs_logical_iou': [0.7, 0.8, 0.6],
            'user_attrs_reg_image_max_corr': [0.9, 0.6, 0.7]
        })
        best = pick_best_registration(study_df)
        assert isinstance(best, pd.Series)
        assert all(k in best for k in ['user_attrs_logical_and', 'user_attrs_logical_iou', 'user_attrs_reg_image_max_corr'])
        assert 'balanced_score' in best 