"""Tests for the workflow module."""
import pytest
import numpy as np
import pandas as pd
from pathlib import Path
import tempfile
import tifffile
import optuna
from typing import Dict, Any

from twocan.workflow import (
    ImageLoader,
    RegistrationOptimizer,
    TiffImageLoader,
    DefaultRegistrationOptimizer,
    run_registration_workflow
)


class MockImageLoader(ImageLoader):
    """Mock image loader that returns predefined arrays."""
    def __init__(self, mock_data: Dict[str, np.ndarray]):
        self.mock_data = mock_data
        
    def load(self, path: str) -> np.ndarray:
        return self.mock_data[str(path)]


class MockOptimizer(RegistrationOptimizer):
    """Mock optimizer that returns fixed parameters and preprocessed images."""
    def suggest_parameters(self, trial: optuna.trial.Trial) -> Dict[str, Any]:
        return {
            "IF_binarization_threshold": 0.5,
            "IF_gaussian_sigma": 1.0,
            "IMC_binarization_threshold": 0.5,
            "IMC_gaussian_sigma": 1.0,
            "binarize_images": True,
            "registration_target": "IF"
        }
    
    def preprocess(self, images: Dict[str, Any], params: Dict[str, Any]) -> Dict[str, Any]:
        return {
            'IF': (images['IF'] > params['IF_binarization_threshold']).astype(float),
            'IMC': (images['IMC'] > params['IMC_binarization_threshold']).astype(float)
        }


def test_tiff_image_loader():
    """Test TiffImageLoader with a temporary TIFF file."""
    # Create test data
    test_data = np.random.rand(10, 100, 100)
    
    with tempfile.NamedTemporaryFile(suffix='.tiff') as tmp:
        # Save test data
        tifffile.imwrite(tmp.name, test_data)
        
        # Test loading
        loader = TiffImageLoader()
        loaded_data = loader.load(tmp.name)
        
        # Check data was loaded correctly
        assert np.array_equal(test_data, loaded_data)


def test_default_registration_optimizer():
    """Test DefaultRegistrationOptimizer parameter suggestions and preprocessing."""
    optimizer = DefaultRegistrationOptimizer()
    
    # Create mock trial
    study = optuna.create_study()
    trial = optuna.trial.Trial(study, study._storage.create_new_trial(study._study_id))
    
    # Test parameter suggestions
    params = optimizer.suggest_parameters(trial)
    
    # Check required parameters are present
    required_params = [
        "IF_binarization_threshold",
        "IF_gaussian_sigma",
        "IMC_arcsinh_normalize",
        "IMC_arcsinh_cofactor",
        "IMC_winsorization_lower_limit",
        "IMC_winsorization_upper_limit",
        "IMC_binarization_threshold",
        "IMC_gaussian_sigma",
        "binarize_images",
        "registration_max_features",
        "registration_percentile",
        "registration_target"
    ]
    for param in required_params:
        assert param in params
        
    # Test preprocessing with mock images
    images = {
        'IF': np.random.rand(3, 100, 100),
        'IMC': np.random.rand(3, 100, 100)
    }
    
    processed = optimizer.preprocess(images, params)
    assert 'IF' in processed
    assert 'IMC' in processed
    assert processed['IF'].shape == (100, 100)  # Preprocessed to 2D
    assert processed['IMC'].shape == (100, 100)  # Preprocessed to 2D


def test_run_registration_workflow():
    """Test the complete registration workflow with mock components."""
    # Create mock data
    mock_if = np.random.rand(3, 100, 100)
    mock_imc = np.random.rand(3, 100, 100)
    
    # Create temporary files
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        
        # Save mock images
        if_path = tmpdir / "if_image.tiff"
        imc_path = tmpdir / "imc_image.tiff"
        tifffile.imwrite(if_path, mock_if)
        tifffile.imwrite(imc_path, mock_imc)
        
        # Create mock panel files
        if_panel = pd.DataFrame(['DAPI', 'CD3', 'CD20'])
        imc_panel = pd.DataFrame(['DAPI', 'CD3', 'CD20'])
        if_panel_path = tmpdir / "if_panel.csv"
        imc_panel_path = tmpdir / "imc_panel.csv"
        if_panel.to_csv(if_panel_path, header=False, index=False)
        imc_panel.to_csv(imc_panel_path, header=False, index=False)
        
        # Run workflow with mock optimizer
        study = run_registration_workflow(
            if_path=if_path,
            imc_path=imc_path,
            if_panel_path=if_panel_path,
            imc_panel_path=imc_panel_path,
            registration_channels=['DAPI'],
            correlation_channels=['CD3', 'CD20'],
            optimizer=MockOptimizer(),
            n_trials=1  # Just run one trial for testing
        )
        
        # Check study completed successfully
        assert len(study.trials) > 0
        assert study.trials[-1].state == optuna.trial.TrialState.COMPLETE


def test_workflow_with_custom_objective():
    """Test workflow with a custom objective function."""
    def custom_objective(trial, images, if_scale, registration_channels, correlation_channels):
        return 1.0  # Always return 1.0 for testing
    
    # Create mock data
    mock_if = np.random.rand(3, 100, 100)
    mock_imc = np.random.rand(3, 100, 100)
    
    # Create temporary files
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        
        # Save mock images
        if_path = tmpdir / "if_image.tiff"
        imc_path = tmpdir / "imc_image.tiff"
        tifffile.imwrite(if_path, mock_if)
        tifffile.imwrite(imc_path, mock_imc)
        
        # Create mock panel files
        if_panel = pd.DataFrame(['DAPI', 'CD3', 'CD20'])
        imc_panel = pd.DataFrame(['DAPI', 'CD3', 'CD20'])
        if_panel_path = tmpdir / "if_panel.csv"
        imc_panel_path = tmpdir / "imc_panel.csv"
        if_panel.to_csv(if_panel_path, header=False, index=False)
        imc_panel.to_csv(imc_panel_path, header=False, index=False)
        
        # Run workflow with custom objective
        study = run_registration_workflow(
            if_path=if_path,
            imc_path=imc_path,
            if_panel_path=if_panel_path,
            imc_panel_path=imc_panel_path,
            registration_channels=['DAPI'],
            correlation_channels=['CD3', 'CD20'],
            optimizer=MockOptimizer(),
            objective_fn=custom_objective,
            n_trials=1
        )
        
        # Check study completed with expected value
        assert len(study.trials) > 0
        assert study.trials[-1].value == 1.0


def test_workflow_with_callbacks():
    """Test workflow with callbacks."""
    callback_called = False
    
    def mock_callback(study, trial):
        nonlocal callback_called
        callback_called = True
    
    # Create mock data
    mock_if = np.random.rand(3, 100, 100)
    mock_imc = np.random.rand(3, 100, 100)
    
    # Create temporary files
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        
        # Save mock images
        if_path = tmpdir / "if_image.tiff"
        imc_path = tmpdir / "imc_image.tiff"
        tifffile.imwrite(if_path, mock_if)
        tifffile.imwrite(imc_path, mock_imc)
        
        # Create mock panel files
        if_panel = pd.DataFrame(['DAPI', 'CD3', 'CD20'])
        imc_panel = pd.DataFrame(['DAPI', 'CD3', 'CD20'])
        if_panel_path = tmpdir / "if_panel.csv"
        imc_panel_path = tmpdir / "imc_panel.csv"
        if_panel.to_csv(if_panel_path, header=False, index=False)
        imc_panel.to_csv(imc_panel_path, header=False, index=False)
        
        # Run workflow with callback
        study = run_registration_workflow(
            if_path=if_path,
            imc_path=imc_path,
            if_panel_path=if_panel_path,
            imc_panel_path=imc_panel_path,
            registration_channels=['DAPI'],
            correlation_channels=['CD3', 'CD20'],
            optimizer=MockOptimizer(),
            study_callbacks=[mock_callback],
            n_trials=1
        )
        
        # Check callback was called
        assert callback_called


def test_invalid_inputs():
    """Test workflow behavior with invalid inputs."""
    with pytest.raises(FileNotFoundError):
        run_registration_workflow(
            if_path="nonexistent.tiff",
            imc_path="nonexistent.tiff",
            if_panel_path="nonexistent.csv",
            imc_panel_path="nonexistent.csv",
            registration_channels=['DAPI'],
            correlation_channels=['CD3', 'CD20']
        ) 