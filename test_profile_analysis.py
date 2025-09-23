#!/usr/bin/env python3
"""
Test script to verify that the profile analysis in compute_model_metrics.py works correctly.
"""

import numpy as np
import sys
import os

# Add the src directory to the path so we can import our modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from compute_model_metrics import compute_single_model_metrics

def test_profile_analysis():
    """Test the profile analysis functionality."""
    print("Testing profile analysis...")
    
    # Create synthetic test data
    np.random.seed(42)
    t, z, lat, lon = 10, 50, 200, 200  # 10 time steps, 50 depth levels, 200x200 spatial grid (large enough for MS-SSIM)
    
    # Create synthetic truth data
    truth_data = np.random.rand(t, z, lat, lon) * 100  # Random values between 0-100
    
    # Create synthetic prediction data with some profiles being better/worse
    pred_data = truth_data.copy()
    
    # Make some profiles have higher errors (worse RMSE)
    pred_data[0] += np.random.rand(z, lat, lon) * 50  # Add significant noise to first profile
    pred_data[1] += np.random.rand(z, lat, lon) * 40  # Add noise to second profile
    
    # Make some profiles have lower errors (better RMSE)
    pred_data[8] += np.random.rand(z, lat, lon) * 5   # Add small noise to 9th profile
    pred_data[9] += np.random.rand(z, lat, lon) * 3   # Add very small noise to 10th profile
    
    # Create depth array
    depth_array = np.linspace(0, 100, z)
    
    # Test with fixed random profile coordinates
    test_random_coords = 5
    
    # Compute metrics
    metrics = compute_single_model_metrics(
        pred_data, truth_data, depth_array, cr=10.0, 
        random_profile_coords=test_random_coords
    )
    
    # Check that we have the expected structure
    assert 'profiles' in metrics, "Profiles key missing from metrics"
    profiles = metrics['profiles']
    
    expected_profile_types = ['best_rmse', 'worst_rmse', 'best_f1', 'worst_f1', 'best_r2', 'worst_r2', 'random']
    
    for profile_type in expected_profile_types:
        assert profile_type in profiles, f"Profile type {profile_type} missing"
        
        profile = profiles[profile_type]
        assert 'idx' in profile, f"Index missing for {profile_type}"
        assert 'truth' in profile, f"Truth data missing for {profile_type}"
        assert 'prediction' in profile, f"Prediction data missing for {profile_type}"
        assert 'rmse' in profile, f"RMSE missing for {profile_type}"
        assert 'f1' in profile, f"F1 missing for {profile_type}"
        assert 'r2' in profile, f"R2 missing for {profile_type}"
        
        # Check that the profile data has the correct shape
        assert profile['truth'].shape == (z, lat, lon), f"Truth shape incorrect for {profile_type}"
        assert profile['prediction'].shape == (z, lat, lon), f"Prediction shape incorrect for {profile_type}"
        
        # Check that the index is within bounds
        assert 0 <= profile['idx'] < t, f"Index out of bounds for {profile_type}"
    
    # Check that the random profile uses the correct index
    assert profiles['random']['idx'] == test_random_coords, "Random profile index incorrect"
    
    # Check that best/worst profiles make sense
    best_rmse_idx = profiles['best_rmse']['idx']
    worst_rmse_idx = profiles['worst_rmse']['idx']
    best_rmse_val = profiles['best_rmse']['rmse']
    worst_rmse_val = profiles['worst_rmse']['rmse']
    
    assert best_rmse_val < worst_rmse_val, "Best RMSE should be lower than worst RMSE"
    
    print("✓ All profile analysis tests passed!")
    print(f"✓ Found {len(profiles)} profiles as expected")
    print(f"✓ Random profile index: {profiles['random']['idx']}")
    print(f"✓ Best RMSE profile: {best_rmse_idx} (RMSE: {best_rmse_val:.3f})")
    print(f"✓ Worst RMSE profile: {worst_rmse_idx} (RMSE: {worst_rmse_val:.3f})")
    
    return True

if __name__ == "__main__":
    test_profile_analysis()
