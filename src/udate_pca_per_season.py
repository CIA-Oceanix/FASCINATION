import os
import sys
import pickle
import numpy as np
import xarray as xr
from tqdm import tqdm

# Set up paths
running_path = "/Odyssey/private/o23gauvr/code/"
sys.path.insert(0, running_path)
os.chdir(running_path)

# Import your existing functions
from FASCINATION.src.pca_compo_dict import *

def load_existing_metrics(pickle_path):
    """Load existing metrics dictionary if it exists"""
    if os.path.exists(pickle_path):
        with open(pickle_path, 'rb') as f:
            return pickle.load(f)
    else:
        return {}

def save_metrics(metrics_dict, pickle_path):
    save_path = pickle_path.replace("pca","pca_test")  
 
    """Save metrics dictionary to pickle file"""
    with open(save_path, 'wb') as f:
        pickle.dump(metrics_dict, f)
    print(f"Saved metrics to {save_path}")

def compute_metrics_for_threshold(data_recon, data_truth, threshold):
    """Compute all metrics for a given threshold and reconstruction"""
    
    # Calculate compression ratio
    original_size = data_truth.size
    compressed_size = np.count_nonzero(data_recon)
    compression_ratio = original_size / compressed_size if compressed_size > 0 else float('inf')
    
    # Calculate metrics
    rmse = np.sqrt(np.mean((data_recon - data_truth) ** 2))
    psnr = compute_psnr(data_truth, data_recon)
    ms_ssim_val = compute_msssim(data_truth, data_recon)
    
    # ECS calculation
    ecs = np.sum(np.abs(data_recon - data_truth))
    
    # MAE
    mae = np.mean(np.abs(data_recon - data_truth))
    
    # Mean error normalized by min-max range
    data_range = np.max(data_truth) - np.min(data_truth)
    mean_error_n_min_max = np.mean(np.abs(data_recon - data_truth)) / data_range if data_range > 0 else 0
    
    # F1 scores
    min_max_idx_truth = get_min_max_idx(data_truth, axs=1)
    min_max_idx_recon = get_min_max_idx(data_recon, axs=1)
    f1_score = calculate_confusion_matrix_and_f1_score(min_max_idx_truth, min_max_idx_recon, axs=1)
    
    # Filtered F1 score (assuming this exists in your original code)
    # You might need to adjust this based on your actual implementation
    filtered_f1_score = f1_score  # Placeholder - adjust as needed
    
    # R2 score
    ss_res = np.sum((data_truth - data_recon) ** 2)
    ss_tot = np.sum((data_truth - np.mean(data_truth)) ** 2)
    r2_score = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
    
    return {
        'CR': compression_ratio,
        'RMSE': rmse,
        'PSNR': psnr,
        'MS-SSIM': ms_ssim_val,
        'ECS': ecs,
        'MAE': mae,
        'mean_error_n_min_max': mean_error_n_min_max,
        'F1_score': f1_score,
        'Filtered_F1_score': filtered_f1_score,
        'R2_score': r2_score
    }

def update_pca_metrics():
    """Main function to update PCA metrics with new thresholds"""
    
    # Define paths
    pickle_path = "/Odyssey/private/o23gauvr/code/FASCINATION/pickle/season_metrics_all_pca.pkl"
    data_path = "/Odyssey/public/enatl60/celerity/eNATL60_BLB002_sound_speed_regrid_0_botm.nc"
    
    # Define all thresholds you want to have
    all_thresholds = [0.999, 0.9999, 0.99999]
    
    # Load existing metrics
    print("Loading existing metrics...")
    metrics_dict = load_existing_metrics(pickle_path)
    
    # Load data
    print("Loading data...")
    da = xr.open_dataarray(data_path)
    
    # Define seasons and combinations
    seasons = {
        'all': slice(None),
        'spring': slice('2009-03', '2009-06'),
        'summer': slice('2009-06', '2009-09'), 
        'autumn': slice('2009-09', '2009-12'),
        'winter': slice('2009-12', '2010-03')
    }
    
    combinations = ['depth', 'spatial', 'time', 'depth_spatial', 'depth_time', 'spatial_time', 'depth_spatial_time']
    
    total_computations = 0
    skipped_computations = 0
    
    # Process each season and combination
    for season_name, season_slice in seasons.items():
        print(f"\nProcessing season: {season_name}")
        
        # Initialize season in metrics_dict if not exists
        if season_name not in metrics_dict:
            metrics_dict[season_name] = {}
        
        # Get seasonal data
        if season_name == 'all':
            seasonal_da = da
        else:
            seasonal_da = da.sel(time=season_slice)
        
        for combination in combinations:
            print(f"  Processing combination: {combination}")
            
            # Initialize combination in metrics_dict if not exists
            if combination not in metrics_dict[season_name]:
                metrics_dict[season_name][combination] = {}
            
            # Check which thresholds are missing
            existing_thresholds = set(metrics_dict[season_name][combination].keys())
            missing_thresholds = [t for t in all_thresholds if t not in existing_thresholds]
            
            if not missing_thresholds:
                print(f"    All thresholds already exist for {season_name}/{combination}")
                skipped_computations += len(all_thresholds)
                continue
            
            print(f"    Missing thresholds: {missing_thresholds}")
            
            # Prepare data for PCA
            print("    Preparing data for PCA...")
            
            # Convert to numpy and handle NaN values
            data_np = seasonal_da.values
            data_np = np.nan_to_num(data_np, nan=0.0)
            
            # Apply PCA based on combination
            if combination == 'depth':
                # PCA along depth dimension
                original_shape = data_np.shape
                data_reshaped = data_np.reshape(-1, data_np.shape[0])
                pca = PCA()
                pca.fit(data_reshaped.T)
                
            elif combination == 'spatial':
                # PCA along spatial dimensions
                original_shape = data_np.shape
                data_reshaped = data_np.reshape(data_np.shape[0], -1)
                pca = PCA()
                pca.fit(data_reshaped)
                
            elif combination == 'time':
                # PCA along time dimension
                original_shape = data_np.shape
                data_reshaped = data_np.reshape(data_np.shape[0], -1).T
                pca = PCA()
                pca.fit(data_reshaped)
                
            elif combination == 'depth_spatial':
                # PCA along depth and spatial dimensions
                original_shape = data_np.shape
                data_reshaped = data_np.reshape(-1, data_np.shape[-1])
                pca = PCA()
                pca.fit(data_reshaped)
                
            elif combination == 'depth_time':
                # PCA along depth and time dimensions
                original_shape = data_np.shape
                data_reshaped = data_np.reshape(-1, data_np.shape[1] * data_np.shape[2])
                pca = PCA()
                pca.fit(data_reshaped)
                
            elif combination == 'spatial_time':
                # PCA along spatial and time dimensions
                original_shape = data_np.shape
                data_reshaped = data_np.reshape(data_np.shape[0], -1)
                pca = PCA()
                pca.fit(data_reshaped)
                
            elif combination == 'depth_spatial_time':
                # PCA along all dimensions
                original_shape = data_np.shape
                data_reshaped = data_np.reshape(-1, 1)
                pca = PCA()
                pca.fit(data_reshaped)
            
            # Compute metrics for missing thresholds
            for threshold in missing_thresholds:
                print(f"      Computing metrics for threshold {threshold}")
                
                # Determine number of components to keep
                cumsum_var = np.cumsum(pca.explained_variance_ratio_)
                n_components = np.argmax(cumsum_var >= threshold) + 1
                
                # Transform and reconstruct
                data_transformed = pca.transform(data_reshaped)[:, :n_components]
                data_reconstructed = pca.inverse_transform(
                    np.column_stack([data_transformed, np.zeros((data_transformed.shape[0], 
                                                               pca.n_components_ - n_components))])
                )
                
                # Reshape back to original shape
                data_reconstructed = data_reconstructed.reshape(original_shape)
                
                # Compute metrics
                metrics = compute_metrics_for_threshold(data_reconstructed, data_np, threshold)
                
                # Store metrics
                metrics_dict[season_name][combination][threshold] = metrics
                total_computations += 1
                
                print(f"        CR: {metrics['CR']:.2f}, RMSE: {metrics['RMSE']:.4f}, PSNR: {metrics['PSNR']:.2f}")
            
            # Save progress after each combination
            save_metrics(metrics_dict, pickle_path)
            print(f"    Saved progress for {season_name}/{combination}")
    
    print(f"\nSummary:")
    print(f"Total new computations: {total_computations}")
    print(f"Skipped existing computations: {skipped_computations}")
    print(f"Final metrics saved to: {pickle_path}")

if __name__ == "__main__":
    update_pca_metrics()