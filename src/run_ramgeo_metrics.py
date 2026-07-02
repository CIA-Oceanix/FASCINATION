import os
import struct
import subprocess
import shutil
from typing import List, Tuple
import numpy as np
import pandas as pd
import xarray as xr
from tqdm import tqdm
from scipy.ndimage import gaussian_filter1d


def ssim_1d(x, y, sigma=1.5, C1=1e-4, C2=1e-4):
    """Compute 1D SSIM for profiles."""
    x = x.astype(np.float64)
    y = y.astype(np.float64)
    
    mu_x = gaussian_filter1d(x, sigma)
    mu_y = gaussian_filter1d(y, sigma)
    
    sigma_x = gaussian_filter1d(x**2, sigma) - mu_x**2
    sigma_y = gaussian_filter1d(y**2, sigma) - mu_y**2
    sigma_xy = gaussian_filter1d(x*y, sigma) - mu_x*mu_y
    
    numerator = (2 * mu_x * mu_y + C1) * (2 * sigma_xy + C2)
    denominator = (mu_x**2 + mu_y**2 + C1) * (sigma_x + sigma_y + C2)
    
    return np.mean(numerator / (denominator + 1e-12))


def ms_ssim_1d(x, y, scales=5):
    """Compute Multi-scale SSIM for profiles."""
    weights = np.array([0.0448, 0.2856, 0.3001, 0.2363, 0.1333])
    weights = weights[:scales]
    
    mssim = []
    x_s, y_s = x, y
    
    for i in range(scales):
        mssim.append(ssim_1d(x_s, y_s))
        if i < scales - 1:
            x_s = x_s[::2]
            y_s = y_s[::2]
    
    mssim = np.array(mssim)
    return np.prod(mssim ** weights)


# ============================================================================
# RAMGEO METRICS FUNCTIONS
# ============================================================================

def _read_fortran_record(file_obj):
    """Read a single Fortran unformatted sequential record."""
    len_bytes = file_obj.read(4)
    if len(len_bytes) < 4:
        return None
    reclen = struct.unpack('i', len_bytes)[0]
    data = file_obj.read(reclen)
    if len(data) < reclen:
        return None
    trailer = file_obj.read(4)
    if len(trailer) < 4:
        return None
    return data


def read_binary_grid(filename):
    """Read tl.grid (real) files written by Fortran unformatted I/O.
    
    Args:
        filename: Path to tl.grid binary file
        
    Returns:
        tuple: (grid_array: 2D numpy array (depth x range), lz: number of depth points)
    """
    try:
        with open(filename, 'rb') as f:
            header = _read_fortran_record(f)
            if header is None:
                return None, None
            
            lz = struct.unpack('i', header)[0]
            
            # Read TL values at each range step
            grid_data = []
            while True:
                record = _read_fortran_record(f)
                if record is None:
                    break
                
                # Parse the TL values (lz float32 values)
                values = np.frombuffer(record, dtype=np.float32, count=lz)
                grid_data.append(values)
            
            if grid_data:
                grid_array = np.array(grid_data).T  # Transpose to get (depth, range)
                return grid_array, lz
            else:
                return None, None
                
    except Exception as e:
        print(f"Error reading {filename}: {e}")
        return None, None


def lon_distance_meters(lat, lon1, lon2):
    """Compute distance in meters between two longitudes at a given latitude."""
    R = 6_371_000  # Earth radius in meters
    lat_rad = np.radians(lat)
    dlon_rad = np.radians(lon2 - lon1)
    distance = R * np.cos(lat_rad) * np.abs(dlon_rad)
    return distance


def create_ramgeo_config(title, ssp_profile, depth_array, 
                         r_update=5000, 
                         freq=500, source_depth=200.0, receiver_depth=300.0,
                         rmax=10000.0, dr=.749, ndr=1,
                         zmax=2000.0, dz=.149, ndz=1, zmplt=1000.0,
                         c0=1500.0, n_pade=8, ns=1, rs=0.0):
    """Create RAMGEO config from SSP profile."""
    r = 0.0
    ssp_profile = ssp_profile.squeeze()  # Remove singleton dimensions if any

    config = f"""{title}
{freq} {source_depth} {receiver_depth}\tFrequency (Hz), source depth (m), receiver depth(s) (m)
{rmax}\t{dr} {ndr}\tMax range (m), range step dr (m) range decimation
{zmax}\t{dz} {ndz} {zmplt}\tMax depth (m), depth step dz (m), depth decimation, depth for plotting (m)
{c0}\t{n_pade} {ns} {rs}\tReference sound speed (m/s) Number of Pade terms, number of stability constraints in Pade approximation, Range of starter
0.0\t{2*zmax}\tbathymetry data r,z in [m, m]
{rmax}\t{2*zmax}
-1\t-1
"""
    # Add SSP profile
    assert ssp_profile.shape[0] == len(depth_array), "SSP profile and depth array must have the same length."

    if len(ssp_profile.shape) == 1:
        ssp_profile = np.expand_dims(ssp_profile, axis=-1)

    for j in range(ssp_profile.shape[1]):
        if j > 0:
            config += f"{r}\t!update profile\n"

        for i, (z, c) in enumerate(zip(depth_array, ssp_profile[:, j])):
            if i == 0:
                config += f"{z:.2f}\t{c:.2f}\tSSP profile (depth, sound speed)\n"
            else:
                config += f"{z:.2f}\t{c:.2f}\n"

        config += """-1\t-1 
0.0\t1800.0\tcompressive sound speed profile in substrate {z,cbp} [m, m/s]
100.0   1800.0
-1\t-1
0.0\t1.7\tdensity profile in substrate {z,rho} [m, g/cm³]
100.0   1.7
-1\t-1
0.0\t0.4\tcompressive attenuation profile  {z,attnp} [m, dB/lambda]
10.0    0.4
100.0\t10.0    buffer layer with total attenuation
-1\t-1
"""
        r += r_update
        if r > rmax:
            break

    output_path = f"/Odyssey/private/o23gauvr/code/RAMGEO2025/RAMGEO2025/config/{title.replace(' ', '_').replace('/', '_')}.in"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(config)
    
    return output_path


def select_random_slices(ssp_truth_shape: tuple, n_slices: int = 5, random_seed: int = 42) -> List[Tuple[int, int]]:
    """Select n random (time, lat) indices with a given seed.
    
    Args:
        ssp_truth_shape: Shape of SSP truth data (time, depth, lat, lon)
        n_slices: Number of slices to select
        random_seed: Random seed for reproducibility
        
    Returns:
        List of (time_idx, lat_idx) tuples
    """
    np.random.seed(random_seed)
    n_time = ssp_truth_shape[0]
    n_lat = ssp_truth_shape[2]
    
    selected_slices = []
    for _ in range(n_slices):
        t_idx = np.random.randint(0, n_time)
        lat_idx = np.random.randint(0, n_lat)
        selected_slices.append((t_idx, lat_idx))
    
    return selected_slices


def compute_tl_mae_ms_ssim(tl_truth, tl_model):
    """Compute MAE and MS-SSIM between truth and model TL grids.
    
    Args:
        tl_truth: Truth TL grid (depth, range)
        tl_model: Model TL grid (depth, range)
        
    Returns:
        dict: {'mae': MAE value, 'ms_ssim': MS-SSIM value}
    """
    # Flatten and compute MAE
    valid_mask = np.isfinite(tl_truth) & np.isfinite(tl_model)
    if not np.any(valid_mask):
        return {'mae': np.nan, 'ms_ssim': np.nan}
    
    mae = np.abs(tl_truth[valid_mask] - tl_model[valid_mask]).mean()
    
    # Compute MS-SSIM (row-wise)
    ms_ssim_vals = []
    for i in range(tl_truth.shape[0]):
        truth_row = tl_truth[i, :]
        model_row = tl_model[i, :]
        if np.isfinite(truth_row).all() and np.isfinite(model_row).all():
            ms_ssim = ms_ssim_1d(truth_row, model_row)
            ms_ssim_vals.append(ms_ssim)
    
    ms_ssim_mean = np.mean(ms_ssim_vals) if ms_ssim_vals else np.nan
    
    return {'mae': mae, 'ms_ssim': ms_ssim_mean}


def run_ramgeo_and_compute_metrics(
    ssp_truth_da: xr.DataArray,
    ssp_ae_mlic_da: xr.DataArray,
    ssp_ae_cae_da: xr.DataArray,
    selected_slices: List[Tuple[int, int]],
    freq_list: List[int] = None,
    output_dir: str = None,
    random_seed: int = 42,
    device: str = 'cuda',
) -> pd.DataFrame:
    """Run RAMGEO simulations and compute transmission loss metrics.
    
    Args:
        ssp_truth_da: Truth SSP DataArray with z coordinate
        ssp_ae_mlic_da: MLIC reconstructed SSP DataArray
        ssp_ae_cae_da: CAE reconstructed SSP DataArray
        selected_slices: List of (time, lat) indices to simulate
        freq_list: List of frequencies to simulate (default: [10, 100, 500, 1000])
        output_dir: Output directory for results
        random_seed: Random seed for reproducibility
        device: Device for torch models
        
    Returns:
        pd.DataFrame: Metrics table with columns [Frequency, Model, MAE, MS-SSIM]
    """
    if freq_list is None:
        freq_list = [10, 100, 500, 1000]
    
    if output_dir is None:
        output_dir = "/Odyssey/private/o23gauvr/code/RAMGEO2025/RAMGEO2025/data/test_metrics_ramgeo"
    
    os.makedirs(output_dir, exist_ok=True)
    
    depth_array = ssp_truth_da.z.values
    ssp_truth = ssp_truth_da.values
    ssp_ae_mlic = ssp_ae_mlic_da.values
    ssp_ae_cae = ssp_ae_cae_da.values
    
    # Get spatial reference for lon_distance calculation
    lat_val = float(ssp_truth_da.lat.values[selected_slices[0][1]])
    lon_vals = ssp_truth_da.lon.values
    
    results_data = []
    
    print(f"\n{'='*80}")
    print(f"Running RAMGEO simulations for {len(selected_slices)} selected slices")
    print(f"{'='*80}")
    
    for slice_idx, (t_idx, lat_idx) in enumerate(tqdm(selected_slices, desc="Processing slices")):
        print(f"\n► Processing slice {slice_idx+1}/{len(selected_slices)}: time={t_idx}, lat={lat_idx}")
        
        for freq in tqdm(freq_list, desc=f"  Frequencies", leave=False):
            # Compute RAMGEO parameters based on frequency
            c0 = 1500
            lmda = c0 / freq
            dr = 0.5 * lmda * 0.99
            dz = 0.1 * lmda * 0.99
            
            # Calculate radial distance
            r_update = abs(float(lon_distance_meters(lat_val, lon_vals[0], lon_vals[1])))
            
            # Create slice identifier
            slice_id = f"t{t_idx}_lat{lat_idx}_freq{freq}"
            slice_dir = os.path.join(output_dir, f"slice_{slice_idx:03d}_{slice_id}")
            os.makedirs(slice_dir, exist_ok=True)
            
            try:
                # Extract 1D SSP profiles (depth dimension only, averaging over lon)
                ssp_truth_profile = ssp_truth[t_idx, :, lat_idx, :].mean(axis=-1)
                ssp_mlic_profile = ssp_ae_mlic[t_idx, :, lat_idx, :].mean(axis=-1)
                ssp_cae_profile = ssp_ae_cae[t_idx, :, lat_idx, :].mean(axis=-1)
                
                # Run RAMGEO for Truth
                truth_subdir = os.path.join(slice_dir, "TRUTH")
                os.makedirs(truth_subdir, exist_ok=True)
                
                config_path_truth = create_ramgeo_config(
                    f"TRUTH_{slice_id}", ssp_truth_profile, depth_array,
                    r_update=r_update, freq=freq,
                    rmax=10000.0, dr=dr, ndr=1,
                    zmax=2000.0, dz=dz, ndz=1, zmplt=1000.0,
                    c0=c0, n_pade=8, ns=1, rs=0.0
                )
                
                # Copy results template
                ramgeo_results_template = "/Odyssey/private/o23gauvr/code/RAMGEO2025/RAMGEO2025/data/results"
                if os.path.exists(ramgeo_results_template):
                    if os.path.exists(truth_subdir):
                        shutil.rmtree(truth_subdir)
                    shutil.copytree(ramgeo_results_template, truth_subdir)
                
                # Run RAMGEO
                try:
                    subprocess.run([
                        '/Odyssey/private/o23gauvr/code/RAMGEO2025/RAMGEO2025/ramgeo',
                        f'--input={config_path_truth}',
                        f'--output-dir={truth_subdir}'
                    ], capture_output=True, timeout=60)
                except Exception as e:
                    print(f"    ✗ RAMGEO failed for Truth: {e}")
                    continue
                
                # Read truth TL grid
                truth_grid_file = os.path.join(truth_subdir, "tl.grid")
                tl_truth, lz = read_binary_grid(truth_grid_file)
                
                if tl_truth is None:
                    print(f"    ✗ Failed to read truth TL grid")
                    continue
                
                # Run RAMGEO for MLIC
                mlic_subdir = os.path.join(slice_dir, "MLIC")
                os.makedirs(mlic_subdir, exist_ok=True)
                
                config_path_mlic = create_ramgeo_config(
                    f"MLIC_{slice_id}", ssp_mlic_profile, depth_array,
                    r_update=r_update, freq=freq,
                    rmax=10000.0, dr=dr, ndr=1,
                    zmax=2000.0, dz=dz, ndz=1, zmplt=1000.0,
                    c0=c0, n_pade=8, ns=1, rs=0.0
                )
                
                if os.path.exists(ramgeo_results_template):
                    if os.path.exists(mlic_subdir):
                        shutil.rmtree(mlic_subdir)
                    shutil.copytree(ramgeo_results_template, mlic_subdir)
                
                try:
                    subprocess.run([
                        '/Odyssey/private/o23gauvr/code/RAMGEO2025/RAMGEO2025/ramgeo',
                        f'--input={config_path_mlic}',
                        f'--output-dir={mlic_subdir}'
                    ], capture_output=True, timeout=60)
                    
                    tl_mlic, _ = read_binary_grid(os.path.join(mlic_subdir, "tl.grid"))
                    if tl_mlic is not None:
                        metrics_mlic = compute_tl_mae_ms_ssim(tl_truth, tl_mlic)
                        results_data.append({
                            'Frequency (Hz)': int(freq),
                            'Model': 'MLIC',
                            'MAE (dB)': metrics_mlic['mae'],
                            'MS-SSIM': metrics_mlic['ms_ssim'],
                            'Time_idx': t_idx,
                            'Lat_idx': lat_idx,
                        })
                except Exception as e:
                    print(f"    ✗ RAMGEO failed for MLIC: {e}")
                
                # Run RAMGEO for CAE
                cae_subdir = os.path.join(slice_dir, "CAE")
                os.makedirs(cae_subdir, exist_ok=True)
                
                config_path_cae = create_ramgeo_config(
                    f"CAE_{slice_id}", ssp_cae_profile, depth_array,
                    r_update=r_update, freq=freq,
                    rmax=10000.0, dr=dr, ndr=1,
                    zmax=2000.0, dz=dz, ndz=1, zmplt=1000.0,
                    c0=c0, n_pade=8, ns=1, rs=0.0
                )
                
                if os.path.exists(ramgeo_results_template):
                    if os.path.exists(cae_subdir):
                        shutil.rmtree(cae_subdir)
                    shutil.copytree(ramgeo_results_template, cae_subdir)
                
                try:
                    subprocess.run([
                        '/Odyssey/private/o23gauvr/code/RAMGEO2025/RAMGEO2025/ramgeo',
                        f'--input={config_path_cae}',
                        f'--output-dir={cae_subdir}'
                    ], capture_output=True, timeout=60)
                    
                    tl_cae, _ = read_binary_grid(os.path.join(cae_subdir, "tl.grid"))
                    if tl_cae is not None:
                        metrics_cae = compute_tl_mae_ms_ssim(tl_truth, tl_cae)
                        results_data.append({
                            'Frequency (Hz)': int(freq),
                            'Model': 'CAE',
                            'MAE (dB)': metrics_cae['mae'],
                            'MS-SSIM': metrics_cae['ms_ssim'],
                            'Time_idx': t_idx,
                            'Lat_idx': lat_idx,
                        })
                except Exception as e:
                    print(f"    ✗ RAMGEO failed for CAE: {e}")
                    
            except Exception as e:
                print(f"    ✗ Error processing slice: {e}")
                continue
    
    # Create results DataFrame
    df_results = pd.DataFrame(results_data)
    
    if len(df_results) > 0:
        # Export to CSV
        csv_file = os.path.join(output_dir, "ramgeo_metrics.csv")
        df_results.to_csv(csv_file, index=False)
        print(f"\n✓ Saved metrics to: {csv_file}")
        
        # Compute aggregated statistics by frequency and model
        agg_stats = df_results.groupby(['Frequency (Hz)', 'Model']).agg({
            'MAE (dB)': ['mean', 'std', 'min', 'max'],
            'MS-SSIM': ['mean', 'std', 'min', 'max']
        }).round(4)
        
        print(f"\n{'='*80}")
        print("RAMGEO Metrics Summary (aggregated by Frequency and Model)")
        print(f"{'='*80}")
        print(agg_stats)
        
        return df_results
    else:
        print("\n✗ No results generated")
        return pd.DataFrame()