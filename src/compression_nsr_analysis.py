"""
Simplified Spectral NSR (Noise-to-Signal Ratio) Analysis for Compression Models

Computes spectral resolution for compressed data by computing NSR as:
    NSR(λ) = PSD_error(λ) / PSD_original(λ)

Two main functions:
    - compute_nsr_along_depth: spectral resolution along depth axis
    - compute_nsr_spatial_map: spectral resolution over lat/lon spatial dimensions

Reference:
    Ballarotta et al. (2019) - On the resolutions of ocean altimetry maps
    Ocean Sci., 15, 1091–1109
"""

import numpy as np
import xarray as xr
from scipy.signal import welch, coherence
from typing import Tuple, Dict, Optional
import logging
import warnings

try:
    from geopy.distance import geodesic
except ImportError:
    geodesic = None

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format='%(message)s')




def find_resolution_crossing(
    wavenumber: np.ndarray,
    nsr: np.ndarray,
    target_ratio: float = 0.5,
) -> Tuple[float, bool]:
    """
    Find the effective resolution (wavelength) where NSR crosses target_ratio.

    Parameters
    ----------
    wavenumber : np.ndarray
        Array of wavenumbers (1/wavelength).
    nsr : np.ndarray
        Noise-to-Signal Ratio values.
    target_ratio : float, optional
        Target NSR threshold, defaults to 0.5.

    Returns
    -------
    float
        Effective resolution (wavelength in same units as spacing).
    bool
        Flag indicating multiple crossings detected.
    """
    flag_multiple_crossing = False
    zero_crossings = np.where(np.diff(np.sign(nsr - target_ratio)))[0]
    
    if len(zero_crossings) > 1:
        flag_multiple_crossing = True
        zero_crossings = zero_crossings[-1:]  # Use last (largest wavelength)
    
    if len(zero_crossings) > 0:
        idx = zero_crossings[0]
        if idx + 1 < len(wavenumber):
            # Linear interpolation in log-space
            nsr1 = nsr[idx] - target_ratio
            nsr2 = nsr[idx + 1] - target_ratio
            k1 = np.log(wavenumber[idx] + 1e-10)
            k2 = np.log(wavenumber[idx + 1] + 1e-10)
            
            if abs(nsr1 - nsr2) > 1e-10:
                log_wavenumber_crossing = k1 - nsr1 * (k1 - k2) / (nsr1 - nsr2)
                resolution = 1.0 / (np.exp(log_wavenumber_crossing) + 1e-10)
            else:
                resolution = 1.0 / (wavenumber[idx] + 1e-10)
        else:
            resolution = 0.0
    else:
        # No crossing: use percentile-based fallback
        #flag_multiple_crossing = True
        resolution = 0.0
        # percentile = np.nanpercentile(nsr, target_ratio * 100)
        # zero_crossings_pct = np.where(np.diff(np.sign(nsr - percentile)))[0]
        
        # if len(zero_crossings_pct) > 0:
        #     idx = zero_crossings_pct[0]
        #     resolution = 1.0 / (wavenumber[idx] + 1e-10) if idx < len(wavenumber) else 0.0
        # else:
        #     # Last resort: use median wavenumber
        #     median_wn = np.nanmedian(wavenumber[wavenumber > 0])
        #     resolution = 1.0 / (median_wn + 1e-10) if median_wn > 0 else 0.0
    
    return resolution, flag_multiple_crossing


def compute_nsr_along_depth(
    truth_da: xr.DataArray,
    reconstructed_da: xr.DataArray,
    target_ratio: float = 0.5,
) -> Dict[str, np.ndarray]:
    """
    Compute spectral NSR-based resolution along the depth (z) axis.

    Handles non-uniform depth sampling by interpolating to uniform grid.

    Parameters
    ----------
    truth_da : xr.DataArray
        True data array with shape (time, z, lat, lon). Z axis in meters.
    reconstructed_da : xr.DataArray
        Reconstructed data array with same shape. Already NaN-filtered.
    target_ratio : float, optional
        Target NSR threshold for resolution (default 0.5).

    Returns
    -------
    dict
        Dictionary with keys:
        - 'wavenumber': wavenumber array (1/m)
        - 'wavelength': wavelength array (m)
        - 'nsr': NSR values as function of wavenumber
        - 'coherence': coherence between truth and reconstructed
        - 'resolution': effective resolution (m) where NSR = target_ratio
        - 'flag': bool indicating if multiple crossings or fallback was used
    """
    logging.info('Computing NSR along depth axis...')
    
    # Get z-axis and check uniformity
    z_vals = truth_da.z.values
    z_diffs = np.diff(z_vals)
    z_mean_diff = np.mean(z_diffs)
    z_uniform_check = np.std(z_diffs) / z_mean_diff < 0.05
    
    # Create uniform z-grid for interpolation
    z_uniform = np.linspace(float(z_vals.min()), float(z_vals.max()), len(z_vals))
    delta_z = np.mean(np.diff(z_uniform))  # uniform spacing
    fs = 1.0 / delta_z  # sampling frequency (1/meter)
    
    if not z_uniform_check:
        logging.info(f'Non-uniform z-axis detected. Interpolating to uniform grid (memory-efficient)...')
    
    # Get raw data as numpy arrays
    truth_raw = truth_da.values  # (time, z, lat, lon)
    recon_raw = reconstructed_da.values  # (time, z, lat, lon)
    
    # Reshape to (time*lat*lon, z) for efficient interpolation
    nt, nz, nlat, nlon = truth_raw.shape
    truth_reshaped = truth_raw.reshape(nt * nlat * nlon, nz)  # (spatial*time, z)
    recon_reshaped = recon_raw.reshape(nt * nlat * nlon, nz)
    
    # Interpolate each profile to uniform grid using np.interp (memory efficient)
    # Process in chunks to avoid peak memory spike
    chunk_size = max(1000, truth_reshaped.shape[0] // 10)  # ~10% chunks
    truth_interp_list = []
    recon_interp_list = []
    
    logging.info(f'Interpolating {truth_reshaped.shape[0]} profiles in chunks of {chunk_size}...')
    for i in range(0, truth_reshaped.shape[0], chunk_size):
        chunk_end = min(i + chunk_size, truth_reshaped.shape[0])
        chunk = slice(i, chunk_end)
        
        # Interpolate chunk: for each profile, interpolate to uniform grid
        truth_chunk_interp = np.zeros((chunk_end - i, len(z_uniform)))
        recon_chunk_interp = np.zeros((chunk_end - i, len(z_uniform)))
        
        for j, (t_profile, r_profile) in enumerate(zip(truth_reshaped[chunk], recon_reshaped[chunk])):
            # Only interpolate where both are valid
            valid_mask = np.isfinite(t_profile) & np.isfinite(r_profile)
            if valid_mask.sum() > 1:
                truth_chunk_interp[j] = np.interp(z_uniform, z_vals[valid_mask], t_profile[valid_mask], 
                                                   left=np.nan, right=np.nan)
                recon_chunk_interp[j] = np.interp(z_uniform, z_vals[valid_mask], r_profile[valid_mask],
                                                   left=np.nan, right=np.nan)
            else:
                truth_chunk_interp[j] = np.nan
                recon_chunk_interp[j] = np.nan
        
        truth_interp_list.append(truth_chunk_interp)
        recon_interp_list.append(recon_chunk_interp)
        del truth_chunk_interp, recon_chunk_interp  # Free chunk memory
    
    # Concatenate chunks
    truth_interp = np.vstack(truth_interp_list)  # (spatial*time, z_uniform)
    recon_interp = np.vstack(recon_interp_list)
    del truth_interp_list, recon_interp_list, truth_reshaped, recon_reshaped, truth_raw, recon_raw
    
    # Reshape back to (time, z_uniform, lat, lon) and flatten for spectral analysis
    truth_interp = truth_interp.reshape(nt, nlat, nlon, len(z_uniform)).transpose(0, 3, 1, 2)
    recon_interp = recon_interp.reshape(nt, nlat, nlon, len(z_uniform)).transpose(0, 3, 1, 2)
    
    # Flatten all dimensions except z: (time, z_uniform, lat, lon) -> (z_uniform, time*lat*lon)
    truth_2d = truth_interp.reshape(len(z_uniform), -1)
    recon_2d = recon_interp.reshape(len(z_uniform), -1)
    del truth_interp, recon_interp
    
    # Remove NaN values
    valid_mask = np.isfinite(truth_2d) & np.isfinite(recon_2d)
    truth_flat = truth_2d[valid_mask]
    recon_flat = recon_2d[valid_mask]
    del truth_2d, recon_2d
    
    # Compute PSDs using Welch's method (on uniformly-sampled data)
    nperseg = max(32, min(256, len(truth_flat) // 4))
    error = truth_flat - recon_flat
    
    wavenumber, psd_error = welch(error, fs=fs, nperseg=nperseg, scaling='density', noverlap=0)
    _, psd_truth = welch(truth_flat, fs=fs, nperseg=nperseg, scaling='density', noverlap=0)
    _, coh = coherence(truth_flat, recon_flat, fs=fs, nperseg=nperseg, noverlap=0)
    
    # Free temporary arrays
    del truth_flat, recon_flat, error
    
    # Compute NSR
    epsilon = np.finfo(float).eps
    nsr = np.divide(psd_error, psd_truth + epsilon, out=np.zeros_like(psd_error))

    print(f'NSR range: {nsr.min():.4f} to {nsr.max():.4f}')
    
    # Find resolution where NSR crosses target_ratio
    resolution, flag = find_resolution_crossing(wavenumber, nsr, target_ratio=target_ratio)
    wavelength = 1.0 / (wavenumber + 1e-10)
    
    logging.info(f'Effective resolution: {resolution:.2f} m (at NSR={target_ratio})')
    
    return {
        'wavenumber': wavenumber,
        'wavelength': wavelength,
        'nsr': nsr,
        'coherence': coh,
        'resolution': resolution,
        'flag': flag,
    }


def compute_nsr_spatial_map(
    truth_da: xr.DataArray,
    reconstructed_da: xr.DataArray,
    target_ratio: float = 0.5,
) -> Dict[str, np.ndarray]:
    """
    Compute spectral NSR-based resolution over spatial (lat, lon) dimensions.

    Computes delta_x and delta_y from lat/lon coordinates using geodesic distances.
    Data is averaged over the depth (z) and time dimensions.

    Parameters
    ----------
    truth_da : xr.DataArray
        True data array with shape (time, z, lat, lon). Z axis in meters.
    reconstructed_da : xr.DataArray
        Reconstructed data array with same shape. Already NaN-filtered.
    target_ratio : float, optional
        Target NSR threshold for resolution (default 0.5).

    Returns
    -------
    dict
        Dictionary with keys:
        - 'wavenumber': wavenumber array (1/m)
        - 'wavelength': wavelength array (m)
        - 'nsr': NSR values as function of wavenumber
        - 'coherence': coherence between truth and reconstructed
        - 'resolution': effective resolution (m) where NSR = target_ratio
        - 'delta_x': median grid spacing in x-direction (m)
        - 'delta_y': median grid spacing in y-direction (m)
        - 'flag': bool indicating if multiple crossings or fallback was used
    """
    logging.info('Computing NSR over spatial (lat/lon) dimensions...')
    
    # Average over time and depth to get spatial field
    truth_spatial = truth_da.mean(dim=['time', 'z']).values  # (lat, lon)
    recon_spatial = reconstructed_da.mean(dim=['time', 'z']).values  # (lat, lon)
    
    # Compute delta_x and delta_y from coordinates using geodesic distance
    lat_vals = truth_da.lat.values
    lon_vals = truth_da.lon.values
    
    # if geodesic is None:
    #     logging.warning('geopy not available. Using simple degree-based spacing.')
    #     # Simple fallback: degrees to km (approximate)
    #     delta_lat_deg = np.median(np.abs(np.diff(lat_vals)))
    #     delta_lon_deg = np.median(np.abs(np.diff(lon_vals)))
        
    #     # Rough conversion: 1 degree ~ 111 km at equator
    #     delta_y = delta_lat_deg * 111.0 * 1000  # in meters
    #     delta_x = delta_lon_deg * 111.0 * np.cos(np.radians(np.mean(lat_vals))) * 1000  # in meters
    # else:
    
    # Use geodesic distance for accurate spacing
    lat_dists = []
    lon_dists = []
    
    # Compute lat direction distances
    for i in range(len(lat_vals) - 1):
        dist = geodesic((lat_vals[i], lon_vals[0]), (lat_vals[i+1], lon_vals[0]), 
                        ellipsoid='WGS-84').meters
        lat_dists.append(dist)
    delta_y = np.median(lat_dists) if lat_dists else 1000  # in meters
    
    # Compute lon direction distances  
    for i in range(len(lon_vals) - 1):
        dist = geodesic((lat_vals[0], lon_vals[i]), (lat_vals[0], lon_vals[i+1]),
                        ellipsoid='WGS-84').meters
        lon_dists.append(dist)
    delta_x = np.median(lon_dists) if lon_dists else 1000  # in meters
    
    logging.info(f'Grid spacing: delta_x={delta_x:.1f} m, delta_y={delta_y:.1f} m')
    
    # Flatten spatial field
    truth_flat = truth_spatial.ravel()
    recon_flat = recon_spatial.ravel()
    
    # Remove NaN
    valid_mask = np.isfinite(truth_flat) & np.isfinite(recon_flat)
    truth_valid = truth_flat[valid_mask]
    recon_valid = recon_flat[valid_mask]
    
    # Use delta_x for spectral analysis (assuming isotropic spacing)
    fs = 1.0 / delta_x
    nperseg = max(32, min(256, len(truth_valid) // 4))
    error = truth_valid - recon_valid
    
    wavenumber, psd_error = welch(error, fs=fs, nperseg=nperseg, scaling='density', noverlap=0)
    _, psd_truth = welch(truth_valid, fs=fs, nperseg=nperseg, scaling='density', noverlap=0)
    _, coh = coherence(truth_valid, recon_valid, fs=fs, nperseg=nperseg, noverlap=0)
    
    # Compute NSR
    epsilon = np.finfo(float).eps
    nsr = np.divide(psd_error, psd_truth + epsilon, out=np.zeros_like(psd_error))

    print("NSR range: {:.4f} to {:.4f}".format(nsr.min(), nsr.max()))
    
    # Find resolution
    resolution, flag = find_resolution_crossing(wavenumber, nsr, target_ratio=target_ratio)
    wavelength = 1.0 / (wavenumber + 1e-10)
    
    logging.info(f'Effective spatial resolution: {resolution:.1f} m (at NSR={target_ratio})')
    # if flag:
    #     logging.info('Note: Multiple crossings') # or fallback method used"
    
    return {
        'wavenumber': wavenumber,
        'wavelength': wavelength,
        'nsr': nsr,
        'coherence': coh,
        'resolution': resolution,
        'delta_x': delta_x,
        'delta_y': delta_y,
        'flag': flag,
    }


if __name__ == '__main__':
    logging.info('Simplified Compression NSR Analysis Module')
    logging.info('Use: compute_nsr_along_depth() or compute_nsr_spatial_map()')

