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
from typing import Tuple, Dict, Optional, List, Any
import logging
import warnings

try:
    from joblib import Parallel, delayed
    HAS_JOBLIB = True
except ImportError:
    HAS_JOBLIB = False

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
) -> Tuple[float, bool, str]:
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
        Flag indicating multiple crossings or fallback case.
    str
        Crossing status: 'crossed', 'always_below', 'always_above', or
        'invalid_input'.
    """
    eps = 1e-10
    valid = np.isfinite(wavenumber) & np.isfinite(nsr) & (wavenumber > 0)
    if not np.any(valid):
        return 0.0, True, 'invalid_input'

    k = wavenumber[valid]
    r = nsr[valid]

    # Ensure ascending wavenumber for robust interpolation.
    order = np.argsort(k)
    k = k[order]
    r = r[order]

    s = r - target_ratio
    crossing_idx = np.where(np.diff(np.sign(s)) != 0)[0]

    if len(crossing_idx) > 0:
        # Prefer the first upward crossing (largest wavelength that fails target).
        upward_idx = np.where((s[:-1] <= 0) & (s[1:] > 0))[0]
        if len(upward_idx) > 0:
            idx = upward_idx[0]
        else:
            idx = crossing_idx[0]

        flag_multiple_crossing = len(crossing_idx) > 1

        if idx + 1 < len(k):
            # Linear interpolation in log-wavenumber space.
            s1 = s[idx]
            s2 = s[idx + 1]
            k1 = np.log(k[idx] + eps)
            k2 = np.log(k[idx + 1] + eps)

            if abs(s1 - s2) > eps:
                log_wavenumber_crossing = k1 - s1 * (k1 - k2) / (s1 - s2)
                resolution = 1.0 / (np.exp(log_wavenumber_crossing) + eps)
            else:
                resolution = 1.0 / (k[idx] + eps)
        else:
            resolution = 1.0 / (k[-1] + eps)

        return resolution, flag_multiple_crossing, 'crossed'

    if np.nanmax(r) < target_ratio:
        # Better than threshold at all resolved scales.
        resolution = 1.0 / (np.nanmax(k) + eps)
        return resolution, True, 'always_below'

    if np.nanmin(r) > target_ratio:
        # Worse than threshold at all resolved scales.
        resolution = 1.0 / (np.nanmin(k) + eps)
        return resolution, True, 'always_above'

    return 0.0, True, 'invalid_input'


def _resolve_resolutions_for_thresholds(
    wavenumber: np.ndarray,
    nsr: np.ndarray,
    target_ratio: float,
    target_ratios: Optional[List[float]] = None,
) -> Tuple[float, bool, str, Dict[str, Dict[str, Any]]]:
    """Resolve one or many NSR thresholds from an already computed NSR curve."""
    ratios: List[float] = [float(target_ratio)]
    if target_ratios is not None:
        ratios.extend(float(r) for r in target_ratios)

    # Keep insertion order while removing duplicates.
    unique_ratios = list(dict.fromkeys(ratios))

    results_by_target: Dict[str, Dict[str, Any]] = {}
    for ratio in unique_ratios:
        resolution, flag, status = find_resolution_crossing(
            wavenumber,
            nsr,
            target_ratio=ratio,
        )
        results_by_target[f"{ratio:g}"] = {
            'resolution': resolution,
            'flag': flag,
            'resolution_status': status,
        }

    selected_key = f"{float(target_ratio):g}"
    selected = results_by_target[selected_key]
    return (
        float(selected['resolution']),
        bool(selected['flag']),
        str(selected['resolution_status']),
        results_by_target,
    )


def compute_nsr_along_depth(
    truth_da: xr.DataArray,
    reconstructed_da: xr.DataArray,
    target_ratio: float = 0.5,
    target_ratios: Optional[List[float]] = None,
) -> Dict[str, Any]:
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
    target_ratios : list[float], optional
        Additional thresholds to resolve from the same NSR curve.
        NSR is computed once and crossing is evaluated for each threshold.

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
        - 'resolution_status': crossing status string
                - 'resolutions_by_target': optional dict keyed by target ratio string
                    (e.g., '0.01') containing resolution, flag, and status.
    """
    logging.info('Computing NSR along depth axis...')
    
    # Get z-axis and check uniformity
    z_vals = truth_da.z.values
    z_diffs = np.diff(z_vals)
    z_mean_diff = np.mean(z_diffs)
    z_uniform_check = np.std(z_diffs) / z_mean_diff < 0.05
    
    # Create target z-grid for interpolation only when needed.
    # If z is already uniform, keep original coordinates to avoid unnecessary resampling.
    if z_uniform_check:
        z_uniform = z_vals.astype(float)
    else:
        z_uniform = np.linspace(float(z_vals.min()), float(z_vals.max()), len(z_vals))

    delta_z = np.mean(np.diff(z_uniform))  # spacing used for spectral frequency
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
    chunk_size = max(5000, min(50000, truth_reshaped.shape[0] // 200))
    nperseg = max(32, min(128, len(z_uniform)))

    psd_error_sum = None
    psd_truth_sum = None
    coh_sum = None
    wavenumber = None
    valid_profile_count = 0
    
    if z_uniform_check:
        logging.info(
            f'Processing {truth_reshaped.shape[0]} profiles in chunks of {chunk_size} '
            '(uniform z-grid; interpolate only profiles with missing points)...'
        )
    else:
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
                # Fast path: no interpolation needed if grid is already uniform and profile is fully valid.
                if z_uniform_check and np.all(valid_mask):
                    truth_chunk_interp[j] = t_profile
                    recon_chunk_interp[j] = r_profile
                else:
                    truth_chunk_interp[j] = np.interp(
                        z_uniform,
                        z_vals[valid_mask],
                        t_profile[valid_mask],
                        left=np.nan,
                        right=np.nan,
                    )
                    recon_chunk_interp[j] = np.interp(
                        z_uniform,
                        z_vals[valid_mask],
                        r_profile[valid_mask],
                        left=np.nan,
                        right=np.nan,
                    )
            else:
                truth_chunk_interp[j] = np.nan
                recon_chunk_interp[j] = np.nan
        
        # Keep only fully finite profiles so depth spectra are valid per profile.
        prof_valid = np.isfinite(truth_chunk_interp).all(axis=1) & np.isfinite(recon_chunk_interp).all(axis=1)
        if np.any(prof_valid):
            truth_valid = truth_chunk_interp[prof_valid]
            recon_valid = recon_chunk_interp[prof_valid]
            error_valid = truth_valid - recon_valid

            w_chunk, psd_error_chunk = welch(
                error_valid,
                fs=fs,
                nperseg=nperseg,
                scaling='density',
                noverlap=0,
                axis=1,
            )
            _, psd_truth_chunk = welch(
                truth_valid,
                fs=fs,
                nperseg=nperseg,
                scaling='density',
                noverlap=0,
                axis=1,
            )
            _, coh_chunk = coherence(
                truth_valid,
                recon_valid,
                fs=fs,
                nperseg=nperseg,
                noverlap=0,
                axis=1,
            )

            if wavenumber is None:
                wavenumber = w_chunk
                psd_error_sum = np.zeros_like(wavenumber, dtype=np.float64)
                psd_truth_sum = np.zeros_like(wavenumber, dtype=np.float64)
                coh_sum = np.zeros_like(wavenumber, dtype=np.float64)

            psd_error_sum += np.nansum(psd_error_chunk, axis=0)
            psd_truth_sum += np.nansum(psd_truth_chunk, axis=0)
            coh_sum += np.nansum(coh_chunk, axis=0)
            valid_profile_count += truth_valid.shape[0]

            del truth_valid, recon_valid, error_valid, psd_error_chunk, psd_truth_chunk, coh_chunk

        del truth_chunk_interp, recon_chunk_interp

    del truth_reshaped, recon_reshaped, truth_raw, recon_raw

    if valid_profile_count == 0 or wavenumber is None:
        logging.warning('No valid finite profiles for depth NSR computation.')
        return {
            'wavenumber': np.array([]),
            'wavelength': np.array([]),
            'nsr': np.array([]),
            'coherence': np.array([]),
            'resolution': 0.0,
            'flag': True,
            'resolution_status': 'invalid_input',
        }

    psd_error = psd_error_sum / valid_profile_count
    psd_truth = psd_truth_sum / valid_profile_count
    coh = coh_sum / valid_profile_count
    
    # Compute NSR
    epsilon = np.finfo(float).eps
    nsr = np.divide(psd_error, psd_truth + epsilon, out=np.zeros_like(psd_error))

    print(f'NSR range: {nsr.min():.4f} to {nsr.max():.4f}')
    
    # Resolve one or many thresholds from a single NSR curve.
    resolution, flag, resolution_status, resolutions_by_target = _resolve_resolutions_for_thresholds(
        wavenumber,
        nsr,
        target_ratio=target_ratio,
        target_ratios=target_ratios,
    )
    wavelength = 1.0 / (wavenumber + 1e-10)
    
    logging.info(
        f'Effective resolution: {resolution:.2f} m '
        f'(at NSR={target_ratio}, status={resolution_status})'
    )
    
    return {
        'wavenumber': wavenumber,
        'wavelength': wavelength,
        'nsr': nsr,
        'coherence': coh,
        'resolution': resolution,
        'flag': flag,
        'resolution_status': resolution_status,
        'resolutions_by_target': resolutions_by_target,
    }


def compute_nsr_spatial_map(
    truth_da: xr.DataArray,
    reconstructed_da: xr.DataArray,
    target_ratio: float = 0.5,
    target_ratios: Optional[List[float]] = None,
) -> Dict[str, Any]:
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
    target_ratios : list[float], optional
        Additional thresholds to resolve from the same NSR curve.
        NSR is computed once and crossing is evaluated for each threshold.

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
        - 'resolution_status': crossing status string
                - 'resolutions_by_target': optional dict keyed by target ratio string
                    (e.g., '0.01') containing resolution, flag, and status.
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
    
    # Resolve one or many thresholds from a single NSR curve.
    resolution, flag, resolution_status, resolutions_by_target = _resolve_resolutions_for_thresholds(
        wavenumber,
        nsr,
        target_ratio=target_ratio,
        target_ratios=target_ratios,
    )
    wavelength = 1.0 / (wavenumber + 1e-10)
    
    logging.info(
        f'Effective spatial resolution: {resolution:.1f} m '
        f'(at NSR={target_ratio}, status={resolution_status})'
    )
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
        'resolution_status': resolution_status,
        'resolutions_by_target': resolutions_by_target,
    }


def compute_nsr_spatial_resolution_stats_by_depth_time(
    truth_da: xr.DataArray,
    reconstructed_da: xr.DataArray,
    depth_indices: List[int],
    target_ratio: float = 0.5,
    target_ratios: Optional[List[float]] = None,
    n_jobs: int = -1,
) -> Dict[str, Any]:
    """
    Compute spatial NSR resolutions for specific depth indices at every time step.

    For each requested depth index, this computes one spatial resolution per time slice
    from the 2D (lat, lon) map, then returns summary statistics across time.
    """
    logging.info(
        'Computing spatial NSR resolution stats by depth/time '
        f'(depth_indices={depth_indices}, n_jobs={n_jobs})...'
    )

    if len(depth_indices) == 0:
        return {
            'delta_x': 0.0,
            'delta_y': 0.0,
            'target_ratios': [],
            'stats_by_depth': {},
        }

    # Build threshold list once.
    ratios: List[float] = [float(target_ratio)]
    if target_ratios is not None:
        ratios.extend(float(r) for r in target_ratios)
    ratios = list(dict.fromkeys(ratios))

    # Compute grid spacing from coordinates.
    lat_vals = truth_da.lat.values
    lon_vals = truth_da.lon.values

    if geodesic is None:
        raise ImportError('geopy is required for spatial NSR computations (missing geodesic).')

    lat_dists = []
    lon_dists = []
    for i in range(len(lat_vals) - 1):
        lat_dists.append(
            geodesic(
                (lat_vals[i], lon_vals[0]),
                (lat_vals[i + 1], lon_vals[0]),
                ellipsoid='WGS-84',
            ).meters
        )
    for i in range(len(lon_vals) - 1):
        lon_dists.append(
            geodesic(
                (lat_vals[0], lon_vals[i]),
                (lat_vals[0], lon_vals[i + 1]),
                ellipsoid='WGS-84',
            ).meters
        )

    delta_y = np.median(lat_dists) if lat_dists else 1000.0
    delta_x = np.median(lon_dists) if lon_dists else 1000.0
    fs = 1.0 / delta_x

    truth_np = truth_da.values
    recon_np = reconstructed_da.values
    n_time, n_depth, _, _ = truth_np.shape

    # Normalize depth indices and validate range.
    normalized_depth_indices: List[int] = []
    for d in depth_indices:
        d_idx = int(d)
        if d_idx < 0:
            d_idx = n_depth + d_idx
        if 0 <= d_idx < n_depth:
            normalized_depth_indices.append(d_idx)

    def _compute_resolution_for_map(truth_map: np.ndarray, recon_map: np.ndarray) -> Dict[str, float]:
        truth_flat = truth_map.ravel()
        recon_flat = recon_map.ravel()
        valid_mask = np.isfinite(truth_flat) & np.isfinite(recon_flat)

        truth_valid = truth_flat[valid_mask]
        recon_valid = recon_flat[valid_mask]
        if truth_valid.size < 8:
            return {f'{r:g}': np.nan for r in ratios}

        nperseg = max(32, min(256, truth_valid.size // 4))
        error = truth_valid - recon_valid

        wavenumber, psd_error = welch(error, fs=fs, nperseg=nperseg, scaling='density', noverlap=0)
        _, psd_truth = welch(truth_valid, fs=fs, nperseg=nperseg, scaling='density', noverlap=0)

        epsilon = np.finfo(float).eps
        nsr = np.divide(psd_error, psd_truth + epsilon, out=np.zeros_like(psd_error))

        resolutions: Dict[str, float] = {}
        for r in ratios:
            resolution, _, _ = find_resolution_crossing(wavenumber, nsr, target_ratio=r)
            resolutions[f'{r:g}'] = float(resolution)
        return resolutions

    stats_by_depth: Dict[str, Any] = {}

    for d_idx in normalized_depth_indices:
        if HAS_JOBLIB and n_jobs != 1:
            per_time_results = Parallel(n_jobs=n_jobs, prefer='threads')(
                delayed(_compute_resolution_for_map)(truth_np[t, d_idx], recon_np[t, d_idx])
                for t in range(n_time)
            )
        else:
            per_time_results = [
                _compute_resolution_for_map(truth_np[t, d_idx], recon_np[t, d_idx])
                for t in range(n_time)
            ]

        depth_stats: Dict[str, Any] = {'resolutions_by_target': {}}
        for r in ratios:
            ratio_key = f'{r:g}'
            vals = np.array([res[ratio_key] for res in per_time_results], dtype=np.float64)
            finite_vals = vals[np.isfinite(vals)]

            if finite_vals.size == 0:
                depth_stats['resolutions_by_target'][ratio_key] = {
                    'count': 0,
                    'mean': np.nan,
                    'median': np.nan,
                    'std': np.nan,
                    'iqr': np.nan,
                }
            else:
                q25, q75 = np.percentile(finite_vals, [25, 75])
                depth_stats['resolutions_by_target'][ratio_key] = {
                    'count': int(finite_vals.size),
                    'mean': float(np.mean(finite_vals)),
                    'median': float(np.median(finite_vals)),
                    'std': float(np.std(finite_vals)),
                    'iqr': float(q75 - q25),
                }

        stats_by_depth[str(d_idx)] = depth_stats

    return {
        'delta_x': float(delta_x),
        'delta_y': float(delta_y),
        'target_ratios': [float(r) for r in ratios],
        'stats_by_depth': stats_by_depth,
    }


if __name__ == '__main__':
    logging.info('Simplified Compression NSR Analysis Module')
    logging.info('Use: compute_nsr_along_depth() or compute_nsr_spatial_map()')

