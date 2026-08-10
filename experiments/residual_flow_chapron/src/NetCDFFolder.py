import os
from pathlib import Path
import xarray as xr
import numpy as np
import torch
from torch.utils.data import Dataset
import pytorch_lightning as pl
from collections import namedtuple
from typing import Any, Dict, Optional, Tuple
import torch.nn.functional as F

TrainingItem = namedtuple("TrainingItem", ["input", "tgt", "valid_mask"], defaults=(None,))

class NetCDFFolder2D(Dataset):
    _HW_ORDER = {
        "lat": ("z", "lon"),   # H=z, W=lon
        "lon": ("z", "lat"),   # H=z, W=lat
        "z":   ("lat", "lon"), # H=lat, W=lon
    }
    _GROUP_CHANNELS = 3

    def __init__(
        self,
        root,
        var_name,
        split="train",
        select_time=None,
        nan_value = np.nan,  
        fill_nan_for_nn = 0.0,
        normalize=None,
        clip_percentiles=None,
        clip_bounds=None,
        soft_clip_tanh: bool = False,
        soft_clip_scale: Optional[float] = None,
        lower_squash_percentile: Optional[float] = None,
        patch_size=None, 
        return_meta=False,
        norm_stats=None,              # dict: {"mean": float, "std": float}
        rgb=True, 
        slice_dim=None,
        slice_len=3,
        slice_stride=None,
        slice_mode="random",
        slice_start=None,
        eps: float = 1e-6,
        z_coord_name: str = "z",
    ):
        self.root = Path(root)
        self.var_name = var_name
        self.split = split
        self.select_time = select_time
        self.nan_value = nan_value
        self.fill_nan_for_nn = fill_nan_for_nn
        self.normalize = normalize
        self.clip_percentiles = clip_percentiles
        self.clip_bounds = clip_bounds
        self.soft_clip_tanh = bool(soft_clip_tanh)
        self.soft_clip_scale = soft_clip_scale
        self.lower_squash_percentile = lower_squash_percentile
        self.clip_log = None
        self.patch_size = patch_size
        self.return_meta = return_meta
        self.norm_stats = norm_stats
        self.eps = eps
        self.rgb = rgb
        self.slice_dim = slice_dim
        self.slice_len = slice_len
        self.slice_stride = int(slice_stride) if slice_stride is not None else self._GROUP_CHANNELS
        if self.slice_stride <= 0:
            raise ValueError(f"slice_stride must be > 0, got {self.slice_stride}")
        self.slice_mode = slice_mode
        self.slice_start = slice_start
        self.z_coord_name = z_coord_name

        splitdir = self.root / split
        
        self.files = sorted([f for f in splitdir.iterdir() if f.is_file() and f.suffix == ".nc"])
        
        if not self.files:
            raise RuntimeError(f"No .nc files found in {splitdir}")

        # If input is 3D, auto-detect slice_len from the first file's slice_dim size
        # and expose multiple dataset samples per file (each with exactly 3 channels).
        self._is_3d_input = self._probe_3d_and_set_slice_len()
        self._groups_per_file = len(self._build_channel_groups(self.slice_len)) if self._is_3d_input else 1

        # Probe z-coordinate values for depth-aware gradient support.
        # z_coords is set only when H is the depth dimension (slice_dim in lat/lon).
        self.z_coords: Optional[np.ndarray] = self._probe_z_coords()
        if self.z_coords is not None:
            print(
                f"[NetCDFFolder2D/{self.split}] z-coords probed: "
                f"{len(self.z_coords)} levels, range [{float(self.z_coords.min()):.2f}, "
                f"{float(self.z_coords.max()):.2f}]"
            )
        
        if self.normalize is True:
            self.normalize = "zscore"
        elif self.normalize is False:
            self.normalize = None

        self.lower_squash_bound: Optional[float] = None
        self.lower_squash_scale: Optional[float] = None
        if self.lower_squash_percentile is not None:
            self.lower_squash_percentile = float(self.lower_squash_percentile)
            if not (0.0 < self.lower_squash_percentile < 100.0):
                raise ValueError(
                    f"lower_squash_percentile must satisfy 0 < q < 100, got {self.lower_squash_percentile}"
                )
        elif self.normalize in ("zscore_squash_q1", "standard_squash_q1", "zscore_squash", "standard_squash"):
            self.lower_squash_percentile = 1.0

        if self.clip_percentiles is not None:
            if len(self.clip_percentiles) != 2:
                raise ValueError("clip_percentiles must be a tuple(low, high)")
            plo, phi = float(self.clip_percentiles[0]), float(self.clip_percentiles[1])
            if not (0.0 <= plo < phi <= 100.0):
                raise ValueError(
                    f"Invalid clip_percentiles={self.clip_percentiles!r}; expected 0 <= low < high <= 100"
                )
            self.clip_percentiles = (plo, phi)

        if self.clip_percentiles is not None and self.clip_bounds is None:
            self.clip_bounds = self._compute_global_percentile_bounds()
            print(
                f"[NetCDFFolder2D/{self.split}] Clip bounds from percentiles {self.clip_percentiles}: {self.clip_bounds}; "
                f"clip fractions low/high/total={self.clip_log}"
            )
        if self.soft_clip_scale is not None:
            self.soft_clip_scale = float(self.soft_clip_scale)
            if not np.isfinite(self.soft_clip_scale) or self.soft_clip_scale <= 0.0:
                raise ValueError(f"soft_clip_scale must be > 0, got {self.soft_clip_scale}")

        if self.soft_clip_tanh and self.clip_bounds is not None:
            print(
                f"[NetCDFFolder2D/{self.split}] Soft clipping enabled (tanh) with "
                f"clip_bounds={self.clip_bounds}, soft_clip_scale={self.soft_clip_scale}"
            )

        if self.lower_squash_percentile is not None:
            ref_p = min(99.0, self.lower_squash_percentile + 9.0)
            self.lower_squash_bound, self.lower_squash_scale = self._compute_global_lower_squash_params(
                low_percentile=self.lower_squash_percentile,
                ref_percentile=ref_p,
            )
            print(
                f"[NetCDFFolder2D/{self.split}] Lower-tail squash enabled at P{self.lower_squash_percentile}: "
                f"bound={self.lower_squash_bound}, scale={self.lower_squash_scale}"
            )

        needs_mean_std = self.normalize in (
            "standard", "zscore", "centered_std",
            "zscore_squash_q1", "standard_squash_q1",
            "zscore_squash", "standard_squash",
        )
        needs_robust = self.normalize in ("robust", "iqr", "median_iqr")
        needs_min_max = self.normalize in ("minmax", "minmax01")

        if (needs_mean_std or needs_min_max or needs_robust) and (self.norm_stats is None):
            if needs_min_max:
                self.norm_stats = self._compute_global_min_max()
            elif needs_robust:
                self.norm_stats = self._compute_global_median_iqr()
            else:
                self.norm_stats = self._compute_global_mean_std()
            print("Norm stats:", self.norm_stats)
    
    def __len__(self):
        return len(self.files) * self._groups_per_file

    def _probe_3d_and_set_slice_len(self) -> bool:
        """Probe the first file: check if 3-D and auto-set slice_len to the
        full extent of slice_dim so every slice is covered."""
        if len(self.files) == 0:
            return False
        fp = self.files[0]
        with xr.open_dataset(fp) as ds:
            da = ds[self.var_name]
            if self.select_time is not None and "time" in da.dims:
                da = da.isel(time=self.select_time)
            is_3d = int(da.ndim) >= 3
            if is_3d and self.slice_dim is not None and self.slice_dim in da.dims:
                dim_size = da.sizes[self.slice_dim]
                self.slice_len = dim_size
                print(
                    f"[NetCDFFolder2D/{self.split}] Auto slice_len={dim_size} "
                    f"(full extent of '{self.slice_dim}', "
                    f"stride={self.slice_stride}, "
                    f"{len(self._build_channel_groups(dim_size))} groups of {self._GROUP_CHANNELS})"
                )
            return is_3d

    def _probe_z_coords(self) -> Optional[np.ndarray]:
        """Extract z-coordinate values from the first file.

        Returns a 1-D float32 array of depth levels if:
          - the file has a coordinate named `self.z_coord_name`, AND
          - that coordinate sits in the H position of the tensor layout, i.e.
            `slice_dim` is "lat" or "lon" (so H = z).
        Returns None otherwise (e.g. slice_dim="z", 2-D data, or missing coord).
        """
        # z is in H only when we slice along lat or lon
        if self.slice_dim not in ("lat", "lon"):
            return None
        if not self.files:
            return None
        fp = self.files[0]
        with xr.open_dataset(fp) as ds:
            da = ds[self.var_name]
            if self.select_time is not None and "time" in da.dims:
                da = da.isel(time=self.select_time)
            zname = self.z_coord_name
            if zname in da.coords:
                return np.asarray(da.coords[zname].values, dtype=np.float32)
            elif zname in da.dims:
                # No coordinate values stored: fall back to integer indices
                return np.arange(da.sizes[zname], dtype=np.float32)
        return None

    def _build_channel_groups(self, total_len: int) -> list:
        """
        Build 3-channel windows. With the default stride=3 this preserves the
        historical non-overlapping grouping plus a tail window. Larger strides
        sample fewer windows, e.g. stride=20 gives [0,1,2], [20,21,22], ...
        """
        if total_len <= self._GROUP_CHANNELS:
            return [list(range(total_len))]

        max_start = total_len - self._GROUP_CHANNELS
        groups = [
            list(range(start, start + self._GROUP_CHANNELS))
            for start in range(0, max_start + 1, self.slice_stride)
        ]

        if self.slice_stride == self._GROUP_CHANNELS and total_len % self._GROUP_CHANNELS != 0:
            tail = list(range(total_len - self._GROUP_CHANNELS, total_len))
            if groups[-1] != tail:
                groups.append(tail)

        return groups

    def _apply_clip_transform(self, arr: np.ndarray) -> np.ndarray:
        if self.clip_bounds is None:
            return arr
        vlow, vhigh = float(self.clip_bounds[0]), float(self.clip_bounds[1])
        if not self.soft_clip_tanh:
            return np.clip(arr, vlow, vhigh)

        center = 0.5 * (vlow + vhigh)
        half_range = max(0.5 * (vhigh - vlow), self.eps)
        scale = float(self.soft_clip_scale) if self.soft_clip_scale is not None else half_range
        scale = max(scale, self.eps)
        return center + half_range * np.tanh((arr - center) / scale)

    def _compute_global_lower_squash_params(self, low_percentile: float = 1.0, ref_percentile: float = 10.0) -> Tuple[float, float]:
        vals = []
        for fp in self.files:
            with xr.open_dataset(fp) as ds:
                da = ds[self.var_name]
                if self.select_time is not None and "time" in da.dims:
                    da = da.isel(time=self.select_time)
                arr = da.values.astype(np.float64, copy=False)

            if not np.isnan(self.nan_value):
                arr = np.where(arr == float(self.nan_value), np.nan, arr)
            if self.clip_bounds is not None:
                arr = self._apply_clip_transform(arr)

            m = np.isfinite(arr)
            if m.sum() == 0:
                continue
            vals.append(arr[m])

        if not vals:
            raise RuntimeError("No valid values found to compute lower-tail squash parameters")

        data = np.concatenate(vals)
        q_low, q_ref = np.percentile(data, [float(low_percentile), float(ref_percentile)])
        q_low = float(q_low)
        q_ref = float(q_ref)
        scale = max(float(q_ref - q_low), self.eps)
        return q_low, scale

    def _apply_lower_tail_squash_transform(self, arr: np.ndarray) -> np.ndarray:
        if self.lower_squash_bound is None:
            return arr
        bound = float(self.lower_squash_bound)
        scale = max(float(self.lower_squash_scale) if self.lower_squash_scale is not None else self.eps, self.eps)
        below = arr < bound
        if not np.any(below):
            return arr
        out = arr.copy()
        d = (bound - out[below]) / scale
        out[below] = bound - scale * np.tanh(d)
        return out

    def _compute_global_mean_std(self):
        """
        Compute dataset-wide mean/std over all .nc files.
        This mirrors BaseDataModule.train_mean_std().
        """
        total_sum = 0.0
        total_sumsq = 0.0
        total_count = 0

        for fp in self.files:
            ds = xr.open_dataset(fp)
            da = ds[self.var_name]

            if self.select_time is not None and "time" in da.dims:
                da = da.isel(time=self.select_time)

            if da.ndim < 2:
                raise RuntimeError(
                    f"{fp}: expected >=2D, got dims={da.dims}, shape={da.shape}"
                )

            arr = da.values.astype(np.float64, copy=False)
            if not np.isnan(self.nan_value):
                arr = np.where(arr == float(self.nan_value), np.nan, arr)
            if self.clip_bounds is not None:
                arr = self._apply_clip_transform(arr)
            if self.lower_squash_bound is not None:
                arr = self._apply_lower_tail_squash_transform(arr)
            m = np.isfinite(arr)
            if m.sum() == 0:
                continue
            v = arr[m]
            total_sum += float(v.sum())
            total_sumsq += float((v * v).sum())
            total_count += int(v.size)

        if total_count == 0:
            raise RuntimeError("No valid values found to compute norm_stats")

        mean = total_sum / total_count
        var = total_sumsq / total_count - mean * mean
        std = float(np.sqrt(max(var, 0.0)))

        return (float(mean), float(std))

    def _compute_global_min_max(self):
        """Compute dataset-wide min/max over all .nc files (ignoring NaNs)."""
        vmin = None
        vmax = None

        for fp in self.files:
            ds = xr.open_dataset(fp)
            da = ds[self.var_name]

            if self.select_time is not None and "time" in da.dims:
                da = da.isel(time=self.select_time)

            if da.ndim < 2:
                raise RuntimeError(
                    f"{fp}: expected >=2D, got dims={da.dims}, shape={da.shape}"
                )

            arr = da.values.astype(np.float64, copy=False)
            if not np.isnan(self.nan_value):
                arr = np.where(arr == float(self.nan_value), np.nan, arr)
            if self.clip_bounds is not None:
                arr = self._apply_clip_transform(arr)
            m = np.isfinite(arr)
            if m.sum() == 0:
                continue
            v = arr[m]
            mn = float(v.min())
            mx = float(v.max())
            vmin = mn if vmin is None else min(vmin, mn)
            vmax = mx if vmax is None else max(vmax, mx)

        if vmin is None or vmax is None:
            raise RuntimeError("No valid values found to compute min/max")

        return (float(vmin), float(vmax))

    def _compute_global_median_iqr(self):
        """Compute dataset-wide median and IQR over all .nc files (ignoring NaNs)."""
        vals = []
        for fp in self.files:
            ds = xr.open_dataset(fp)
            da = ds[self.var_name]

            if self.select_time is not None and "time" in da.dims:
                da = da.isel(time=self.select_time)

            if da.ndim < 2:
                raise RuntimeError(
                    f"{fp}: expected >=2D, got dims={da.dims}, shape={da.shape}"
                )

            arr = da.values.astype(np.float64, copy=False)
            if not np.isnan(self.nan_value):
                arr = np.where(arr == float(self.nan_value), np.nan, arr)
            if self.clip_bounds is not None:
                arr = self._apply_clip_transform(arr)
            m = np.isfinite(arr)
            if m.sum() == 0:
                continue
            vals.append(arr[m])

        if not vals:
            raise RuntimeError("No valid values found to compute robust norm stats")

        data = np.concatenate(vals)
        median = float(np.median(data))
        q25, q75 = np.percentile(data, [25.0, 75.0])
        iqr = float(max(float(q75) - float(q25), self.eps))
        return (median, iqr)

    def _compute_global_percentile_bounds(self):
        vals = []
        for fp in self.files:
            ds = xr.open_dataset(fp)
            da = ds[self.var_name]

            if self.select_time is not None and "time" in da.dims:
                da = da.isel(time=self.select_time)

            arr = da.values.astype(np.float64, copy=False)
            if not np.isnan(self.nan_value):
                arr = np.where(arr == float(self.nan_value), np.nan, arr)
            m = np.isfinite(arr)
            if m.sum() == 0:
                continue
            vals.append(arr[m])

        if not vals:
            raise RuntimeError("No valid values found to compute percentile clip bounds")

        data = np.concatenate(vals)
        plo, phi = self.clip_percentiles
        vlow, vhigh = np.percentile(data, [plo, phi])
        vlow, vhigh = float(vlow), float(vhigh)

        if not (np.isfinite(vlow) and np.isfinite(vhigh)):
            raise RuntimeError("Computed non-finite percentile clip bounds")
        if not (vlow < vhigh):
            raise RuntimeError(
                f"Degenerate percentile clip bounds: ({vlow}, {vhigh}) from clip_percentiles={self.clip_percentiles}"
            )

        total = int(data.size)
        n_low = int(np.sum(data < vlow))
        n_high = int(np.sum(data > vhigh))
        self.clip_log = (
            round(n_low / total, 6),
            round(n_high / total, 6),
            round((n_low + n_high) / total, 6),
        )

        return vlow, vhigh

    def _standardize(self, t):
        if self.per_sample_norm:
            mean = t.mean(dim=(-2, -1), keepdim=True)
            std  = t.std(dim=(-2, -1), keepdim=True)
        else:
            if self.norm_stats is None:
                raise RuntimeError("norm_stats is None")

            if isinstance(self.norm_stats, dict):
                m, s = self.norm_stats["mean"], self.norm_stats["std"]
            else:
                m, s = self.norm_stats  # (mean, std)

            mean = torch.as_tensor(m, dtype=t.dtype).view(1, 1, 1)
            std  = torch.as_tensor(s, dtype=t.dtype).view(1, 1, 1)

        t_norm = (t - mean) / (std + self.eps)
        return t_norm, mean, std

    def _load_full(self, fp: Path) -> np.ndarray:
        with xr.open_dataset(fp) as ds:
            da = ds[self.var_name]
            if self.select_time is not None and "time" in da.dims:
                da = da.isel(time=self.select_time)
            if da.ndim < 2:
                raise RuntimeError(f"{fp}: expected >=2D, got dims={da.dims}, shape={da.shape}")
            arr = da.values.astype(np.float32, copy=False)
        # Optionally convert sentinel values to NaN so downstream masking/stats ignore them.
        if not np.isnan(self.nan_value):
            arr = np.where(arr == float(self.nan_value), np.nan, arr)
        return arr

    def _pick_slice_dim(self, da: xr.DataArray) -> Optional[str]:
        """
        Choose which dim becomes the channel dim (length = slice_len).
        If slice_dim is None or "min", prefer the smallest among (lat, lon)
        so that W stays aligned with z in the resulting (C, H, W) layout.
        Fall back to z only when lat/lon are not present.
        """
        candidates = [d for d in ("z", "lat", "lon") if d in da.dims]
        if not candidates:
            return None

        if self.slice_dim is None or self.slice_dim == "min":
            preferred = [d for d in ("lat", "lon") if d in da.dims]
            if preferred:
                sizes = {d: int(da.sizes[d]) for d in preferred}
                return min(sizes, key=sizes.get)
            sizes = {d: int(da.sizes[d]) for d in candidates}
            return min(sizes, key=sizes.get)

        if self.slice_dim not in da.dims:
            raise RuntimeError(f"slice_dim={self.slice_dim} not found in dims={da.dims}")

        return self.slice_dim

    def _select_slices(self, da: xr.DataArray, group_idx: int = 0) -> Tuple[xr.DataArray, Dict[str, Any]]:
        # If 2D already -> no slicing needed
        if da.ndim == 2:
            dims_out = da.dims
            coords_out = {d: np.asarray(da.coords[d].values) for d in dims_out if d in da.coords}
            meta = {
                "dims": dims_out,
                "coords": coords_out,
                "var_name": self.var_name,
                "slice_dim": None,
                "slice_indices": None,
            }
            return da, meta

        # Expect (z,lat,lon) after optional time selection; allow extra singleton dims (e.g., time=1)
        slice_dim = self._pick_slice_dim(da)
        if slice_dim is None:
            raise RuntimeError(f"{self.var_name}: cannot auto-pick slice dim from dims={da.dims}")

        # Drop extra singleton dims except slice_dim
        if da.ndim > 3:
            for d in list(da.dims):
                if d == slice_dim:
                    continue
                if int(da.sizes[d]) == 1:
                    da = da.isel({d: 0})
            if da.ndim != 3:
                raise RuntimeError(
                    f"{self.var_name}: expected 3D after removing singleton dims, got dims={da.dims}"
                )

        if da.ndim != 3:
            raise RuntimeError(f"{self.var_name}: expected 2D or 3D, got dims={da.dims}")

        # Now da is 3D
        if slice_dim not in self._HW_ORDER:
            raise RuntimeError(f"Unsupported slice_dim={slice_dim}; expected one of {tuple(self._HW_ORDER.keys())}")

        expected_hw = self._HW_ORDER[slice_dim]
        missing_hw = [d for d in expected_hw if d not in da.dims]
        if missing_hw:
            raise RuntimeError(
                f"{self.var_name}: expected dims {expected_hw} for slice_dim={slice_dim}, "
                f"but missing {missing_hw} in dims={da.dims}"
            )

        other_dims = [expected_hw[0], expected_hw[1]]

        n = int(da.sizes[slice_dim])
        if n < self.slice_len:
            raise RuntimeError(
                f"{self.var_name}: slice_dim={slice_dim} size {n} < slice_len {self.slice_len}"
            )

        # choose start index
        if self.slice_mode == "random":
            start = np.random.randint(0, n - self.slice_len + 1)
        elif self.slice_mode == "center":
            start = (n - self.slice_len) // 2
        elif self.slice_mode == "fixed":
            if self.slice_start is None:
                raise RuntimeError("slice_start must be set when slice_mode='fixed'")
            start = int(self.slice_start)
            if not (0 <= start <= n - self.slice_len):
                raise RuntimeError(f"slice_start={start} invalid for size={n} and slice_len={self.slice_len}")
        else:
            raise ValueError(f"Unknown slice_mode: {self.slice_mode}")

        groups = [
            [start + offset for offset in group]
            for group in self._build_channel_groups(self.slice_len)
        ]
        if group_idx < 0:
            raise RuntimeError(f"group_idx must be >= 0, got {group_idx}")
        if group_idx >= len(groups):
            raise RuntimeError(
                f"group_idx={group_idx} out of range for {len(groups)} groups "
                f"(slice_len={self.slice_len}, groups={groups})"
            )
        selected_indices = groups[group_idx]

        # Keep slice_dim as first dim => (C, H, W), with fixed H/W order
        da_s = da.isel({slice_dim: selected_indices}).transpose(slice_dim, other_dims[0], other_dims[1])

        coords_out: Dict[str, Any] = {}
        for d in da_s.dims:
            c = da_s.coords.get(d, None)
            if c is not None and c.ndim == 1:
                coords_out[d] = np.asarray(c.values)

        meta = {
            "dims": da_s.dims,            # (slice_dim, other0, other1)
            "coords": coords_out,
            "var_name": self.var_name,
            "slice_dim": slice_dim,
            "slice_indices": selected_indices,
            "slice_indices_full": [idx for group in groups for idx in group],
            "slice_group_index": group_idx,
            "slice_num_groups": len(groups),
            "spatial_dims": (other_dims[0], other_dims[1]),
        }
        return da_s, meta

    def __getitem__(self, idx):
        if self._groups_per_file > 1:
            file_idx = idx // self._groups_per_file
            group_idx = idx % self._groups_per_file
        else:
            file_idx = idx
            group_idx = 0

        fp = self.files[file_idx]
        with xr.open_dataset(fp) as ds:
            da = ds[self.var_name]
            if self.select_time is not None and "time" in da.dims:
                da = da.isel(time=self.select_time)
            da, meta = self._select_slices(da, group_idx=group_idx)
            arr = da.values.astype(np.float32, copy=False)

        # Convert sentinel values to NaN before normalization and mask creation.
        if not np.isnan(self.nan_value):
            arr = np.where(arr == float(self.nan_value), np.nan, arr)

        # crop BEFORE turning into torch
        if self.patch_size is not None:
            if arr.ndim == 2:
                H, W = arr.shape
            else:
                _, H, W = arr.shape
            ps = self.patch_size
            top = np.random.randint(0, H - ps + 1)
            left = np.random.randint(0, W - ps + 1)
            if arr.ndim == 2:
                arr = arr[top:top+ps, left:left+ps]
            else:
                arr = arr[:, top:top+ps, left:left+ps]

        # Clip outliers BEFORE normalization
        if self.clip_bounds is not None:
            arr = self._apply_clip_transform(arr)
        if self.lower_squash_bound is not None:
            arr = self._apply_lower_tail_squash_transform(arr)

        # normalize using GLOBAL stats
        if self.normalize:
            if self.norm_stats is None:
                raise RuntimeError("norm_stats is None")
            if self.normalize in ("standard", "zscore", "centered_std"):
                m, s = self.norm_stats
                s = max(float(s), self.eps)
                arr = (arr - float(m)) / s
            elif self.normalize in (
                "zscore_squash_q1", "standard_squash_q1",
                "zscore_squash", "standard_squash",
            ):
                m, s = self.norm_stats
                s = max(float(s), self.eps)
                z = (arr - float(m)) / s
                # Keep squash modes in [0,1] to match min-max-like input range.
                z = np.clip(z, -30.0, 30.0)
                arr = 1.0 / (1.0 + np.exp(-z))
            elif self.normalize in ("robust", "iqr", "median_iqr"):
                med, iqr = self.norm_stats
                iqr = max(float(iqr), self.eps)
                arr = (arr - float(med)) / iqr
            elif self.normalize in ("minmax", "minmax01"):
                vmin, vmax = self.norm_stats
                denom = max(float(vmax) - float(vmin), self.eps)
                arr = (arr - float(vmin)) / denom
            else:
                raise ValueError(f"Unknown normalize mode: {self.normalize}")

        # build validity mask BEFORE filling NaNs
        valid = np.isfinite(arr)

        # fill NaNs AFTER normalization
        arr = np.nan_to_num(arr, nan=float(self.fill_nan_for_nn)).astype(np.float32, copy=False)

        # Convert to torch with channels first
        if arr.ndim == 2:
            # (H,W) -> (1,H,W)
            t = torch.from_numpy(np.ascontiguousarray(arr)).unsqueeze(0)
            valid_t = torch.from_numpy(np.ascontiguousarray(valid.astype(np.bool_))).unsqueeze(0)
        else:
            # (C,H,W) already
            t = torch.from_numpy(np.ascontiguousarray(arr))
            valid_t = torch.from_numpy(np.ascontiguousarray(valid.astype(np.bool_)))

        # Ensure 3 channels if rgb=True
        if self.rgb and self.slice_dim is None:
            # Only duplicate when NOT using multi-slice mode
            if t.shape[0] == 1:
                t = t.repeat(3, 1, 1)
                valid_t = valid_t.repeat(3, 1, 1)
            elif t.shape[0] != 3:
                raise RuntimeError(
                    f"rgb=True but channel dim is {t.shape[0]} (expected 1 or 3). "
                    f"Set slice_len=3 or rgb=False."
                )

        item = TrainingItem(input=t, tgt=t, valid_mask=valid_t)

        if not self.return_meta:
            return item

        meta.update({
            "path": str(fp),
            "normalize": self.normalize,
            "out_shape": tuple(t.shape),
        })
        # (keep your norm info as you already do)
        return item, meta

class NetCDFVolume2DSlices(Dataset):
    """
    Yields slices along `slice_dim`. Layout rules:

      slice_dim="lat" -> (C=n_lat, H=n_lon, W=n_z)
      slice_dim="lon" -> (C=n_lon, H=n_lat, W=n_z)
      slice_dim="z"   -> (C=n_z,   H=n_lat, W=n_lon)

    For lat/lon slices: z is always W (innermost/fastest varying dim).
    For z   slices:     lon is always W.
    """

    # hw_dims order (H, W) for each slice_dim
    _HW_ORDER = {
        "lat": ("lon", "z"),
        "lon": ("lat", "z"),
        "z":   ("lat", "lon"),
    }

    def __init__(
        self,
        root,
        var_name,
        split="test",
        slice_dim="lat",
        slice_len=1,
        normalize=None,
        norm_stats=None,
        eps=1e-6,
        fill_nan=0.0,
        lat_name="lat",
        lon_name="lon",
        z_name="z",
    ):
        self.root = Path(root)
        self.var_name = var_name
        self.split = split
        self.slice_dim = slice_dim
        self.slice_len = slice_len
        self.normalize = normalize
        self.norm_stats = norm_stats
        self.eps = eps
        self.fill_nan = fill_nan
        self.lat_name = lat_name
        self.lon_name = lon_name
        self.z_name = z_name

        # Map abstract dim names ("lat","lon","z") to actual NC dim names
        self._name_map = {"lat": lat_name, "lon": lon_name, "z": z_name}

        all_dim_names = [lat_name, lon_name, z_name]
        if slice_dim not in all_dim_names:
            raise ValueError(f"slice_dim={slice_dim!r} must be one of {all_dim_names}")

        # Resolve H and W dim names using the fixed ordering table
        abstract_key = {lat_name: "lat", lon_name: "lon", z_name: "z"}[slice_dim]
        h_abstract, w_abstract = self._HW_ORDER[abstract_key]
        self.hw_dims = (self._name_map[h_abstract], self._name_map[w_abstract])  # (H_name, W_name)

        splitdir = self.root / split
        self.files = sorted(f for f in splitdir.iterdir() if f.suffix == ".nc")
        if not self.files:
            raise RuntimeError(f"No .nc files found in {splitdir}")

        self.index = []
        self.file_meta = []

        for fi, fp in enumerate(self.files):
            with xr.open_dataset(fp) as ds:
                da = ds[var_name]
                self._validate_dims(da, fp)
                fm = self._extract_file_meta(da, fi, fp)
            self.file_meta.append(fm)
            n = fm["n_slice"]
            for si in range(0, n - slice_len + 1):
                self.index.append((fi, si))

    # ------------------------------------------------------------------
    def _validate_dims(self, da: xr.DataArray, fp: Path):
        required = {self.lat_name, self.lon_name, self.z_name}
        missing = required - set(da.dims)
        if missing:
            raise RuntimeError(f"{fp}: missing dims {missing}, found {set(da.dims)}")

    def _extract_file_meta(self, da: xr.DataArray, fi: int, fp: Path) -> dict:
        def _coord(name):
            return np.asarray(da.coords[name].values) if name in da.coords else np.arange(da.sizes[name])

        return {
            "file_idx": fi,
            "path":     str(fp),
            "n_slice":  da.sizes[self.slice_dim],
            self.lat_name: _coord(self.lat_name),
            self.lon_name: _coord(self.lon_name),
            self.z_name:   _coord(self.z_name),
        }

    # ------------------------------------------------------------------
    def _normalize(self, arr: np.ndarray) -> np.ndarray:
        if not self.normalize or self.norm_stats is None:
            return arr
        if self.normalize in ("zscore", "standard"):
            m, s = self.norm_stats
            return (arr - float(m)) / max(float(s), self.eps)
        if self.normalize in ("robust", "iqr", "median_iqr"):
            med, iqr = self.norm_stats
            return (arr - float(med)) / max(float(iqr), self.eps)
        if self.normalize in ("minmax", "minmax01"):
            vmin, vmax = self.norm_stats
            return (arr - float(vmin)) / max(float(vmax) - float(vmin), self.eps)
        raise ValueError(f"Unknown normalize mode: {self.normalize}")

    # ------------------------------------------------------------------
    def __len__(self):
        return len(self.index)

    def __getitem__(self, idx):
        fi, si = self.index[idx]
        fp = Path(self.file_meta[fi]["path"])
        fm = self.file_meta[fi]

        with xr.open_dataset(fp) as ds:
            da = ds[self.var_name]
            da_slice = da.isel({self.slice_dim: slice(si, si + self.slice_len)})
            # Explicit transpose to (slice_dim[C], H_dim, W_dim)
            da_slice = da_slice.transpose(self.slice_dim, *self.hw_dims)
            arr = da_slice.values.astype(np.float32)
            # arr.shape: (slice_len, n_H, n_W)

        arr = self._normalize(arr)
        valid = np.isfinite(arr)
        arr = np.nan_to_num(arr, nan=float(self.fill_nan))

        t = torch.from_numpy(np.ascontiguousarray(arr))  # (C, H, W)
        valid_t = torch.from_numpy(np.ascontiguousarray(valid.astype(np.bool_)))

        meta = {
            "file_idx":        fi,
            "slice_start":     si,
            "slice_end":       si + self.slice_len,
            "slice_dim":       self.slice_dim,
            "hw_dims":         self.hw_dims,
            "path":            str(fp),
            self.lat_name:     fm[self.lat_name],
            self.lon_name:     fm[self.lon_name],
            self.z_name:       fm[self.z_name],
            "slice_coord_vals": fm[self.slice_dim][si: si + self.slice_len],
        }

        return TrainingItem(input=t, tgt=t, valid_mask=valid_t), meta
    
class NetCDF2DDataModule(pl.LightningDataModule):
    def __init__(
        self,
        root,
        var_name,
        dl_kw,
        select_time=None,
        patch_size=None,
        normalize=True,
        fill_nan_for_nn=0.0,
        norm_stats=None,           # optional (m,s). If None, computed from TRAIN.
        eps=1e-6,
        rgb=False,
        slice_dim=None,
        slice_len=3,
        slice_mode="random",
        slice_start=None,
    ):
        super().__init__()
        self.root = root
        self.var_name = var_name
        self.dl_kw = dl_kw
        self.select_time = select_time
        self.patch_size = patch_size
        self.normalize = normalize
        self.fill_nan_for_nn = fill_nan_for_nn
        self._norm_stats = norm_stats
        self.eps = eps
        self.rgb = rgb
        self.slice_dim = slice_dim
        self.slice_len = slice_len
        self.slice_mode = slice_mode
        self.slice_start = slice_start

        self.train_ds = None
        self.val_ds = None
        self.test_ds = None

    def norm_stats(self):
        # matches your BaseDataModule API
        if self._norm_stats is None:
            if self.normalize in ("minmax", "minmax01"):
                self._norm_stats = self._compute_train_min_max()
            elif self.normalize in ("robust", "iqr", "median_iqr"):
                self._norm_stats = self._compute_train_median_iqr()
            else:
                self._norm_stats = self._compute_train_mean_std()
            print("Norm stats", self._norm_stats)
        return self._norm_stats

    def _compute_train_mean_std(self):
        # NaN-safe global mean/std over train split, using raw values (before filling)
        train_ds = NetCDFFolder2D(
            self.root, self.var_name, split="train",
            select_time=self.select_time,
            patch_size=None,              # IMPORTANT: stats on full fields
            normalize=False,
            nan_value=np.nan,
            fill_nan_for_nn=self.fill_nan_for_nn,
            norm_stats=(0.0, 1.0),        # unused when normalize=False
            rgb=False,
            return_meta=False,
        )

        total_sum = 0.0
        total_sumsq = 0.0
        total_count = 0

        for i in range(len(train_ds)):
            fp = train_ds.files[i]
            arr = train_ds._load_full(fp)  # may contain NaNs
            m = np.isfinite(arr)
            if m.sum() == 0:
                continue
            v = arr[m].astype(np.float64, copy=False)
            total_sum += float(v.sum())
            total_sumsq += float((v * v).sum())
            total_count += int(v.size)

        if total_count == 0:
            raise RuntimeError("No finite values in train set for mean/std.")

        mean = total_sum / total_count
        var = total_sumsq / total_count - mean * mean
        std = float(np.sqrt(max(var, 0.0)))
        std = max(std, self.eps)
        return (float(mean), float(std))

    def _compute_train_min_max(self):
        # NaN-safe global min/max over train split
        train_ds = NetCDFFolder2D(
            self.root, self.var_name, split="train",
            select_time=self.select_time,
            patch_size=None,
            normalize=False,
            nan_value=np.nan,
            fill_nan_for_nn=self.fill_nan_for_nn,
            norm_stats=(0.0, 1.0),
            rgb=False,
            return_meta=False,
        )

        vmin = None
        vmax = None

        for i in range(len(train_ds)):
            fp = train_ds.files[i]
            arr = train_ds._load_full(fp)
            m = np.isfinite(arr)
            if m.sum() == 0:
                continue
            v = arr[m].astype(np.float64, copy=False)
            mn = float(v.min())
            mx = float(v.max())
            vmin = mn if vmin is None else min(vmin, mn)
            vmax = mx if vmax is None else max(vmax, mx)

        if vmin is None or vmax is None:
            raise RuntimeError("No finite values in train set for min/max.")

        return (float(vmin), float(vmax))

    def _compute_train_median_iqr(self):
        # NaN-safe global median/IQR over train split
        train_ds = NetCDFFolder2D(
            self.root, self.var_name, split="train",
            select_time=self.select_time,
            patch_size=None,
            normalize=False,
            nan_value=np.nan,
            fill_nan_for_nn=self.fill_nan_for_nn,
            norm_stats=(0.0, 1.0),
            rgb=False,
            return_meta=False,
        )

        vals = []
        for i in range(len(train_ds)):
            fp = train_ds.files[i]
            arr = train_ds._load_full(fp)
            m = np.isfinite(arr)
            if m.sum() == 0:
                continue
            vals.append(arr[m].astype(np.float64, copy=False))

        if not vals:
            raise RuntimeError("No finite values in train set for median/IQR.")

        data = np.concatenate(vals)
        median = float(np.median(data))
        q25, q75 = np.percentile(data, [25.0, 75.0])
        iqr = max(float(q75) - float(q25), self.eps)
        return (median, iqr)

    def setup(self, stage=None):
        stats = self.norm_stats() if self.normalize else None

        self.train_ds = NetCDFFolder2D(
            self.root, self.var_name, split="train",
            select_time=self.select_time,
            patch_size=self.patch_size,
            normalize=self.normalize,
            fill_nan_for_nn=self.fill_nan_for_nn,
            norm_stats=stats,
            eps=self.eps,
            rgb=self.rgb,
            slice_dim=self.slice_dim,
            slice_len=self.slice_len,
            slice_mode=self.slice_mode,
            slice_start=self.slice_start,
            return_meta=False,
        )
        self.val_ds = NetCDFFolder2D(
            self.root, self.var_name, split="val",
            select_time=self.select_time,
            patch_size=None,
            normalize=self.normalize,
            fill_nan_for_nn=self.fill_nan_for_nn,
            norm_stats=stats,
            eps=self.eps,
            rgb=self.rgb,
            slice_dim=self.slice_dim,
            slice_len=self.slice_len,
            slice_mode=self.slice_mode,
            slice_start=self.slice_start,
            return_meta=False,
        )
        self.test_ds = NetCDFFolder2D(
            self.root, self.var_name, split="test",
            select_time=self.select_time,
            patch_size=None,
            normalize=self.normalize,
            fill_nan_for_nn=self.fill_nan_for_nn,
            norm_stats=stats,
            eps=self.eps,
            rgb=self.rgb,
            slice_dim=self.slice_dim,
            slice_len=self.slice_len,
            slice_mode=self.slice_mode,
            slice_start=self.slice_start,
            return_meta=False,
        )

    def train_dataloader(self):
        return torch.utils.data.DataLoader(self.train_ds, shuffle=True, **self.dl_kw)

    def val_dataloader(self):
        return torch.utils.data.DataLoader(self.val_ds, shuffle=False, **self.dl_kw)

    def test_dataloader(self):
        return torch.utils.data.DataLoader(self.test_ds, shuffle=False, **self.dl_kw)

class NetCDFVolumeDataModule(pl.LightningDataModule):
    """
    LightningDataModule wrapping NetCDFVolume2DSlices.

    Handles:
      - norm_stats computation from train split (if not provided)
      - train/val/test dataset construction
      - dataloader creation

    Layout follows NetCDFVolume2DSlices rules:
      slice_dim="lat" -> (C=slice_len, H=n_lon, W=n_z)
      slice_dim="lon" -> (C=slice_len, H=n_lat, W=n_z)
      slice_dim="z"   -> (C=slice_len, H=n_lat, W=n_lon)
    """

    def __init__(
        self,
        root,
        var_name,
        dl_kw: dict,                   # passed directly to DataLoader (batch_size, num_workers, etc.)
        slice_dim="lat",               # dimension to iterate + use as channel axis
        slice_len=1,                   # consecutive slices stacked as channels
        normalize="zscore",
        norm_stats=None,               # (mean, std) or (vmin, vmax). If None, computed from train.
        eps=1e-6,
        fill_nan=0.0,
        lat_name="lat",
        lon_name="lon",
        z_name="z",
        # train-specific
        train_slice_mode="random",     # "random" | "sequential"
        train_slice_stride=1,          # stride for sequential mode
    ):
        super().__init__()
        self.root = root
        self.var_name = var_name
        self.dl_kw = dl_kw
        self.slice_dim = slice_dim
        self.slice_len = slice_len
        self.normalize = normalize
        self._norm_stats = norm_stats
        self.eps = eps
        self.fill_nan = fill_nan
        self.lat_name = lat_name
        self.lon_name = lon_name
        self.z_name = z_name
        self.train_slice_mode = train_slice_mode
        self.train_slice_stride = train_slice_stride

        self.train_ds = None
        self.val_ds   = None
        self.test_ds  = None

    # ------------------------------------------------------------------
    # Norm stats (lazy, computed once from train split)
    # ------------------------------------------------------------------
    def norm_stats(self):
        if self._norm_stats is None:
            if self.normalize in ("minmax", "minmax01"):
                self._norm_stats = self._compute_train_min_max()
            else:
                self._norm_stats = self._compute_train_mean_std()
            print(f"Computed norm_stats ({self.normalize}): {self._norm_stats}")
        return self._norm_stats

    def _compute_train_mean_std(self):
        """NaN-safe global mean/std over the train split."""
        tmp_ds = NetCDFVolume2DSlices(
            root=self.root, var_name=self.var_name, split="train",
            slice_dim=self.slice_dim, slice_len=1,
            normalize=None, norm_stats=None,
            eps=self.eps, fill_nan=0.0,   # keep NaNs for stat computation
            lat_name=self.lat_name, lon_name=self.lon_name, z_name=self.z_name,
        )
        total_sum   = 0.0
        total_sumsq = 0.0
        total_count = 0

        for fi, fp in enumerate(tmp_ds.files):
            with xr.open_dataset(fp) as ds:
                arr = ds[self.var_name].values.astype(np.float64, copy=False)
            m = np.isfinite(arr)
            if m.sum() == 0:
                continue
            v = arr[m]
            total_sum   += float(v.sum())
            total_sumsq += float((v * v).sum())
            total_count += int(v.size)

        if total_count == 0:
            raise RuntimeError("No finite values in train split for mean/std.")

        mean = total_sum / total_count
        var  = total_sumsq / total_count - mean * mean
        std  = float(np.sqrt(max(var, 0.0)))
        return (float(mean), max(std, self.eps))

    def _compute_train_min_max(self):
        """NaN-safe global min/max over the train split."""
        tmp_ds = NetCDFVolume2DSlices(
            root=self.root, var_name=self.var_name, split="train",
            slice_dim=self.slice_dim, slice_len=1,
            normalize=None, norm_stats=None,
            eps=self.eps, fill_nan=0.0,
            lat_name=self.lat_name, lon_name=self.lon_name, z_name=self.z_name,
        )
        vmin, vmax = None, None

        for fp in tmp_ds.files:
            with xr.open_dataset(fp) as ds:
                arr = ds[self.var_name].values.astype(np.float64, copy=False)
            m = np.isfinite(arr)
            if m.sum() == 0:
                continue
            v = arr[m]
            vmin = float(v.min()) if vmin is None else min(vmin, float(v.min()))
            vmax = float(v.max()) if vmax is None else max(vmax, float(v.max()))

        if vmin is None:
            raise RuntimeError("No finite values in train split for min/max.")
        return (float(vmin), float(vmax))

    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------
    def setup(self, stage=None):
        stats = self.norm_stats() if self.normalize else None

        # Shared kwargs for all splits
        common = dict(
            root=self.root, var_name=self.var_name,
            slice_dim=self.slice_dim, slice_len=self.slice_len,
            normalize=self.normalize, norm_stats=stats,
            eps=self.eps, fill_nan=self.fill_nan,
            lat_name=self.lat_name, lon_name=self.lon_name, z_name=self.z_name,
        )

        self.train_ds = NetCDFVolume2DSlices(split="train", **common)
        self.val_ds   = NetCDFVolume2DSlices(split="val",   **common)
        self.test_ds  = NetCDFVolume2DSlices(split="test",  **common)

    # ------------------------------------------------------------------
    # Dataloaders
    # ------------------------------------------------------------------
    def train_dataloader(self):
        return torch.utils.data.DataLoader(
            self.train_ds,
            shuffle=True,
            collate_fn=volume_collate_fn,
            **self.dl_kw,
        )

    def val_dataloader(self):
        return torch.utils.data.DataLoader(
            self.val_ds,
            shuffle=False,
            collate_fn=volume_collate_fn,
            **self.dl_kw,
        )

    def test_dataloader(self):
        return torch.utils.data.DataLoader(
            self.test_ds,
            shuffle=False,
            collate_fn=volume_collate_fn,
            **self.dl_kw,
        )
        
# -----------------------------------------------------------------------
# Helpers for the DataLoader: numpy arrays can't be default-collated
# when they have variable length, so we use a simple collate
# -----------------------------------------------------------------------
def volume_collate_fn(batch):
    """Collate (TrainingItem, meta) pairs; keeps numpy coord arrays as lists."""
    items, metas = zip(*batch)
    inputs = torch.stack([it.input for it in items])
    tgts   = torch.stack([it.tgt   for it in items])
    if any(it.valid_mask is not None for it in items):
        masks = torch.stack([
            it.valid_mask if it.valid_mask is not None else torch.ones_like(it.tgt, dtype=torch.bool)
            for it in items
        ])
    else:
        masks = None

    collated_meta = {}
    for k in metas[0].keys():
        vals = [m[k] for m in metas]
        # scalars and strings -> list; tensors/arrays -> list (not stacked, sizes may differ)
        collated_meta[k] = vals

    return TrainingItem(input=inputs, tgt=tgts, valid_mask=masks), collated_meta


# -----------------------------------------------------------------------
# Inverse: (C, H, W) tensor -> 2D np array in (hw_dim0, hw_dim1) order
# -----------------------------------------------------------------------
def _tensor_to_slices(rec_chw: np.ndarray) -> np.ndarray:
    """
    rec_chw: (C, H, W) numpy array
    Returns (C, H, W) — no reordering needed because the tensor was already
    built as (slice_dim[C], hw0[H], hw1[W]).
    """
    return rec_chw   # shape (slice_len, n_hw0, n_hw1)


# -----------------------------------------------------------------------
# Reconstruction
# -----------------------------------------------------------------------

def denorm_bchw(
    x: torch.Tensor,
    norm_stats: Optional[Tuple[float, float]],
    norm_mode: Optional[str] = None,
    eps: float = 1e-6,
) -> torch.Tensor:
    """
    Denormalize a ``(B, C, H, W)`` tensor.

    ``norm_mode="zscore"|"standard"``  →  ``x_phys = x * (std + eps) + mean``
    ``norm_mode="minmax"|"minmax01"``  →  ``x_phys = x * (vmax - vmin + eps) + vmin``

    Returns *x* unchanged when *norm_stats* is ``None``.
    """
    if norm_stats is None:
        return x

    def _s(v: float) -> torch.Tensor:
        return torch.as_tensor(v, dtype=x.dtype, device=x.device).view(1, 1, 1, 1)

    if norm_mode in ("minmax", "minmax01"):
        vmin, vmax = norm_stats
        return x * (_s(vmax) - _s(vmin) + eps) + _s(vmin)

    # zscore / standard (default fallback)
    mean, std = norm_stats
    return x * (_s(std) + eps) + _s(mean)

def reconstruct_volumes(
    net,
    dataset: NetCDFVolume2DSlices,
    save_dir: str,
    norm_stats=None,
    norm_mode=None,
    batch_size: int = 1,
    device=None,
):
    """
    Run all slices through net and reassemble full volumes.
    Output NetCDF always has dims in (lat, lon, z) canonical order.

    When slice_len > 1, overlapping positions are averaged.
    """
    from collections import defaultdict

    if device is None:
        device = next(net.parameters()).device

    net.eval()
    os.makedirs(save_dir, exist_ok=True)

    lat_name  = dataset.lat_name
    lon_name  = dataset.lon_name
    z_name    = dataset.z_name
    slice_dim = dataset.slice_dim
    hw_dims   = dataset.hw_dims   # (H_dim_name, W_dim_name)

    # accumulator[fi] = (sum_array, count_array) in (slice_dim, hw0, hw1) order
    accumulators = {}

    loader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=False,
        collate_fn=volume_collate_fn,
    )

    with torch.no_grad():
        for batch_item, meta in loader:
            x = batch_item.tgt.to(device)   # (B, C, H, W)
            B, C, H, W = x.shape

            pad_h = (64 - H % 64) % 64
            pad_w = (64 - W % 64) % 64
            x_pad = F.pad(x, (0, pad_w, 0, pad_h), mode="reflect")

            out = net(x_pad)
            x_hat = out["x_hat"][:, :, :H, :W]   # (B, C, H, W)

            x_hat_phys = denorm_bchw(x_hat, norm_stats, norm_mode=norm_mode)

            for b in range(B):
                fi          = int(meta["file_idx"][b])
                slice_start = int(meta["slice_start"][b])
                slice_end   = int(meta["slice_end"][b])

                fm = dataset.file_meta[fi]

                # Initialise accumulator for this file on first encounter
                if fi not in accumulators:
                    n_slice = fm["n_slice"]
                    n_hw0   = len(fm[hw_dims[0]])
                    n_hw1   = len(fm[hw_dims[1]])
                    accumulators[fi] = {
                        "sum":   np.zeros((n_slice, n_hw0, n_hw1), dtype=np.float64),
                        "count": np.zeros((n_slice, n_hw0, n_hw1), dtype=np.float64),
                    }

                rec = x_hat_phys[b].cpu().numpy()   # (C, H, W) = (slice_len, n_hw0, n_hw1)
                rec = _tensor_to_slices(rec)

                accumulators[fi]["sum"][slice_start:slice_end]   += rec
                accumulators[fi]["count"][slice_start:slice_end] += 1.0

    # --- Assemble and save ---
    for fi, acc in accumulators.items():
        fm = dataset.file_meta[fi]

        with np.errstate(invalid="ignore"):
            vol = (acc["sum"] / np.where(acc["count"] > 0, acc["count"], np.nan)).astype(np.float32)
        # vol: (n_slice, n_hw0, n_hw1) in (slice_dim, hw_dims[0], hw_dims[1]) order

        da_rec = xr.DataArray(
            vol,
            dims=[slice_dim, hw_dims[0], hw_dims[1]],
            coords={
                slice_dim:  fm[slice_dim],
                hw_dims[0]: fm[hw_dims[0]],
                hw_dims[1]: fm[hw_dims[1]],
            },
            name=f"{dataset.var_name}_rec",
        ).transpose(lat_name, lon_name, z_name)   # canonical output order

        with xr.open_dataset(fm["path"]) as ds:
            da_gt = ds[dataset.var_name].transpose(lat_name, lon_name, z_name)

        ds_out = xr.Dataset({
            f"{dataset.var_name}_gt":  da_gt.astype(np.float32),
            f"{dataset.var_name}_rec": da_rec.astype(np.float32),
        })

        out_path = os.path.join(save_dir, Path(fm["path"]).stem + "_rec.nc")
        ds_out.to_netcdf(out_path)
        print(f"Saved {out_path} | shape={da_rec.shape} | dims={tuple(da_rec.dims)}")