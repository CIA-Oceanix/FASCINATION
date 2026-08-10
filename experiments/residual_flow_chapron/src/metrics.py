# import torch
# import numpy as np
# import PIL.Image as Image
# from typing import Dict, List, Optional, Tuple, Union
# from pytorch_msssim import ms_ssim


# def compute_metrics(
#     a: Union[np.array, Image.Image],
#     b: Union[np.array, Image.Image],
#     max_val: float = 255.0,
# ) -> Tuple[float, float]:
#     """Returns PSNR and MS-SSIM between images `a` and `b`. """
#     if isinstance(a, Image.Image):
#         a = np.asarray(a)
#     if isinstance(b, Image.Image):
#         b = np.asarray(b)

#     a = torch.from_numpy(a.copy()).float().unsqueeze(0)
#     if a.size(3) == 3:
#         a = a.permute(0, 3, 1, 2)
#     b = torch.from_numpy(b.copy()).float().unsqueeze(0)
#     if b.size(3) == 3:
#         b = b.permute(0, 3, 1, 2)

#     mse = torch.mean((a - b) ** 2).item()
#     p = 20 * np.log10(max_val) - 10 * np.log10(mse)
#     m = ms_ssim(a, b, data_range=max_val).item()
#     return p, m

from pathlib import Path
import numpy as np
import torch
from PIL import Image
from pytorch_msssim import ssim
from typing import Any, Dict, List, Optional, Tuple, Union
import torch.nn.functional as F
import xarray as xr
from PIL import Image
from scipy.ndimage import convolve
from scipy.interpolate import interp1d
from scipy.signal import butter, filtfilt
from pytorch_msssim import ms_ssim
from FASCINATION.experiments.residual_flow_chapron.src.utils import *

try:
    from dtaidistance import dtw
except Exception:
    dtw = None

try:
    import scipy.signal as signal
except Exception:
    signal = None

try:
    from scipy.ndimage import convolve as _nd_convolve
except Exception:
    _nd_convolve = None

_dtw_lib = dtw
_scipy_signal = signal

DEPTH_DIM_CANDIDATES = ("z", "depth", "deptht", "z_c", "z_f")

# =========================
#   Meta / xarray helpers
# =========================

from pathlib import Path
import json
import os
from typing import Any, Dict, List, Optional, Tuple, Union
 
import numpy as np
import torch
import torch.nn.functional as F
import xarray as xr
from PIL import Image
from pytorch_msssim import ms_ssim, ssim
 
try:
    from scipy.ndimage import convolve as _nd_convolve
except Exception:
    _nd_convolve = None
 
try:
    import scipy.signal as signal
    _scipy_signal = signal
except Exception:
    signal = None
    _scipy_signal = None
 
try:
    from dtaidistance import dtw as _dtw_lib
except Exception:
    _dtw_lib = None

try:
    from dtaidistance import dtw_ndim as _dtw_ndim_lib
except Exception:
    _dtw_ndim_lib = None
 
DEPTH_DIM_CANDIDATES = ("z", "depth", "deptht", "z_c", "z_f")
 
 
# ===========================================================================
# Meta / xarray helpers
# ===========================================================================
 
def _unbatch_meta(meta: Any) -> Optional[Dict[str, Any]]:
    if isinstance(meta, (list, tuple)) and len(meta) > 0:
        meta0 = meta[0]
        if isinstance(meta0, dict):
            return meta0
        return None
    if not isinstance(meta, dict):
        return None
    out: Dict[str, Any] = {}
    for k, v in meta.items():
        if isinstance(v, (list, tuple)) and len(v) > 0:
            out[k] = v[0]
        else:
            out[k] = v
    return out
 
 
def denorm_bchw(
    x: torch.Tensor,
    norm_stats: Optional[Tuple[float, float]],
    norm_mode: Optional[str] = None,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Denormalize (B,C,H,W) using global stats. No-op if norm_stats is None."""
    if norm_stats is None:
        return x
    if norm_mode in ("minmax", "minmax01"):
        vmin, vmax = norm_stats
        vmin_t = torch.as_tensor(vmin, dtype=x.dtype, device=x.device).view(1, 1, 1, 1)
        vmax_t = torch.as_tensor(vmax, dtype=x.dtype, device=x.device).view(1, 1, 1, 1)
        return x * (vmax_t - vmin_t + eps) + vmin_t
    if norm_mode in ("zscore_squash", "zscore_squash_q1", "standard_squash", "standard_squash_q1"):
        m, s = norm_stats
        m_t = torch.as_tensor(m, dtype=x.dtype, device=x.device).view(1, 1, 1, 1)
        s_t = torch.as_tensor(s, dtype=x.dtype, device=x.device).view(1, 1, 1, 1)
        x_safe = x.clamp(min=eps, max=1.0 - eps)
        z = torch.log(x_safe / (1.0 - x_safe))
        return z * (s_t + eps) + m_t
    m, s = norm_stats
    m_t = torch.as_tensor(m, dtype=x.dtype, device=x.device).view(1, 1, 1, 1)
    s_t = torch.as_tensor(s, dtype=x.dtype, device=x.device).view(1, 1, 1, 1)
    return x * (s_t + eps) + m_t
 
 
def mse_np(
    a: np.ndarray,
    b: np.ndarray,
    valid_mask: Optional[np.ndarray] = None,
) -> float:
    """NaN-safe MSE between two arrays, optionally restricted to valid_mask."""
    a = np.asarray(a)
    b = np.asarray(b)
    m = np.isfinite(a)
    if valid_mask is not None:
        vm = np.asarray(valid_mask).astype(bool, copy=False)
        if vm.shape != a.shape:
            raise ValueError(f"valid_mask shape mismatch: {vm.shape} vs {a.shape}")
        m = m & vm
    if m.sum() == 0:
        return float("nan")
    a_masked = np.where(m, a, 0.0)
    b_masked = np.where(m, np.nan_to_num(b, nan=0.0, posinf=0.0, neginf=0.0), 0.0)
    d = a_masked[m] - b_masked[m]
    return float(np.mean(d * d))


def mse_bulk_outliers_np(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    valid_mask: Optional[np.ndarray] = None,
    p_low: float = 1.0,
    p_high: float = 99.0,
) -> Dict[str, float]:
    """
    Split MSE into bulk and outliers based on GT percentiles.

    bulk_mask := p_low <= y_true <= p_high
    outliers  := complement of bulk_mask
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    if y_true.shape != y_pred.shape:
        raise ValueError(f"Shape mismatch: {y_true.shape} vs {y_pred.shape}")
    if not (0.0 <= float(p_low) < float(p_high) <= 100.0):
        raise ValueError(f"Expected 0 <= p_low < p_high <= 100, got ({p_low}, {p_high})")

    valid = np.isfinite(y_true)
    if valid_mask is not None:
        vm = np.asarray(valid_mask).astype(bool, copy=False)
        if vm.shape != y_true.shape:
            raise ValueError(f"valid_mask shape mismatch: {vm.shape} vs {y_true.shape}")
        valid = valid & vm

    n_valid = int(valid.sum())
    if n_valid == 0:
        return {
            "mse_bulk": float("nan"),
            "mse_outliers": float("nan"),
            "p_low": float("nan"),
            "p_high": float("nan"),
            "n_bulk": 0.0,
            "n_outliers": 0.0,
        }

    gt_valid = y_true[valid].astype(np.float64, copy=False)
    pred_valid = np.nan_to_num(y_pred[valid], nan=0.0, posinf=0.0, neginf=0.0).astype(np.float64, copy=False)

    q_low, q_high = np.percentile(gt_valid, [float(p_low), float(p_high)])
    bulk_mask = (gt_valid >= q_low) & (gt_valid <= q_high)
    out_mask = ~bulk_mask

    mse_bulk = float(np.mean((gt_valid[bulk_mask] - pred_valid[bulk_mask]) ** 2)) if np.any(bulk_mask) else float("nan")
    mse_out = float(np.mean((gt_valid[out_mask] - pred_valid[out_mask]) ** 2)) if np.any(out_mask) else float("nan")

    return {
        "mse_bulk": mse_bulk,
        "mse_outliers": mse_out,
        "p_low": float(q_low),
        "p_high": float(q_high),
        "n_bulk": float(int(bulk_mask.sum())),
        "n_outliers": float(int(out_mask.sum())),
    }
 
 
def apply_gt_nan_mask_torch(
    gt: torch.Tensor,
    pred: torch.Tensor,
    valid_mask: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Build mask from GT finite values and set invalid positions to 0.0.
    Returns: gt_masked, pred_masked, combined_valid_mask
    """
    combined_valid = torch.isfinite(gt)
    if valid_mask is not None:
        if valid_mask.shape != gt.shape:
            raise ValueError(f"valid_mask shape mismatch: {valid_mask.shape} vs {gt.shape}")
        combined_valid = combined_valid & valid_mask.to(dtype=torch.bool, device=gt.device)
    gt_masked = torch.where(combined_valid, gt, torch.zeros_like(gt))
    pred_finite = torch.nan_to_num(pred, nan=0.0, posinf=0.0, neginf=0.0)
    pred_masked = torch.where(combined_valid, pred_finite, torch.zeros_like(pred_finite))
    return gt_masked, pred_masked, combined_valid
 
 
def to_png_physical(x: torch.Tensor, eps: float = 1e-6) -> Image.Image:
    """Scale physical tensor to [0,1] per-image for visualization PNGs."""
    if x.dim() == 4:
        x = x[0]
    vmin = x.amin(dim=(-2, -1), keepdim=True)
    vmax = x.amax(dim=(-2, -1), keepdim=True)
    x_vis = (x - vmin) / (vmax - vmin + eps)
    return torch2img(x_vis)
 
 
def compression_rate_float64_from_bpp(bpp: float) -> float:
    return float("inf") if bpp <= 0 else 64.0 / float(bpp)
 
 
def make_2d_dataarray(arr2d: np.ndarray, meta: Any, name: str) -> xr.DataArray:
    if arr2d.ndim != 2:
        raise ValueError(f"{name}: expected 2D array, got {arr2d.shape}")
    n0, n1 = arr2d.shape
    meta = _unbatch_meta(meta)
    dims: Tuple[str, str] = ("y", "x")
    coords: Dict[str, Any] = {dims[0]: np.arange(n0), dims[1]: np.arange(n1)}
    if isinstance(meta, dict):
        mdims = meta.get("dims", None)
        mcoords = meta.get("coords", None)
        if isinstance(mdims, (list, tuple)) and len(mdims) == 2 and isinstance(mcoords, dict):
            d0, d1 = mdims
            c0 = np.asarray(mcoords.get(d0, []))
            c1 = np.asarray(mcoords.get(d1, []))
            if c0.ndim == 1 and c1.ndim == 1 and len(c0) == n0 and len(c1) == n1:
                dims = (d0, d1)
                coords = {d0: c0, d1: c1}
                for k, v in mcoords.items():
                    if k not in dims and np.ndim(v) == 0:
                        coords[k] = v
    return xr.DataArray(arr2d, dims=dims, coords=coords, name=name)
 
 
def make_3d_dataarray(arr3d: np.ndarray, meta: Any, name: str) -> xr.DataArray:
    if arr3d.ndim != 3:
        raise ValueError(f"{name}: expected 3D array, got {arr3d.shape}")
    n0, n1, n2 = arr3d.shape
    meta = _unbatch_meta(meta)
    dims: Tuple[str, str, str] = ("c", "y", "x")
    coords: Dict[str, Any] = {
        dims[0]: np.arange(n0), dims[1]: np.arange(n1), dims[2]: np.arange(n2),
    }
    if isinstance(meta, dict):
        mdims = meta.get("dims", None)
        mcoords = meta.get("coords", None)
        slice_dim = meta.get("slice_dim", None)
        spatial_dims = meta.get("spatial_dims", None)
        if isinstance(mdims, (list, tuple)) and len(mdims) == 3 and isinstance(mcoords, dict):
            d0, d1, d2 = mdims
            c0 = np.asarray(mcoords.get(d0, []))
            c1 = np.asarray(mcoords.get(d1, []))
            c2 = np.asarray(mcoords.get(d2, []))
            if (
                c0.ndim == 1 and c1.ndim == 1 and c2.ndim == 1
                and len(c0) == n0 and len(c1) == n1 and len(c2) == n2
            ):
                if (
                    isinstance(slice_dim, str)
                    and isinstance(spatial_dims, (list, tuple))
                    and len(spatial_dims) == 2
                    and d0 == slice_dim
                    and d1 == spatial_dims[0]
                    and d2 == spatial_dims[1]
                ):
                    arr3d = np.transpose(arr3d, (1, 2, 0))
                    dims = (d1, d2, d0)
                    coords = {d1: c1, d2: c2, d0: c0}
                    for k, v in mcoords.items():
                        if k not in dims and np.ndim(v) == 0:
                            coords[k] = v
                    return xr.DataArray(arr3d, dims=dims, coords=coords, name=name)
                dims = (d0, d1, d2)
                coords = {d0: c0, d1: c1, d2: c2}
                for k, v in mcoords.items():
                    if k not in dims and np.ndim(v) == 0:
                        coords[k] = v
    return xr.DataArray(arr3d, dims=dims, coords=coords, name=name)
 
 
def infer_sample_3d_dims_from_meta(meta: Any) -> Optional[Tuple[str, str, str]]:
    meta = _unbatch_meta(meta)
    if not isinstance(meta, dict):
        return None
    mdims = meta.get("dims", None)
    if isinstance(mdims, (list, tuple)) and len(mdims) == 3 and all(isinstance(d, str) for d in mdims):
        return tuple(mdims)  # type: ignore[return-value]
    return None
 
 
def pick_depth_axis(dims3: Optional[Tuple[str, str, str]]) -> int:
    if dims3 is None:
        return 0
    for axis, dim_name in enumerate(dims3):
        if dim_name.lower() in DEPTH_DIM_CANDIDATES:
            return axis
    return 0
 
 
# ===========================================================================
# Internal tensor helpers
# ===========================================================================
 
def _ensure_chw(x: torch.Tensor, name: str = "x") -> torch.Tensor:
    """Accept (C,H,W) or (1,C,H,W); always return (C,H,W)."""
    if x.ndim == 4:
        if x.shape[0] != 1:
            raise ValueError(f"{name}: batch size must be 1, got {x.shape[0]}")
        x = x.squeeze(0)
    if x.ndim != 3:
        raise ValueError(f"{name}: expected (C,H,W) or (1,C,H,W), got shape {x.shape}")
    return x
 
 
def _nan_mask(x: torch.Tensor) -> torch.Tensor:
    """Return True where x is finite (not NaN/Inf)."""
    return torch.isfinite(x)
 
 
# ===========================================================================
# Extrema detection helpers
# ===========================================================================
 
def _auto_grad_eps(x: torch.Tensor) -> float:
    grad_all = torch.diff(x, dim=1)
    abs_grad = grad_all[torch.isfinite(grad_all)].abs()
    if abs_grad.numel() == 0:
        return 0.0
    p5 = float(torch.quantile(abs_grad, 0.1).item())
    return p5 * 0.5
 
 
def _smooth_nan_1d_torch_recap(x: torch.Tensor, window: int) -> torch.Tensor:
    if window <= 1:
        return x
    kernel = torch.ones(window, dtype=x.dtype, device=x.device)
    valid = torch.isfinite(x).to(x.dtype)
    x0 = torch.where(torch.isfinite(x), x, torch.zeros_like(x))
    num = F.conv1d(x0.view(1, 1, -1), kernel.view(1, 1, -1), padding=window // 2).view(-1)
    den = F.conv1d(valid.view(1, 1, -1), kernel.view(1, 1, -1), padding=window // 2).view(-1)
    out = torch.full_like(x, torch.nan)
    ok = den > 0
    out[ok] = num[ok] / den[ok]
    return out
 
 
def _fill_zero_signs_1d_torch_recap(s: torch.Tensor) -> torch.Tensor:
    out = s.clone()
    for k in range(1, out.numel()):
        if out[k] == 0:
            out[k] = out[k - 1]
    for k in range(out.numel() - 2, -1, -1):
        if out[k] == 0:
            out[k] = out[k + 1]
    return out
 
 
def _suppress_close_extrema_1d_torch_recap(
    mask: torch.Tensor, min_separation: int
) -> torch.Tensor:
    if min_separation <= 1:
        return mask
    idx = torch.nonzero(mask, as_tuple=False).view(-1)
    if idx.numel() <= 1:
        return mask
    kept = torch.zeros_like(mask, dtype=torch.bool)
    last = -(10**9)
    for j in idx.tolist():
        if j - last >= min_separation:
            kept[j] = True
            last = j
    return kept

def _filter_candidates_by_side_support_1d_torch_recap(
    candidate_mask: torch.Tensor,
    grad_1d: torch.Tensor,
    eps: float,
    mode: str,
    side_window: int,
    min_support_count: int,
) -> torch.Tensor:
    """
    candidate_mask: (H-2,) bool, where candidate_mask[k] corresponds to an extremum at x-index h=k+1
    grad_1d:        (H-1,) gradient along H
    mode:           "min" or "max"
    """
    if mode not in ("min", "max"):
        raise ValueError(f"mode must be 'min' or 'max', got {mode}")

    if side_window <= 0 or min_support_count <= 0:
        return candidate_mask

    kept = torch.zeros_like(candidate_mask, dtype=torch.bool)
    g_len = grad_1d.numel()

    for k in torch.nonzero(candidate_mask, as_tuple=False).view(-1).tolist():
        h = k + 1  # extremum location in x, since turning lives on [1:-1]

        left = grad_1d[max(0, h - side_window):h]
        right = grad_1d[h:min(g_len, h + side_window)]

        if left.numel() == 0 or right.numel() == 0:
            continue

        left_finite = torch.isfinite(left)
        right_finite = torch.isfinite(right)

        if mode == "min":
            left_count = int(((left < -eps) & left_finite).sum().item())
            right_count = int(((right > eps) & right_finite).sum().item())
        else:  # mode == "max"
            left_count = int(((left > eps) & left_finite).sum().item())
            right_count = int(((right < -eps) & right_finite).sum().item())

        if left_count >= min_support_count and right_count >= min_support_count:
            kept[k] = True

    return kept

def detect_extrema_v2(
    x: torch.Tensor,
    kind: str = "both",
    grad_eps: Union[float, str] = "auto",
    smooth_window: int = 1,
    min_separation: int = 3,
    fill_zero_signs: bool = True,
    grad_eps_reference: Optional[torch.Tensor] = None,
    side_window: int = 5,
    min_support_count: int = 1,
) -> torch.Tensor:
    """
    Detect local extrema along H axis of a (C,H,W) tensor, with stricter
    left/right neighborhood support.

    Compared to detect_extrema(), this v2 keeps only extrema that have:
      - for a minimum: at least `min_support_count` gradients < -eps on the left
                       and at least `min_support_count` gradients > +eps on the right
      - for a maximum: at least `min_support_count` gradients > +eps on the left
                       and at least `min_support_count` gradients < -eps on the right

    Parameters
    ----------
    x : torch.Tensor
        Input tensor of shape (C,H,W) or (1,C,H,W).
    kind : {"min", "max", "both"}
        Which extrema to return.
    grad_eps : float or "auto"
        Gradient threshold. Small gradients in [-eps, +eps] are treated as flat.
    smooth_window : int
        Optional moving-average smoothing window along H before detection.
    min_separation : int
        Minimum separation between detected extrema along H.
    fill_zero_signs : bool
        Whether to propagate nonzero signs across flat/weak-gradient zones.
    grad_eps_reference : Optional[torch.Tensor]
        If provided and grad_eps=="auto", estimate eps from this tensor instead of x.
    side_window : int
        Number of gradient samples to inspect on each side of a candidate extremum.
    min_support_count : int
        Minimum number of sufficiently strong gradients required on each side.

    Returns
    -------
    torch.Tensor
        Float mask of shape (C,H,W), with 1.0 at extrema locations and 0.0 elsewhere.
    """
    if kind not in ("min", "max", "both"):
        raise ValueError(f"kind must be 'min', 'max', or 'both', got '{kind}'")

    x = _ensure_chw(x)
    C, H, W = x.shape

    if H < 3:
        return torch.zeros_like(x, dtype=torch.float32)

    smooth_window = max(1, int(smooth_window))
    if smooth_window % 2 == 0:
        smooth_window += 1

    side_window = max(1, int(side_window))
    min_support_count = max(1, int(min_support_count))

    x_work = x.to(torch.float64).clone()

    if smooth_window > 1:
        for c in range(C):
            for w in range(W):
                x_work[c, :, w] = _smooth_nan_1d_torch_recap(x_work[c, :, w], smooth_window)

    if grad_eps == "auto":
        ref = grad_eps_reference if grad_eps_reference is not None else x_work
        eps = _auto_grad_eps(ref)
    else:
        eps = float(grad_eps)

    grad = torch.diff(x_work, dim=1)  # (C, H-1, W)

    if eps > 0:
        sign = torch.where(
            grad > eps, torch.ones_like(grad),
            torch.where(grad < -eps, -torch.ones_like(grad), torch.zeros_like(grad)),
        )
    else:
        sign = torch.sign(grad)

    if fill_zero_signs:
        for c in range(C):
            for w in range(W):
                sign[c, :, w] = _fill_zero_signs_1d_torch_recap(sign[c, :, w])

    turning = torch.diff(sign, dim=1)  # (C, H-2, W)

    core_min = turning > 0
    core_max = turning < 0

    # Additional neighborhood support check
    for c in range(C):
        for w in range(W):
            grad_1d = grad[c, :, w]

            if kind in ("min", "both"):
                core_min[c, :, w] = _filter_candidates_by_side_support_1d_torch_recap(
                    candidate_mask=core_min[c, :, w],
                    grad_1d=grad_1d,
                    eps=eps,
                    mode="min",
                    side_window=side_window,
                    min_support_count=min_support_count,
                )

            if kind in ("max", "both"):
                core_max[c, :, w] = _filter_candidates_by_side_support_1d_torch_recap(
                    candidate_mask=core_max[c, :, w],
                    grad_1d=grad_1d,
                    eps=eps,
                    mode="max",
                    side_window=side_window,
                    min_support_count=min_support_count,
                )

    if kind == "min":
        core = core_min
    elif kind == "max":
        core = core_max
    else:
        core = core_min | core_max

    if min_separation > 1:
        for c in range(C):
            for w in range(W):
                core[c, :, w] = _suppress_close_extrema_1d_torch_recap(
                    core[c, :, w], min_separation
                )

    out = torch.zeros(C, H, W, dtype=torch.float32, device=x.device)
    out[:, 1:-1, :] = core.to(torch.float32)
    out = out * torch.isfinite(x).to(torch.float32)
    return out
 
def detect_extrema(
    x: torch.Tensor,
    kind: str = "both",
    grad_eps: Union[float, str] = "auto",
    smooth_window: int = 1,
    min_separation: int = 3,
    fill_zero_signs: bool = True,
    grad_eps_reference: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Detect local extrema along H axis of a (C,H,W) tensor.

    grad_eps_reference: if provided and grad_eps=="auto", compute the
        threshold from this tensor (typically GT) instead of x.
    """
    if kind not in ("min", "max", "both"):
        raise ValueError(f"kind must be 'min', 'max', or 'both', got '{kind}'")
    x = _ensure_chw(x)
    C, H, W = x.shape
    if H < 3:
        return torch.zeros_like(x, dtype=torch.float32)
    smooth_window = max(1, int(smooth_window))
    if smooth_window % 2 == 0:
        smooth_window += 1
    x_work = x.to(torch.float64).clone()
    if smooth_window > 1:
        for c in range(C):
            for w in range(W):
                x_work[c, :, w] = _smooth_nan_1d_torch_recap(x_work[c, :, w], smooth_window)
    if grad_eps == "auto":
        ref = grad_eps_reference if grad_eps_reference is not None else x_work
        eps = _auto_grad_eps(ref)
    else:
        eps = float(grad_eps)
    grad = torch.diff(x_work, dim=1)
    if eps > 0:
        sign = torch.where(
            grad > eps, torch.ones_like(grad),
            torch.where(grad < -eps, -torch.ones_like(grad), torch.zeros_like(grad)),
        )
    else:
        sign = torch.sign(grad)
    if fill_zero_signs:
        for c in range(C):
            for w in range(W):
                sign[c, :, w] = _fill_zero_signs_1d_torch_recap(sign[c, :, w])
    turning = torch.diff(sign, dim=1)
    if kind == "min":
        core = turning > 0
    elif kind == "max":
        core = turning < 0
    else:
        core = turning != 0
    if min_separation > 1:
        for c in range(C):
            for w in range(W):
                core[c, :, w] = _suppress_close_extrema_1d_torch_recap(core[c, :, w], min_separation)
    out = torch.zeros(C, H, W, dtype=torch.float32, device=x.device)
    out[:, 1:-1, :] = core.to(torch.float32)
    out = out * torch.isfinite(x).to(torch.float32)
    return out
 
 
def lowpass_filter(
    x: torch.Tensor,
    axis: int = 1,
    order: int = 2,
    wn: float = 0.107,
) -> torch.Tensor:
    """Zero-phase Butterworth low-pass filter along `axis`. Falls back to identity."""
    if _scipy_signal is None:
        return x
    x = _ensure_chw(x)
    arr = x.detach().cpu().numpy().astype(np.float64)
    b, a = _scipy_signal.butter(N=order, Wn=wn, btype="low", analog=False)
    filtered = _scipy_signal.filtfilt(b, a, arr, axis=axis)
    return torch.from_numpy(np.ascontiguousarray(filtered)).to(dtype=x.dtype, device=x.device)
 
 
def lowpass_filter_torch_along_axis(
    arr: torch.Tensor,
    axis: int = -1,
    order: int = 2,
    wn: float = 0.107,
) -> torch.Tensor:
    """Apply zero-phase Butterworth low-pass filtering along `axis`."""
    if signal is None:
        return arr
    arr_np = arr.detach().cpu().numpy().astype(np.float64, copy=False)
    b, a = signal.butter(N=order, Wn=wn, btype="low", analog=False)
    filtered_np = signal.filtfilt(b, a, arr_np, axis=axis)
    filtered_np = np.ascontiguousarray(filtered_np)
    return torch.from_numpy(filtered_np).to(device=arr.device, dtype=arr.dtype)
 
 
# ===========================================================================
# Extrema index detection (gradient sign inversion)
# ===========================================================================
 
def get_min_max_idx(
    arr: Union[np.ndarray, torch.Tensor],
    axs: int = 1,
    pad: bool = True,
) -> Union[np.ndarray, torch.Tensor]:
    """Gradient-sign inversion extrema detection (NumPy or Torch)."""
    if isinstance(arr, np.ndarray):
        if axs < 0:
            axs = arr.ndim + axs
        grad = np.diff(arr, axis=axs)
        grad_sign = np.sign(grad)
        min_max = np.abs(np.sign(np.diff(grad_sign, axis=axs)))
        if pad:
            pad_width = [(0, 0)] * arr.ndim
            pad_width[axs] = (1, 1)
            min_max = np.pad(min_max, pad_width, mode="constant", constant_values=1)
        return min_max
 
    if not isinstance(arr, torch.Tensor):
        raise TypeError(f"arr must be np.ndarray or torch.Tensor, got {type(arr)}")
    if axs < 0:
        axs = arr.ndim + axs
    grad = torch.diff(arr, dim=axs)
    grad_sign = torch.sign(grad)
    min_max_core = torch.abs(torch.sign(torch.diff(grad_sign, dim=axs))).to(torch.float32)
    if not pad:
        return min_max_core
    out = torch.ones_like(arr, dtype=torch.float32)
    core_slices = [slice(None)] * arr.ndim
    core_slices[axs] = slice(1, -1)
    out[tuple(core_slices)] = min_max_core
    return out
 
 
def get_min_or_max_idx(
    arr: torch.Tensor,
    axs: int = 1,
    kind: str = "min",
    pad: bool = True,
) -> torch.Tensor:
    """Gradient-sign inversion extrema detector split by type ('min' or 'max')."""
    if kind not in ("min", "max"):
        raise ValueError(f"Unsupported kind={kind}; expected 'min' or 'max'.")
    if axs < 0:
        axs = arr.ndim + axs
    if arr.shape[axs] < 3:
        if pad:
            return torch.zeros_like(arr, dtype=torch.float32)
        shape = list(arr.shape)
        shape[axs] = max(0, arr.shape[axs] - 2)
        return torch.zeros(shape, device=arr.device, dtype=torch.float32)
    grad = torch.diff(arr, dim=axs)
    grad_sign = torch.sign(grad)
    turning = torch.diff(grad_sign, dim=axs)
    core = (turning > 0).to(torch.float32) if kind == "min" else (turning < 0).to(torch.float32)
    if not pad:
        return core
    out = torch.zeros_like(arr, dtype=torch.float32)
    core_slices = [slice(None)] * arr.ndim
    core_slices[axs] = slice(1, -1)
    out[tuple(core_slices)] = core
    return out
 
 
# ===========================================================================
# One-hot helpers
# ===========================================================================
 
def _onehot_from_argidx(
    arg_idx: torch.Tensor, W: int, device=None, dtype=torch.float32
) -> torch.Tensor:
    if device is None:
        device = arg_idx.device
    out = torch.zeros((*arg_idx.shape, W), device=device, dtype=dtype)
    out.scatter_(-1, arg_idx.unsqueeze(-1), 1.0)
    return out
 
 
def minmax_onehot_along_w(x: torch.Tensor, kind: str = "min") -> torch.Tensor:
    """x: (C,H,W) or (B,C,H,W). Returns one-hot mask along W."""
    if x.ndim == 3:
        C, H, W = x.shape
        fn = torch.argmin if kind == "min" else torch.argmax
        idx = fn(x, dim=-1)
        return _onehot_from_argidx(idx, W, device=x.device)
    if x.ndim == 4:
        B, C, H, W = x.shape
        fn = torch.argmin if kind == "min" else torch.argmax
        idx = fn(x, dim=-1)
        return _onehot_from_argidx(idx, W, device=x.device)
    raise ValueError(f"Unsupported ndim={x.ndim}")
 
 
def minmax_onehot_along_w_masked(
    x: torch.Tensor,
    valid_mask: torch.Tensor,
    kind: str = "min",
) -> torch.Tensor:
    """NaN/invalid-aware min/max one-hot along W."""
    if x.shape != valid_mask.shape:
        raise ValueError(f"Shape mismatch: {x.shape} vs {valid_mask.shape}")
    fill_value = torch.inf if kind == "min" else -torch.inf
    x_masked = torch.where(valid_mask, x, torch.full_like(x, fill_value))
    fn = torch.argmin if kind == "min" else torch.argmax
    idx = fn(x_masked, dim=-1)
    onehot = _onehot_from_argidx(idx, x.shape[-1], device=x.device)
    any_valid = valid_mask.any(dim=-1, keepdim=True)
    onehot = onehot * any_valid.to(onehot.dtype)
    return onehot
 
 
# ===========================================================================
# Core F1 engine
# ===========================================================================
 
def _get_f1_score_torch_detailed(
    min_max_idx_truth: torch.Tensor,
    min_max_idx_ae: torch.Tensor,
    axs: int = -2,
    kernel_size: int = 10,
) -> Dict[str, Any]:
    """
    Tolerant F1 score with per-level breakdown, confusion matrices,
    GT/REC extrema counts per level, and a GT-extrema-weighted F1 mean.
 
    Returns
    -------
    f1_per_w            : list[float]  — F1 per W column
    f1_per_level / f1_per_h : list[float]  — F1 per H level
    confusion_per_w     : list[dict(tp,fp,fn,tn)]
    confusion_per_level / confusion_per_h : list[dict]
    f1_mean             : float  — unweighted mean of per-W F1
    f1_mean_h           : float  — unweighted mean of per-H F1
    gt_extrema_per_h    : list[int]  — GT extrema count per H level
    rec_extrema_per_h   : list[int]  — REC extrema count per H level
    f1_mean_weighted_h  : float  — per-H F1 weighted by GT extrema count
                                    (levels with 0 GT extrema are excluded)
    """
    if min_max_idx_truth.shape != min_max_idx_ae.shape:
        raise ValueError(f"Shape mismatch: {min_max_idx_truth.shape} vs {min_max_idx_ae.shape}")
    if min_max_idx_truth.ndim < 2:
        raise ValueError(f"Expected at least 2 dimensions, got {min_max_idx_truth.ndim}")
    if kernel_size < 1:
        raise ValueError(f"kernel_size must be >= 1, got {kernel_size}")
 
    ndim = min_max_idx_truth.ndim
    if axs < 0:
        axs = ndim + axs
 
    truth = (min_max_idx_truth > 0)
    pred  = (min_max_idx_ae   > 0)
 
    # Rearrange so H is last: (C, W, H)
    truth_h_last = truth.movedim(axs, -1)
    pred_h_last  = pred.movedim(axs, -1)
    h_size = truth_h_last.shape[-1]
 
    flat_truth = truth_h_last.reshape(-1, 1, h_size).to(torch.float32)
    flat_pred  = pred_h_last.reshape(-1, 1, h_size).to(torch.float32)
 
    kernel = torch.ones(1, 1, kernel_size, device=min_max_idx_truth.device, dtype=torch.float32)
    pad    = kernel_size // 2
 
    truth_expanded = F.conv1d(flat_truth, kernel, padding=pad)
    pred_expanded  = F.conv1d(flat_pred,  kernel, padding=pad)
 
    if kernel_size % 2 == 0:
        truth_expanded = truth_expanded[:, :, :h_size]
        pred_expanded  = pred_expanded[:, :, :h_size]
 
    truth_win = (truth_expanded > 0).reshape(*truth_h_last.shape)
    pred_win  = (pred_expanded  > 0).reshape(*pred_h_last.shape)
 
    tp_map = ( truth_win  &  pred_h_last)
    fp_map = (~truth_win  &  pred_h_last)
    fn_map = ( truth_h_last & ~pred_win )
 
    def _f1_from_maps(tp_m, fp_m, fn_m, dim):
        tp_ = tp_m.sum(dim=dim).to(torch.float64).sum(dim=0)
        fp_ = fp_m.sum(dim=dim).to(torch.float64).sum(dim=0)
        fn_ = fn_m.sum(dim=dim).to(torch.float64).sum(dim=0)
        prec_den = tp_ + fp_
        rec_den  = tp_ + fn_
        prec = torch.where(prec_den == 0, torch.zeros_like(tp_), tp_ / prec_den)
        rec  = torch.where(rec_den  == 0, torch.zeros_like(tp_), tp_ / rec_den)
        f1_den = prec + rec
        f1 = torch.where(f1_den == 0, torch.zeros_like(f1_den), 2.0 * prec * rec / f1_den)
        return f1, tp_, fp_, fn_
 
    # F1 per W: reduce over H (dim=-1) → (W,)
    f1_per_w, tp_w, fp_w, fn_w = _f1_from_maps(tp_map, fp_map, fn_map, dim=-1)
    # F1 per H: reduce over W (dim=-2) → (H,)
    f1_per_h, tp_h, fp_h, fn_h = _f1_from_maps(tp_map, fp_map, fn_map, dim=-2)
 
    def _confusion_list(tp_, fp_, fn_, total_per_level):
        tn_ = torch.clamp(
            torch.full_like(tp_, float(total_per_level)) - tp_ - fp_ - fn_, min=0.0
        )
        return [
            {"tp": int(t), "fp": int(p), "fn": int(n), "tn": int(tn)}
            for t, p, n, tn in zip(
                tp_.flatten().cpu().tolist(), fp_.flatten().cpu().tolist(),
                fn_.flatten().cpu().tolist(), tn_.flatten().cpu().tolist(),
            )
        ]
 
    total_per_h = int(tp_map.shape[0] * tp_map.shape[1])  # C * W
    total_per_w = int(tp_map.shape[0] * tp_map.shape[2])  # C * H
 
    # ------------------------------------------------------------------
    # GT and REC extrema counts per H level (sum over C and W)
    # truth_h_last / pred_h_last shape: (C, W, H)
    # ------------------------------------------------------------------
    gt_extrema_per_h  = truth_h_last.sum(dim=(0, 1)).long()   # (H,)
    rec_extrema_per_h = pred_h_last.sum(dim=(0, 1)).long()    # (H,)
 
    gt_counts  = [int(v) for v in gt_extrema_per_h.cpu().tolist()]
    rec_counts = [int(v) for v in rec_extrema_per_h.cpu().tolist()]
 
    # ------------------------------------------------------------------
    # F1 weighted by GT extrema count per H level
    # Levels with 0 GT extrema contribute weight=0 (excluded from mean).
    # This avoids empty levels dragging the unweighted mean down to 0.
    # ------------------------------------------------------------------
    weights      = gt_extrema_per_h.to(torch.float64)
    total_weight = float(weights.sum().item())
    if total_weight > 0:
        f1_mean_weighted_h = float(
            (f1_per_h.to(torch.float64) * weights).sum().item() / total_weight
        )
    else:
        f1_mean_weighted_h = float("nan")
 
    return {
        # --- existing keys (unchanged for backward compatibility) ---
        "f1_per_w":            [float(v) for v in f1_per_w.flatten().cpu().tolist()],
        "confusion_per_w":     _confusion_list(tp_w, fp_w, fn_w, total_per_w),
        "f1_per_level":        [float(v) for v in f1_per_h.flatten().cpu().tolist()],
        "f1_per_h":            [float(v) for v in f1_per_h.flatten().cpu().tolist()],
        "confusion_per_level": _confusion_list(tp_h, fp_h, fn_h, total_per_h),
        "confusion_per_h":     _confusion_list(tp_h, fp_h, fn_h, total_per_h),
        "f1_mean":             float(f1_per_w.mean().item()) if f1_per_w.numel() > 0 else float("nan"),
        "f1_mean_h":           float(f1_per_h.mean().item()) if f1_per_h.numel() > 0 else float("nan"),
        # --- new keys ---
        "gt_extrema_per_h":   gt_counts,
        "rec_extrema_per_h":  rec_counts,
        "f1_mean_weighted_h": f1_mean_weighted_h,
    }
 
 
def f1_score(
    truth_onehot: torch.Tensor,
    pred_onehot: torch.Tensor,
    kernel_size: int = 5,
) -> float:
    """Tolerant F1 along H, averaged over H levels. Scalar output."""
    if truth_onehot.shape != pred_onehot.shape:
        raise ValueError(f"Shape mismatch: {truth_onehot.shape} vs {pred_onehot.shape}")
    if truth_onehot.ndim == 2:
        truth_onehot = truth_onehot.unsqueeze(0)
        pred_onehot  = pred_onehot.unsqueeze(0)
    if truth_onehot.ndim == 4:
        if truth_onehot.shape[0] != 1:
            raise ValueError("Batch size > 1 not supported; pass one sample at a time.")
        truth_onehot = truth_onehot.squeeze(0)
        pred_onehot  = pred_onehot.squeeze(0)
    if truth_onehot.ndim != 3:
        raise ValueError(f"Expected (C,H,W) tensor, got shape {truth_onehot.shape}")
    out = _get_f1_score_torch_detailed(
        min_max_idx_truth=truth_onehot,
        min_max_idx_ae=pred_onehot,
        axs=-2,
        kernel_size=kernel_size,
    )
    return float(out["f1_mean"])
 
 
def f1_and_confusion_per_level(
    truth_onehot: torch.Tensor,
    pred_onehot: torch.Tensor,
    kernel_size: int = 5,
) -> Dict[str, Any]:
    """
    Tolerant F1 + confusion matrix + extrema counts per H level.
    Full dict output from _get_f1_score_torch_detailed.
    """
    if truth_onehot.shape != pred_onehot.shape:
        raise ValueError(f"Shape mismatch: {truth_onehot.shape} vs {pred_onehot.shape}")
    if truth_onehot.ndim == 2:
        truth_onehot = truth_onehot.unsqueeze(0)
        pred_onehot  = pred_onehot.unsqueeze(0)
    if truth_onehot.ndim == 4:
        if truth_onehot.shape[0] != 1:
            raise ValueError("Batch size > 1 not supported; pass one sample at a time.")
        truth_onehot = truth_onehot.squeeze(0)
        pred_onehot  = pred_onehot.squeeze(0)
    if truth_onehot.ndim != 3:
        raise ValueError(f"Expected (C,H,W) tensor, got shape {truth_onehot.shape}")
    return _get_f1_score_torch_detailed(
        min_max_idx_truth=truth_onehot,
        min_max_idx_ae=pred_onehot,
        axs=-2,
        kernel_size=kernel_size,
    )
 
 
def f1_and_confusion_per_level_split(
    truth_min_onehot: torch.Tensor,
    truth_max_onehot: torch.Tensor,
    pred_onehot: torch.Tensor,
    kernel_size: int = 5,
) -> Dict[str, Any]:
    """Tolerant F1 split by extremum type (min / max), then combined.

    Runs the same tolerant F1 engine independently against the GT min-only
    and GT max-only masks, then reports both individual scores and their
    unweighted average as the combined score.

    Parameters
    ----------
    truth_min_onehot : (C, H, W) binary mask — GT local minima positions.
    truth_max_onehot : (C, H, W) binary mask — GT local maxima positions.
    pred_onehot      : (C, H, W) binary mask — predicted extrema positions.
    kernel_size : int
        Tolerance window: a prediction within ``kernel_size // 2`` levels of
        a GT extremum counts as a TP.

    Returns
    -------
    dict with keys:
      - ``min``  : full result dict for min extrema (same keys as
                   :func:`f1_and_confusion_per_level`)
      - ``max``  : full result dict for max extrema
      - ``f1_min_mean``, ``f1_max_mean``, ``f1_combined_mean``
      - ``f1_min_mean_h``, ``f1_max_mean_h``, ``f1_combined_mean_h``
      - ``f1_min_weighted_h``, ``f1_max_weighted_h``, ``f1_combined_weighted_h``
    """
    def _prep(t: torch.Tensor, name: str) -> torch.Tensor:
        if t.ndim == 2:
            t = t.unsqueeze(0)
        if t.ndim == 4:
            if t.shape[0] != 1:
                raise ValueError(f"{name}: batch size must be 1, got {t.shape[0]}")
            t = t.squeeze(0)
        if t.ndim != 3:
            raise ValueError(f"{name}: expected (C,H,W), got {t.shape}")
        return t

    truth_min_onehot = _prep(truth_min_onehot, "truth_min_onehot")
    truth_max_onehot = _prep(truth_max_onehot, "truth_max_onehot")
    pred_onehot      = _prep(pred_onehot,      "pred_onehot")

    if truth_min_onehot.shape != pred_onehot.shape:
        raise ValueError(
            f"Shape mismatch: truth_min {truth_min_onehot.shape} vs pred {pred_onehot.shape}"
        )
    if truth_max_onehot.shape != pred_onehot.shape:
        raise ValueError(
            f"Shape mismatch: truth_max {truth_max_onehot.shape} vs pred {pred_onehot.shape}"
        )

    res_min = _get_f1_score_torch_detailed(truth_min_onehot, pred_onehot, axs=-2, kernel_size=kernel_size)
    res_max = _get_f1_score_torch_detailed(truth_max_onehot, pred_onehot, axs=-2, kernel_size=kernel_size)

    def _nanmean2(a: float, b: float) -> float:
        vals = [v for v in (a, b) if v == v]  # exclude NaN
        return float(np.mean(vals)) if vals else float("nan")

    nan = float("nan")
    return {
        "min": res_min,
        "max": res_max,
        "f1_min_mean":            res_min.get("f1_mean",            nan),
        "f1_max_mean":            res_max.get("f1_mean",            nan),
        "f1_combined_mean":       _nanmean2(res_min.get("f1_mean",  nan), res_max.get("f1_mean",  nan)),
        "f1_min_mean_h":          res_min.get("f1_mean_h",          nan),
        "f1_max_mean_h":          res_max.get("f1_mean_h",          nan),
        "f1_combined_mean_h":     _nanmean2(res_min.get("f1_mean_h",nan), res_max.get("f1_mean_h",nan)),
        "f1_min_weighted_h":      res_min.get("f1_mean_weighted_h", nan),
        "f1_max_weighted_h":      res_max.get("f1_mean_weighted_h", nan),
        "f1_combined_weighted_h": _nanmean2(
            res_min.get("f1_mean_weighted_h", nan),
            res_max.get("f1_mean_weighted_h", nan),
        ),
    }


def get_f1_score(
    min_max_idx_truth: np.ndarray,
    min_max_idx_ae: np.ndarray,
    axs: int = 1,
    kernel_size: int = 10,
) -> np.ndarray:
    """NumPy tolerant F1 for extrema masks."""
    if min_max_idx_truth.shape != min_max_idx_ae.shape:
        raise ValueError(f"Shape mismatch: {min_max_idx_truth.shape} vs {min_max_idx_ae.shape}")
    if kernel_size < 1:
        raise ValueError(f"kernel_size must be >= 1, got {kernel_size}")
    if _nd_convolve is None:
        raise ImportError("scipy.ndimage.convolve is required for get_f1_score")
    if axs < 0:
        axs = min_max_idx_truth.ndim + axs
    kernel_shape = [1] * min_max_idx_truth.ndim
    kernel_shape[axs] = kernel_size
    kernel = np.ones(kernel_shape, dtype=np.float64)
    truth = np.asarray(min_max_idx_truth)
    pred  = np.asarray(min_max_idx_ae)
    truth_expanded = _nd_convolve(truth, kernel, mode="constant", cval=0.0)
    ae_expanded    = _nd_convolve(pred,  kernel, mode="constant", cval=0.0)
    true_positives  = (truth_expanded > 0) & (pred > 0)
    false_positives = (truth_expanded == 0) & (pred > 0)
    false_negatives = (truth > 0) & (ae_expanded == 0)
    num_tp = np.sum(true_positives, axis=axs)
    num_fp = np.sum(false_positives, axis=axs)
    num_fn = np.sum(false_negatives, axis=axs)
    precision_den = num_tp + num_fp
    recall_den    = num_tp + num_fn
    precision = np.where(precision_den == 0, 0, num_tp / precision_den)
    recall    = np.where(recall_den    == 0, 0, num_tp / recall_den)
    sum_scores = precision + recall
    f1 = np.where(sum_scores == 0, 0, 2 * (precision * recall) / sum_scores)
    return f1
 
 
# ===========================================================================
# F1 with improved extrema detection (detect_extrema-based public API)
# ===========================================================================
 
def _dilate_along_h(mask: torch.Tensor, kernel_size: int) -> torch.Tensor:
    if kernel_size < 1:
        raise ValueError(f"kernel_size must be >= 1, got {kernel_size}")
    C, H, W = mask.shape
    pad = kernel_size // 2
    kernel = torch.ones(1, 1, kernel_size, device=mask.device, dtype=torch.float32)
    flat = mask.permute(0, 2, 1).reshape(C * W, 1, H)
    expanded = F.conv1d(flat, kernel, padding=pad)
    expanded = expanded[:, :, :H]
    return (expanded > 0).reshape(C, W, H).permute(0, 2, 1)
 
 
def _f1_from_masks(
    gt_mask: torch.Tensor,
    rec_mask: torch.Tensor,
    valid: torch.Tensor,
    kernel_size: int,
) -> Dict:
    if kernel_size < 1:
        raise ValueError(f"kernel_size must be >= 1, got {kernel_size}")
    C, H, W = gt_mask.shape
    gt_bin  = (gt_mask > 0) & valid
    rec_bin = (rec_mask > 0) & valid
    gt_win  = _dilate_along_h(gt_bin.float(), kernel_size)
    rec_win = _dilate_along_h(rec_bin.float(), kernel_size)
    f1_per_level: List[float] = []
    confusion: List[Dict] = []
    for h in range(H):
        t  = gt_bin[:, h, :]
        p  = rec_bin[:, h, :]
        tw = gt_win[:, h, :]
        pw = rec_win[:, h, :]
        valid_h = valid[:, h, :]
        tp_prec = int((tw & p).sum().item())
        tp_rec  = int((t & pw).sum().item())
        fp = int((~tw & p & valid_h).sum().item())
        fn = int((t & ~pw & valid_h).sum().item())
        tn = int((valid_h & ~t & ~p).sum().item())
        precision = tp_prec / (tp_prec + fp + 1e-12)
        recall    = tp_rec  / (tp_rec  + fn + 1e-12)
        f1 = 2.0 * precision * recall / (precision + recall + 1e-12)
        f1_per_level.append(float(f1))
        confusion.append({"tp": tp_prec + tp_rec, "fp": fp, "fn": fn, "tn": tn})
    f1_mean = float(np.mean(f1_per_level)) if f1_per_level else float("nan")
    return {"f1_mean": f1_mean, "f1_per_level": f1_per_level, "confusion": confusion}


def _safe_div(num: float, den: float) -> float:
    return float(num) / float(den) if den > 0 else 0.0


def _f1_from_precision_recall(precision: float, recall: float) -> float:
    return 2.0 * precision * recall / (precision + recall + 1e-12)


def _match_event_indices_1d(
    gt_idx: Union[torch.Tensor, np.ndarray, List[int], Tuple[int, ...]],
    pred_idx: Union[torch.Tensor, np.ndarray, List[int], Tuple[int, ...]],
    tolerance: int,
) -> List[Tuple[int, int, int]]:
    gt_arr = np.asarray(gt_idx, dtype=np.int64).reshape(-1)
    pred_arr = np.asarray(pred_idx, dtype=np.int64).reshape(-1)
    if gt_arr.size == 0 or pred_arr.size == 0:
        return []

    candidates: List[Tuple[int, int, int]] = []
    for ig, g in enumerate(gt_arr.tolist()):
        lo = np.searchsorted(pred_arr, g - tolerance, side="left")
        hi = np.searchsorted(pred_arr, g + tolerance, side="right")
        for ip in range(lo, hi):
            d = abs(int(g) - int(pred_arr[ip]))
            candidates.append((d, ig, ip))

    candidates.sort(key=lambda t: (t[0], t[1], t[2]))
    used_gt = np.zeros(gt_arr.size, dtype=bool)
    used_pred = np.zeros(pred_arr.size, dtype=bool)
    matches: List[Tuple[int, int, int]] = []
    for d, ig, ip in candidates:
        if used_gt[ig] or used_pred[ip]:
            continue
        used_gt[ig] = True
        used_pred[ip] = True
        matches.append((int(gt_arr[ig]), int(pred_arr[ip]), int(d)))
    return matches


def event_match_stats_per_level(
    truth_onehot: torch.Tensor,
    pred_onehot: torch.Tensor,
    kernel_size: int = 5,
    axs: int = -2,
) -> Dict[str, Any]:
    truth = _ensure_chw(truth_onehot, "truth_onehot") > 0
    pred = _ensure_chw(pred_onehot, "pred_onehot") > 0
    if truth.shape != pred.shape:
        raise ValueError(f"Shape mismatch: {tuple(truth.shape)} vs {tuple(pred.shape)}")
    if kernel_size < 1:
        raise ValueError(f"kernel_size must be >= 1, got {kernel_size}")

    ndim = truth.ndim
    if axs < 0:
        axs = ndim + axs

    truth_h_last = truth.movedim(axs, -1)
    pred_h_last = pred.movedim(axs, -1)
    *prefix, H = truth_h_last.shape
    n_profiles = int(np.prod(prefix)) if prefix else 1
    truth_flat = truth_h_last.reshape(n_profiles, H).detach().cpu().numpy().astype(bool, copy=False)
    pred_flat = pred_h_last.reshape(n_profiles, H).detach().cpu().numpy().astype(bool, copy=False)
    tolerance = kernel_size // 2

    gt_count_per_h = np.zeros(H, dtype=np.int64)
    pred_count_per_h = np.zeros(H, dtype=np.int64)
    matched_gt_per_h = np.zeros(H, dtype=np.int64)
    matched_pred_per_h = np.zeros(H, dtype=np.int64)
    offset_sum_per_h = np.zeros(H, dtype=np.float64)
    offset_count_per_h = np.zeros(H, dtype=np.int64)

    gt_count_per_h += truth_flat.sum(axis=0, dtype=np.int64)
    pred_count_per_h += pred_flat.sum(axis=0, dtype=np.int64)

    total_gt = int(gt_count_per_h.sum())
    total_pred = int(pred_count_per_h.sum())
    total_matches = 0
    total_abs_offset = 0.0

    active = np.flatnonzero(truth_flat.any(axis=1) | pred_flat.any(axis=1))
    for prof_idx in active.tolist():
        gt_idx = np.flatnonzero(truth_flat[prof_idx])
        pred_idx = np.flatnonzero(pred_flat[prof_idx])
        matches = _match_event_indices_1d(gt_idx, pred_idx, tolerance=tolerance)
        total_matches += len(matches)
        for g, p, d in matches:
            matched_gt_per_h[g] += 1
            matched_pred_per_h[p] += 1
            offset_sum_per_h[g] += float(d)
            offset_count_per_h[g] += 1
            total_abs_offset += float(d)

    match_stats_per_h: List[Dict[str, Any]] = []
    event_f1_per_h: List[float] = []
    for h in range(H):
        gt_count = int(gt_count_per_h[h])
        pred_count = int(pred_count_per_h[h])
        matched_gt = int(matched_gt_per_h[h])
        matched_pred = int(matched_pred_per_h[h])
        recall_h = _safe_div(matched_gt, gt_count)
        precision_h = _safe_div(matched_pred, pred_count)
        f1_h = _f1_from_precision_recall(precision_h, recall_h)
        event_f1_per_h.append(f1_h)
        match_stats_per_h.append(
            {
                "gt_count": gt_count,
                "pred_count": pred_count,
                "matched_gt": matched_gt,
                "matched_pred": matched_pred,
                "fn": gt_count - matched_gt,
                "fp": pred_count - matched_pred,
                "recall": float(recall_h),
                "precision": float(precision_h),
                "f1": float(f1_h),
                "mean_abs_offset": float(offset_sum_per_h[h] / offset_count_per_h[h]) if offset_count_per_h[h] > 0 else float("nan"),
            }
        )

    overall_precision = _safe_div(total_matches, total_pred)
    overall_recall = _safe_div(total_matches, total_gt)
    overall_f1 = _f1_from_precision_recall(overall_precision, overall_recall)
    weights = gt_count_per_h.astype(np.float64)
    total_weight = float(weights.sum())

    return {
        "match_stats_per_h": match_stats_per_h,
        "gt_count_per_h": [int(v) for v in gt_count_per_h.tolist()],
        "pred_count_per_h": [int(v) for v in pred_count_per_h.tolist()],
        "matched_gt_per_h": [int(v) for v in matched_gt_per_h.tolist()],
        "matched_pred_per_h": [int(v) for v in matched_pred_per_h.tolist()],
        "event_f1_per_h": [float(v) for v in event_f1_per_h],
        "event_f1_mean_h": float(np.mean(event_f1_per_h)) if event_f1_per_h else float("nan"),
        "event_f1_mean_weighted_h": float(np.dot(np.asarray(event_f1_per_h, dtype=np.float64), weights) / total_weight) if total_weight > 0 else float("nan"),
        "overall_event_stats": {
            "gt_count": int(total_gt),
            "pred_count": int(total_pred),
            "tp": int(total_matches),
            "fp": int(total_pred - total_matches),
            "fn": int(total_gt - total_matches),
            "precision": float(overall_precision),
            "recall": float(overall_recall),
            "f1": float(overall_f1),
            "mean_abs_offset": float(total_abs_offset / total_matches) if total_matches > 0 else float("nan"),
        },
    }


def f1_and_match_stats_per_level(
    truth_onehot: torch.Tensor,
    pred_onehot: torch.Tensor,
    kernel_size: int = 5,
) -> Dict[str, Any]:
    truth_onehot = _ensure_chw(truth_onehot, "truth_onehot")
    pred_onehot = _ensure_chw(pred_onehot, "pred_onehot")
    if truth_onehot.shape != pred_onehot.shape:
        raise ValueError(f"Shape mismatch: {tuple(truth_onehot.shape)} vs {tuple(pred_onehot.shape)}")
    return event_match_stats_per_level(
        truth_onehot=truth_onehot,
        pred_onehot=pred_onehot,
        kernel_size=kernel_size,
        axs=-2,
    )


def extrema_f1_along_h(
    gt_mask: torch.Tensor,
    pred_mask: torch.Tensor,
    tolerance: int = 2,
) -> Dict[str, Any]:
    gt_mask = _ensure_chw(gt_mask, "gt_mask") > 0
    pred_mask = _ensure_chw(pred_mask, "pred_mask") > 0
    if gt_mask.shape != pred_mask.shape:
        raise ValueError(f"Shape mismatch: {tuple(gt_mask.shape)} vs {tuple(pred_mask.shape)}")
    if tolerance < 0:
        raise ValueError(f"tolerance must be >= 0, got {tolerance}")

    C, H, W = gt_mask.shape
    gt_np = gt_mask.detach().cpu().numpy().astype(bool, copy=False)
    pred_np = pred_mask.detach().cpu().numpy().astype(bool, copy=False)

    tp_total = 0
    fp_total = 0
    fn_total = 0

    match_stats_per_h: List[Dict[str, Any]] = []
    gt_count_per_h = np.zeros(H, dtype=np.int64)
    pred_count_per_h = np.zeros(H, dtype=np.int64)
    matched_gt_per_h = np.zeros(H, dtype=np.int64)
    matched_pred_per_h = np.zeros(H, dtype=np.int64)
    offset_sum_per_h = np.zeros(H, dtype=np.float64)
    offset_count_per_h = np.zeros(H, dtype=np.int64)

    for c in range(C):
        for w in range(W):
            gt_idx = np.flatnonzero(gt_np[c, :, w])
            pred_idx = np.flatnonzero(pred_np[c, :, w])
            gt_count_per_h[gt_idx] += 1
            pred_count_per_h[pred_idx] += 1

            matches = _match_event_indices_1d(gt_idx, pred_idx, tolerance=tolerance)
            n_tp = len(matches)
            n_fp = int(pred_idx.size - n_tp)
            n_fn = int(gt_idx.size - n_tp)
            tp_total += n_tp
            fp_total += n_fp
            fn_total += n_fn

            for g, p, d in matches:
                matched_gt_per_h[g] += 1
                matched_pred_per_h[p] += 1
                offset_sum_per_h[g] += float(d)
                offset_count_per_h[g] += 1

    event_f1_per_h: List[float] = []
    for h in range(H):
        gt_count = int(gt_count_per_h[h])
        pred_count = int(pred_count_per_h[h])
        matched_gt = int(matched_gt_per_h[h])
        matched_pred = int(matched_pred_per_h[h])
        recall_h = _safe_div(matched_gt, gt_count)
        precision_h = _safe_div(matched_pred, pred_count)
        f1_h = _f1_from_precision_recall(precision_h, recall_h)
        event_f1_per_h.append(f1_h)
        match_stats_per_h.append(
            {
                "gt_count": gt_count,
                "pred_count": pred_count,
                "matched_gt": matched_gt,
                "matched_pred": matched_pred,
                "fn": gt_count - matched_gt,
                "fp": pred_count - matched_pred,
                "recall": float(recall_h),
                "precision": float(precision_h),
                "f1": float(f1_h),
                "mean_abs_offset": float(offset_sum_per_h[h] / offset_count_per_h[h]) if offset_count_per_h[h] > 0 else float("nan"),
            }
        )

    precision = _safe_div(tp_total, tp_total + fp_total)
    recall = _safe_div(tp_total, tp_total + fn_total)
    f1 = _f1_from_precision_recall(precision, recall)

    return {
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "tp": int(tp_total),
        "fp": int(fp_total),
        "fn": int(fn_total),
        "match_stats_per_h": match_stats_per_h,
        "gt_count_per_h": [int(v) for v in gt_count_per_h.tolist()],
        "pred_count_per_h": [int(v) for v in pred_count_per_h.tolist()],
        "matched_gt_per_h": [int(v) for v in matched_gt_per_h.tolist()],
        "matched_pred_per_h": [int(v) for v in matched_pred_per_h.tolist()],
        "event_f1_per_h": [float(v) for v in event_f1_per_h],
        "event_f1_mean_h": float(np.mean(event_f1_per_h)) if event_f1_per_h else float("nan"),
    }


def f1_score_v2(
    truth_onehot: torch.Tensor,
    pred_onehot: torch.Tensor,
    kernel_size: int = 5,
) -> float:
    """Event-based tolerant F1 for extrema along H."""
    if truth_onehot.shape != pred_onehot.shape:
        raise ValueError(f"Shape mismatch: {truth_onehot.shape} vs {pred_onehot.shape}")

    if truth_onehot.ndim == 2:
        truth_onehot = truth_onehot.unsqueeze(0)
        pred_onehot = pred_onehot.unsqueeze(0)

    if truth_onehot.ndim == 4:
        if truth_onehot.shape[0] != 1:
            raise ValueError("Batch size > 1 not supported; pass one sample at a time.")
        truth_onehot = truth_onehot.squeeze(0)
        pred_onehot = pred_onehot.squeeze(0)

    if truth_onehot.ndim != 3:
        raise ValueError(f"Expected (C,H,W) tensor, got shape {truth_onehot.shape}")

    tolerance = kernel_size // 2
    out = extrema_f1_along_h(
        gt_mask=truth_onehot,
        pred_mask=pred_onehot,
        tolerance=tolerance,
    )
    return float(out["f1"])
 
 
def f1_extrema(
    gt: torch.Tensor,
    rec: torch.Tensor,
    kind: str = "both",
    kernel_size: int = 5,
    filtered_rec: Optional[torch.Tensor] = None,
    grad_eps: Union[float, str] = 0,
    smooth_window: int = 1,
    min_separation: int = 3,
    fill_zero_signs: bool = True,
    detector_version: str = "v2",
    side_window: int = 5,
    min_support_count: int = 1,
) -> Dict[str, Any]:
    """Tolerant F1 using improved extrema detection.

    When grad_eps="auto", the threshold is always derived from GT and applied
    to both GT and reconstruction for consistent comparison.
    """
    gt = _ensure_chw(gt)
    rec = _ensure_chw(rec)
    valid = _nan_mask(gt)

    if detector_version == "v2":
        kwargs = dict(
            kind=kind,
            grad_eps=grad_eps,
            smooth_window=smooth_window,
            min_separation=min_separation,
            fill_zero_signs=fill_zero_signs,
            side_window=side_window,
            min_support_count=min_support_count,
        )
        gt_mask = detect_extrema_v2(gt, **kwargs)
        rec_mask = detect_extrema_v2(rec, **kwargs, grad_eps_reference=gt)
    elif detector_version == "v1":
        kwargs = dict(
            kind=kind,
            grad_eps=grad_eps,
            smooth_window=smooth_window,
            min_separation=min_separation,
            fill_zero_signs=fill_zero_signs,
        )
        gt_mask = detect_extrema(gt, **kwargs)
        rec_mask = detect_extrema(rec, **kwargs, grad_eps_reference=gt)
    else:
        raise ValueError(f"detector_version must be 'v1' or 'v2', got {detector_version!r}")

    result = _f1_from_masks(gt_mask, rec_mask, valid, kernel_size)
    if filtered_rec is not None:
        filtered_rec = _ensure_chw(filtered_rec)
        if detector_version == "v2":
            rec_mask_f = detect_extrema_v2(filtered_rec, **kwargs, grad_eps_reference=gt)
        else:
            rec_mask_f = detect_extrema(filtered_rec, **kwargs, grad_eps_reference=gt)
        res_f = _f1_from_masks(gt_mask, rec_mask_f, valid, kernel_size)
        result["f1_mean_filtered"] = res_f["f1_mean"]
        result["f1_per_level_filtered"] = res_f["f1_per_level"]
        result["confusion_filtered"] = res_f["confusion"]
    return result
 
 
# ===========================================================================
# MSE per H level
# ===========================================================================
 
def masked_mse_and_ratio_per_level(
    gt: torch.Tensor,
    rec: torch.Tensor,
    valid_mask: Optional[torch.Tensor] = None,
    eps: float = 1e-12,
) -> Dict[str, Any]:
    """
    Per-level (H dimension) MSE and MSE/std(GT), ignoring invalid GT values.
    gt/rec: (C,H,W) or (B,C,H,W) with B=1.
    """
    if gt.shape != rec.shape:
        raise ValueError(f"Shape mismatch: {gt.shape} vs {rec.shape}")
    if gt.ndim == 4:
        if gt.shape[0] != 1:
            raise ValueError("Batch size > 1 not supported.")
        gt  = gt.squeeze(0)
        rec = rec.squeeze(0)
    if gt.ndim != 3:
        raise ValueError(f"Expected (C,H,W), got {gt.shape}")
 
    valid_np = None
    if valid_mask is not None:
        vm = valid_mask
        if vm.ndim == 4:
            if vm.shape[0] != 1:
                raise ValueError("Batch size > 1 not supported.")
            vm = vm.squeeze(0)
        if vm.shape != gt.shape:
            raise ValueError(f"valid_mask shape mismatch: {vm.shape} vs {gt.shape}")
        valid_np = vm.detach().cpu().numpy().astype(bool)
 
    gt_np  = gt.detach().cpu().numpy().astype(np.float64)
    rec_np = rec.detach().cpu().numpy().astype(np.float64)
    H = gt_np.shape[1]
 
    mse_per_level: List[float] = []
    mse_over_std_per_level: List[float] = []
 
    for h in range(H):
        g = gt_np[:, h, :]
        r = rec_np[:, h, :]
        valid = np.isfinite(g)
        if valid_np is not None:
            valid = valid & valid_np[:, h, :]
        if not np.any(valid):
            mse_per_level.append(float("nan"))
            mse_over_std_per_level.append(float("nan"))
            continue
        g_valid = g[valid]
        r_valid = np.nan_to_num(r[valid], nan=0.0, posinf=0.0, neginf=0.0)
        mse_val = float(np.mean((g_valid - r_valid) ** 2))
        std_val = float(np.std(g_valid, ddof=0))
        mse_per_level.append(mse_val)
        mse_over_std_per_level.append(mse_val / (std_val + eps))
 
    return {
        "mse_per_level":                mse_per_level,
        "mse_over_std_per_level":       mse_over_std_per_level,
        "mean_mse_per_level":           float(np.nanmean(mse_per_level))         if mse_per_level         else float("nan"),
        "mean_mse_over_std_per_level":  float(np.nanmean(mse_over_std_per_level)) if mse_over_std_per_level else float("nan"),
    }
 
 
# ===========================================================================
# DTW and extrema depth/amplitude errors
# ===========================================================================
 
def _count_extrema_1d(profile: np.ndarray) -> int:
    if profile.size < 3:
        return 0
    grad = np.diff(profile)
    sign = np.sign(grad)
    turning = np.diff(sign)
    return int(np.sum(turning != 0))
 
 
def dtw_profiles(
    gt: torch.Tensor,
    rec: torch.Tensor,
    window: Optional[int] = None,
) -> Dict:
    """DTW between GT and REC H-profiles, scaled to RMSE-like magnitude.

    We convert DTW distance to an RMSE-like scale by dividing by sqrt(profile_length),
    so values are directly comparable in magnitude to RMSE (same physical units).
    """
    gt  = _ensure_chw(gt)
    rec = _ensure_chw(rec)
    C, H, W = gt.shape
    gt_np  = gt.detach().cpu().numpy().astype(np.float64)
    rec_np = rec.detach().cpu().numpy().astype(np.float64)
    gt_profiles  = gt_np.reshape(C, H, W).transpose(0, 2, 1).reshape(C * W, H)
    rec_profiles = rec_np.reshape(C, H, W).transpose(0, 2, 1).reshape(C * W, H)
    distances: List[float] = []
    n_used = 0
    for gt_p, rec_p in zip(gt_profiles, rec_profiles):
        if not (np.all(np.isfinite(gt_p)) and np.all(np.isfinite(rec_p))):
            continue
        n_used += 1
        prof_len = max(1, int(gt_p.shape[0]))
        rmse_scale = float(np.sqrt(prof_len))
        if _dtw_lib is not None:
            dist = float(_dtw_lib.distance_fast(
                gt_p.astype(np.float64, copy=False),
                rec_p.astype(np.float64, copy=False),
                window=window, use_pruning=True,
            ))
            dist = dist / rmse_scale
        else:
            # Fallback: Euclidean RMSE along profile (same unit and scale family as DTW-RMSE).
            dist = float(np.sqrt(np.mean((gt_p - rec_p) ** 2)))
        if np.isfinite(dist):
            distances.append(float(dist))
    return {
        "dtw_mean":         float(np.mean(distances)) if distances else float("nan"),
        "dtw_std":          float(np.std(distances))  if distances else float("nan"),
        "n_profiles_used":  n_used,
        "n_profiles_total": int(C * W),
        "dtw_available":    _dtw_lib is not None,
        "dtw_normalization": "rmse_like_sqrt_profile_len",
    }
 
 
def compute_dtw_and_extrema_metrics(
    gt_3d: np.ndarray,
    rec_3d: np.ndarray,
    dims3: Optional[Tuple[str, str, str]] = None,
    window: Optional[int] = None,
    prominence: float = 0.5,
) -> Dict[str, Any]:
    """Profile-wise DTW and extrema errors on 3D (D0, D1, D2) fields."""
    if gt_3d.shape != rec_3d.shape:
        raise ValueError(f"Shape mismatch: {gt_3d.shape} vs {rec_3d.shape}")
    if gt_3d.ndim != 3:
        raise ValueError(f"Expected 3D arrays, got {gt_3d.shape}")
    depth_axis = pick_depth_axis(dims3)
    gt_depth_first  = np.moveaxis(gt_3d,  depth_axis, 0)
    rec_depth_first = np.moveaxis(rec_3d, depth_axis, 0)
    n_depth = int(gt_depth_first.shape[0])
    gt_profiles  = gt_depth_first.reshape(n_depth, -1).T
    rec_profiles = rec_depth_first.reshape(n_depth, -1).T
 
    dtw_distances: List[float] = []
    extrema_depth_errs: List[float] = []
    extrema_amp_errs:   List[float] = []
    gt_extrema_counts:  List[int]   = []
    rec_extrema_counts: List[int]   = []
    used_profiles = 0
 
    def _gradient_extrema_indices_1d(profile: np.ndarray) -> np.ndarray:
        if profile.ndim != 1 or profile.size < 3:
            return np.empty((0,), dtype=np.int64)
        grad = np.diff(profile)
        grad_sign = np.sign(grad)
        turning = np.diff(grad_sign)
        return np.where(turning != 0)[0].astype(np.int64) + 1
 
    for gt_profile, rec_profile in zip(gt_profiles, rec_profiles):
        if not bool(np.all(np.isfinite(gt_profile)) and np.all(np.isfinite(rec_profile))):
            continue
        used_profiles += 1
        gt_extrema  = _gradient_extrema_indices_1d(gt_profile)
        rec_extrema = _gradient_extrema_indices_1d(rec_profile)
        gt_extrema_counts.append(int(gt_extrema.size))
        rec_extrema_counts.append(int(rec_extrema.size))
        prof_len = max(1, int(gt_profile.shape[0]))
        rmse_scale = float(np.sqrt(prof_len))
        if _dtw_lib is not None:
            dist = _dtw_lib.distance_fast(
                gt_profile.astype(np.float64, copy=False),
                rec_profile.astype(np.float64, copy=False),
                window=window, use_pruning=True,
            )
            dist = float(dist) / rmse_scale
        else:
            dist = float(np.sqrt(np.mean((gt_profile - rec_profile) ** 2)))
        if np.isfinite(dist):
            dtw_distances.append(float(dist))
        if rec_extrema.size == 0 or gt_extrema.size == 0:
            continue
        for gt_idx in gt_extrema.tolist():
            closest = int(rec_extrema[np.argmin(np.abs(rec_extrema - gt_idx))])
            extrema_depth_errs.append(float(abs(int(gt_idx) - closest)))
            extrema_amp_errs.append(float(abs(float(gt_profile[gt_idx]) - float(rec_profile[closest]))))
 
    return {
        "dtw_mean":               float(np.mean(dtw_distances))     if dtw_distances     else float("nan"),
        "dtw_std":                float(np.std(dtw_distances))      if dtw_distances     else float("nan"),
        "extrema_depth_err_mean": float(np.mean(extrema_depth_errs)) if extrema_depth_errs else float("nan"),
        "extrema_amp_err_mean":   float(np.mean(extrema_amp_errs))   if extrema_amp_errs   else float("nan"),
        "gt_extrema_count_mean":  float(np.mean(gt_extrema_counts))  if gt_extrema_counts  else float("nan"),
        "rec_extrema_count_mean": float(np.mean(rec_extrema_counts)) if rec_extrema_counts else float("nan"),
        "num_profiles_used":      int(used_profiles),
        "num_profiles_total":     int(gt_profiles.shape[0]),
        "dtw_enabled":            bool(_dtw_lib is not None),
        "dtw_normalization":      "rmse_like_sqrt_profile_len",
        "extrema_enabled":        True,
    }


def _dtw_multidim_sequence_distance(
    seq_a: np.ndarray,
    seq_b: np.ndarray,
    weights: Optional[np.ndarray] = None,
    window: Optional[int] = None,
) -> float:
    """DTW distance between two N-D point sequences after per-dimension standardization."""
    seq_a = np.asarray(seq_a, dtype=np.float64)
    seq_b = np.asarray(seq_b, dtype=np.float64)
    if seq_a.ndim != 2 or seq_b.ndim != 2 or seq_a.shape[1] != seq_b.shape[1]:
        raise ValueError(f"Expected two 2D arrays with same feature count, got {seq_a.shape} vs {seq_b.shape}")
    if seq_a.shape[0] == 0 or seq_b.shape[0] == 0:
        return float("nan")

    all_pts = np.vstack([seq_a, seq_b])
    mu = np.mean(all_pts, axis=0)
    sig = np.std(all_pts, axis=0) + 1e-8
    a_std = (seq_a - mu) / sig
    b_std = (seq_b - mu) / sig

    feat_dim = int(seq_a.shape[1])
    if weights is None:
        w = np.ones((feat_dim,), dtype=np.float64)
    else:
        w = np.asarray(weights, dtype=np.float64).reshape(-1)
        if w.size != feat_dim:
            raise ValueError(f"weights must have {feat_dim} entries, got {w.size}")

    a_weighted = a_std * w
    b_weighted = b_std * w
    if window is None and _dtw_ndim_lib is not None:
        try:
            return float(
                _dtw_ndim_lib.distance_fast(
                    np.ascontiguousarray(a_weighted, dtype=np.double),
                    np.ascontiguousarray(b_weighted, dtype=np.double),
                    window=None,
                    use_pruning=False,
                    inner_dist="euclidean",
                )
            )
        except Exception:
            pass

    n, m = int(a_std.shape[0]), int(b_std.shape[0])
    cost = np.full((n, m), np.inf, dtype=np.float64)
    window_eff = max(n, m) if window is None else max(0, int(window))

    def _local_dist(i: int, j: int) -> float:
        diff = a_weighted[i] - b_weighted[j]
        return float(np.sqrt(np.dot(diff, diff)))

    for i in range(n):
        j_lo = 0 if window is None else max(0, i - window_eff)
        j_hi = m if window is None else min(m, i + window_eff + 1)
        for j in range(j_lo, j_hi):
            d = _local_dist(i, j)
            if i == 0 and j == 0:
                cost[i, j] = d
                continue
            prev = []
            if i > 0:
                prev.append(cost[i - 1, j])
            if j > 0:
                prev.append(cost[i, j - 1])
            if i > 0 and j > 0:
                prev.append(cost[i - 1, j - 1])
            if prev:
                cost[i, j] = d + min(prev)

    return float(cost[n - 1, m - 1])


def _gt_profile_prominence_threshold(
    prominence: float,
    prominence_ratio: Optional[float],
    reference_profile: Optional[np.ndarray],
) -> float:
    if prominence_ratio is None:
        return max(1e-12, float(prominence))

    if reference_profile is None:
        return max(1e-12, float(prominence_ratio))

    ref = np.asarray(reference_profile, dtype=np.float64).reshape(-1)
    ref = ref[np.isfinite(ref)]
    scale = float(np.std(ref, ddof=0)) if ref.size > 0 else 0.0
    return max(1e-12, float(prominence_ratio) * max(scale, 1e-12))


def _extract_profile_extrema_sequence(
    profile: np.ndarray,
    kind: str,
    prominence: float,
    min_distance_h: int,
    include_kind_dim: bool = False,
    prominence_ratio: Optional[float] = None,
    prominence_reference_profile: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Return ordered extrema points for one 1D profile."""
    if _scipy_signal is None:
        raise ImportError("scipy.signal is required for extrema DTW metrics")

    arr = np.asarray(profile, dtype=np.float64).reshape(-1)
    feat_dim = 3 if include_kind_dim else 2
    if arr.size < 3 or not np.all(np.isfinite(arr)):
        return np.empty((0, feat_dim), dtype=np.float64)

    dist_h = max(1, int(min_distance_h))
    prom = _gt_profile_prominence_threshold(
        prominence=prominence,
        prominence_ratio=prominence_ratio,
        reference_profile=prominence_reference_profile,
    )
    records: List[Tuple[float, ...]] = []

    if kind in ("max", "both"):
        peaks, _ = _scipy_signal.find_peaks(arr, prominence=prom, distance=dist_h)
        for idx in peaks.tolist():
            if include_kind_dim:
                records.append((float(idx), float(arr[idx]), 1.0))
            else:
                records.append((float(idx), float(arr[idx])))

    if kind in ("min", "both"):
        peaks, _ = _scipy_signal.find_peaks(-arr, prominence=prom, distance=dist_h)
        for idx in peaks.tolist():
            if include_kind_dim:
                records.append((float(idx), float(arr[idx]), -1.0))
            else:
                records.append((float(idx), float(arr[idx])))

    if not records:
        return np.empty((0, feat_dim), dtype=np.float64)

    seq = np.asarray(records, dtype=np.float64)
    order = np.argsort(seq[:, 0], kind="mergesort")
    return seq[order]


def compute_extrema_dtw_metrics(
    gt_3d: np.ndarray,
    rec_3d: np.ndarray,
    dims3: Optional[Tuple[str, str, str]] = None,
    window: Optional[int] = None,
    prominence: float = 0.5,
    prominence_ratio: Optional[float] = None,
    min_distance_h: int = 5,
    depth_weight: float = 1.0,
    amplitude_weight: float = 1.0,
    kind_weight: float = 2.0,
) -> Dict[str, Any]:
    """Compute per-profile DTW on extrema sequences normalized by GT extrema count."""
    if gt_3d.shape != rec_3d.shape:
        raise ValueError(f"Shape mismatch: {gt_3d.shape} vs {rec_3d.shape}")
    if gt_3d.ndim != 3:
        raise ValueError(f"Expected 3D arrays, got {gt_3d.shape}")

    depth_axis = pick_depth_axis(dims3)
    gt_depth_first = np.moveaxis(gt_3d, depth_axis, 0)
    rec_depth_first = np.moveaxis(rec_3d, depth_axis, 0)
    n_depth = int(gt_depth_first.shape[0])
    gt_profiles = gt_depth_first.reshape(n_depth, -1).T
    rec_profiles = rec_depth_first.reshape(n_depth, -1).T

    dists_min: List[float] = []
    dists_max: List[float] = []
    dists_both: List[float] = []
    dists_combined: List[float] = []
    used_profiles = 0

    def _normalize_by_gt_count(dist: float, gt_count: int) -> float:
        if not np.isfinite(dist):
            return float("nan")
        return float(dist) / float(max(1, int(gt_count)))

    def _profile_distances(gt_profile: np.ndarray, rec_profile: np.ndarray) -> Tuple[bool, float, float, float, float]:
        if not bool(np.all(np.isfinite(gt_profile)) and np.all(np.isfinite(rec_profile))):
            return False, float("nan"), float("nan"), float("nan"), float("nan")

        gt_min = _extract_profile_extrema_sequence(
            gt_profile,
            "min",
            prominence,
            min_distance_h,
            prominence_ratio=prominence_ratio,
            prominence_reference_profile=gt_profile,
        )
        rec_min = _extract_profile_extrema_sequence(
            rec_profile,
            "min",
            prominence,
            min_distance_h,
            prominence_ratio=prominence_ratio,
            prominence_reference_profile=gt_profile,
        )
        gt_max = _extract_profile_extrema_sequence(
            gt_profile,
            "max",
            prominence,
            min_distance_h,
            prominence_ratio=prominence_ratio,
            prominence_reference_profile=gt_profile,
        )
        rec_max = _extract_profile_extrema_sequence(
            rec_profile,
            "max",
            prominence,
            min_distance_h,
            prominence_ratio=prominence_ratio,
            prominence_reference_profile=gt_profile,
        )
        gt_both = _extract_profile_extrema_sequence(
            gt_profile,
            "both",
            prominence,
            min_distance_h,
            include_kind_dim=True,
            prominence_ratio=prominence_ratio,
            prominence_reference_profile=gt_profile,
        )
        rec_both = _extract_profile_extrema_sequence(
            rec_profile,
            "both",
            prominence,
            min_distance_h,
            include_kind_dim=True,
            prominence_ratio=prominence_ratio,
            prominence_reference_profile=gt_profile,
        )

        dtw_min = _dtw_multidim_sequence_distance(
            gt_min,
            rec_min,
            weights=np.array([depth_weight, amplitude_weight], dtype=np.float64),
            window=window,
        )
        dtw_max = _dtw_multidim_sequence_distance(
            gt_max,
            rec_max,
            weights=np.array([depth_weight, amplitude_weight], dtype=np.float64),
            window=window,
        )
        dtw_both = _dtw_multidim_sequence_distance(
            gt_both,
            rec_both,
            weights=np.array([depth_weight, amplitude_weight, kind_weight], dtype=np.float64),
            window=window,
        )

        gt_min_count = int(gt_min.shape[0])
        gt_max_count = int(gt_max.shape[0])
        gt_both_count = int(gt_both.shape[0])
        dtw_min = _normalize_by_gt_count(dtw_min, gt_min_count)
        dtw_max = _normalize_by_gt_count(dtw_max, gt_max_count)
        dtw_both = _normalize_by_gt_count(dtw_both, gt_both_count)

        combined_num = 0.0
        combined_den = 0
        if np.isfinite(dtw_min) and gt_min_count > 0:
            combined_num += float(dtw_min) * float(gt_min_count)
            combined_den += gt_min_count
        if np.isfinite(dtw_max) and gt_max_count > 0:
            combined_num += float(dtw_max) * float(gt_max_count)
            combined_den += gt_max_count
        if combined_den > 0:
            dtw_combined = float(combined_num / float(combined_den))
        else:
            dtw_combined = float("nan")

        return True, float(dtw_min), float(dtw_max), float(dtw_both), float(dtw_combined)

    def _consume_result(result: Tuple[bool, float, float, float, float]) -> None:
        nonlocal used_profiles
        used, dtw_min, dtw_max, dtw_both, dtw_combined = result
        if not used:
            return
        used_profiles += 1
        if np.isfinite(dtw_min):
            dists_min.append(float(dtw_min))
        if np.isfinite(dtw_max):
            dists_max.append(float(dtw_max))
        if np.isfinite(dtw_both):
            dists_both.append(float(dtw_both))
        if np.isfinite(dtw_combined):
            dists_combined.append(float(dtw_combined))

    try:
        worker_count = int(os.environ.get("EXTREMA_DTW_WORKERS", "1"))
    except Exception:
        worker_count = 1
    worker_count = max(1, worker_count)

    if worker_count > 1 and gt_profiles.shape[0] > 1:
        from concurrent.futures import ThreadPoolExecutor

        with ThreadPoolExecutor(max_workers=worker_count) as pool:
            for result in pool.map(lambda pair: _profile_distances(pair[0], pair[1]), zip(gt_profiles, rec_profiles)):
                _consume_result(result)
    else:
        for gt_profile, rec_profile in zip(gt_profiles, rec_profiles):
            _consume_result(_profile_distances(gt_profile, rec_profile))

    def _mean_or_nan(vals: List[float]) -> float:
        return float(np.mean(vals)) if vals else float("nan")

    return {
        "extrema_dtw_min": float(_mean_or_nan(dists_min)),
        "extrema_dtw_max": float(_mean_or_nan(dists_max)),
        "extrema_dtw_both": float(_mean_or_nan(dists_both)),
        "extrema_dtw_combined": float(_mean_or_nan(dists_combined)),
        "num_profiles_used": int(used_profiles),
        "num_profiles_total": int(gt_profiles.shape[0]),
        "extrema_dtw_prominence": float(prominence),
        "extrema_dtw_prominence_ratio": (None if prominence_ratio is None else float(prominence_ratio)),
        "extrema_dtw_min_distance_h": int(min_distance_h),
        "extrema_dtw_window": (None if window is None else int(window)),
        "extrema_dtw_normalization": "per_gt_extrema_count_max1",
    }
 
 
# ===========================================================================
# Image quality metrics
# ===========================================================================
 
def _masked_global_ssim(
    a_t: torch.Tensor, b_t: torch.Tensor, valid_mask: torch.Tensor, max_val: float
) -> float:
    v = valid_mask.bool()
    n_valid = int(v.sum().item())
    if n_valid == 0:
        return float("nan")
    a_valid = a_t[v]
    b_valid = b_t[v]
    mu_a = a_valid.mean()
    mu_b = b_valid.mean()
    da = a_valid - mu_a
    db = b_valid - mu_b
    var_a  = (da * da).mean()
    var_b  = (db * db).mean()
    cov_ab = (da * db).mean()
    c1 = (0.01 * max_val) ** 2
    c2 = (0.03 * max_val) ** 2
    num = (2.0 * mu_a * mu_b + c1) * (2.0 * cov_ab + c2)
    den = (mu_a * mu_a + mu_b * mu_b + c1) * (var_a + var_b + c2)
    if den == 0:
        return 1.0 if num == 0 else float("nan")
    return float((num / den).item())
 
 
def compute_metrics(
    a: Union[np.ndarray, Image.Image],
    b: Union[np.ndarray, Image.Image],
    max_val: float = 255.0,
) -> Tuple[float, float]:
    """Returns PSNR and SSIM between images `a` and `b`."""
    if isinstance(a, Image.Image):
        a = np.asarray(a)
    if isinstance(b, Image.Image):
        b = np.asarray(b)
 
    def to_nchw(x):
        t = torch.from_numpy(np.ascontiguousarray(x)).float()
        if t.ndim == 2:
            t = t.unsqueeze(0)
        elif t.ndim == 3 and t.shape[-1] in (1, 3):
            t = t.permute(2, 0, 1)
        return t.unsqueeze(0)
 
    a_t = to_nchw(a)
    b_t = to_nchw(b)
    valid_mask = torch.isfinite(a_t) & torch.isfinite(b_t)
    a_t = torch.where(valid_mask, a_t, torch.zeros_like(a_t))
    b_t = torch.where(valid_mask, b_t, torch.zeros_like(b_t))
    n_valid = int(valid_mask.sum().item())
 
    if n_valid == 0:
        return float("nan"), float("nan")
    mse = (((a_t - b_t) ** 2) * valid_mask).sum().item() / float(n_valid)
    p   = float("inf") if mse == 0 else 20 * np.log10(max_val) - 10 * np.log10(mse)
    m   = ssim(a_t, b_t, data_range=max_val, size_average=True).item() if bool(valid_mask.all()) \
          else _masked_global_ssim(a_t, b_t, valid_mask, max_val=max_val)
    return p, m
 
 
# ===========================================================================
# MAE
# ===========================================================================

def mae_np(
    a: np.ndarray,
    b: np.ndarray,
    valid_mask: Optional[np.ndarray] = None,
) -> float:
    """NaN-safe Mean Absolute Error between two arrays.

    Ignores positions where ``a`` is not finite. Optionally further restricted
    by ``valid_mask``.
    """
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    if a.shape != b.shape:
        raise ValueError(f"Shape mismatch: {a.shape} vs {b.shape}")
    m = np.isfinite(a)
    if valid_mask is not None:
        vm = np.asarray(valid_mask, dtype=bool)
        if vm.shape != a.shape:
            raise ValueError(f"valid_mask shape mismatch: {vm.shape} vs {a.shape}")
        m = m & vm
    if m.sum() == 0:
        return float("nan")
    b_safe = np.nan_to_num(b, nan=0.0, posinf=0.0, neginf=0.0)
    return float(np.mean(np.abs(a[m] - b_safe[m])))


def mae_torch(
    gt: torch.Tensor,
    rec: torch.Tensor,
    valid_mask: Optional[torch.Tensor] = None,
) -> float:
    """NaN-safe MAE for (C,H,W) or (1,C,H,W) tensors."""
    gt  = _ensure_chw(gt,  "gt")
    rec = _ensure_chw(rec, "rec")
    if gt.shape != rec.shape:
        raise ValueError(f"Shape mismatch: {gt.shape} vs {rec.shape}")
    valid = torch.isfinite(gt)
    if valid_mask is not None:
        valid = valid & _ensure_chw(valid_mask, "valid_mask").bool()
    if not valid.any():
        return float("nan")
    rec_safe = torch.nan_to_num(rec, nan=0.0, posinf=0.0, neginf=0.0)
    return float((gt - rec_safe).abs()[valid].mean().item())


# ===========================================================================
# Gradient RMSE  (and normalize-by-field-magnitude variant)
# ===========================================================================

def gradient_rmse(
    gt: torch.Tensor,
    rec: torch.Tensor,
    axis: int = 1,
    valid_mask: Optional[torch.Tensor] = None,
) -> float:
    """RMSE of the first-order finite-difference gradient along ``axis`` (H by default).

    Computes ``sqrt(mean((diff(gt, axis) - diff(rec, axis))^2))`` over all
    positions where both adjacent GT values are finite.

    Parameters
    ----------
    gt, rec : (C, H, W) or (1, C, H, W) tensors in physical units.
    axis    : axis along which to differentiate (default 1 = H / depth).
    valid_mask : optional boolean mask (same shape as gt) for additional filtering.
    """
    gt  = _ensure_chw(gt,  "gt").to(torch.float64)
    rec = _ensure_chw(rec, "rec").to(torch.float64)
    if gt.shape != rec.shape:
        raise ValueError(f"Shape mismatch: {gt.shape} vs {rec.shape}")

    dgt  = torch.diff(gt,  dim=axis)
    drec = torch.diff(rec, dim=axis)

    # valid: both neighbouring GT values finite
    valid = torch.isfinite(dgt)
    if valid_mask is not None:
        vm = _ensure_chw(valid_mask, "valid_mask").bool()
        # gradient at h uses positions h and h+1
        slc_left  = [slice(None)] * vm.ndim; slc_left[axis]  = slice(None, -1)
        slc_right = [slice(None)] * vm.ndim; slc_right[axis] = slice(1, None)
        valid = valid & vm[tuple(slc_left)] & vm[tuple(slc_right)]

    err2 = (dgt - drec) ** 2
    masked = err2[valid]
    if masked.numel() == 0:
        return float("nan")
    return float(masked.mean().sqrt().item())


def gradient_rmse_z(
    gt: torch.Tensor,
    rec: torch.Tensor,
    z_coords: torch.Tensor,
    axis: int = 1,
    valid_mask: Optional[torch.Tensor] = None,
) -> float:
    """RMSE of the physical gradient dfield/dz using real z-coordinate spacing.

    Computes the actual gradient by dividing field finite-differences by the
    true physical spacing ``Δz = diff(z_coords)`` along ``axis``:

        grad_gt [h]  = (gt[h+1]  - gt[h])  / (z_coords[h+1] - z_coords[h])
        grad_rec[h]  = (rec[h+1] - rec[h]) / (z_coords[h+1] - z_coords[h])

    then returns ``sqrt(mean((grad_gt - grad_rec)^2))``.

    The result has units of **[field_unit / z_unit]** and measures how
    accurately the reconstruction preserves the actual vertical gradient of
    the field.

    Parameters
    ----------
    gt, rec   : (C, H, W) or (1, C, H, W) tensors in field units.
    z_coords  : 1-D tensor of length H with the physical z value at each
                depth level (e.g. metres, pressure, …).  Must be monotone
                so that all Δz have the same sign.
    axis      : axis along which to differentiate (default 1 = H / depth).
    valid_mask: optional boolean mask (same shape as gt).
    """
    gt  = _ensure_chw(gt,  "gt").to(torch.float64)
    rec = _ensure_chw(rec, "rec").to(torch.float64)
    if gt.shape != rec.shape:
        raise ValueError(f"Shape mismatch: {gt.shape} vs {rec.shape}")

    z = z_coords.to(torch.float64).reshape(-1)
    H = gt.shape[axis]
    if z.numel() != H:
        raise ValueError(
            f"z_coords length {z.numel()} must match the size of axis {axis} ({H})"
        )

    dz = torch.diff(z)          # (H-1,)
    if (dz == 0).any():
        raise ValueError("z_coords must be strictly monotone (no repeated values)")

    dgt  = torch.diff(gt,  dim=axis)
    drec = torch.diff(rec, dim=axis)

    # Broadcast dz to (C, H-1, W) for axis=1 (generalised for any axis)
    shape = [1] * gt.ndim
    shape[axis] = H - 1
    dz_b = dz.reshape(shape)        # broadcastable divisor

    grad_gt  = dgt  / dz_b
    grad_rec = drec / dz_b

    slc_left  = [slice(None)] * gt.ndim; slc_left[axis]  = slice(None, -1)
    slc_right = [slice(None)] * gt.ndim; slc_right[axis] = slice(1, None)

    valid = torch.isfinite(grad_gt)
    if valid_mask is not None:
        vm = _ensure_chw(valid_mask, "valid_mask").bool()
        valid = valid & vm[tuple(slc_left)] & vm[tuple(slc_right)]

    err2 = (grad_gt - grad_rec) ** 2
    masked = err2[valid]
    if masked.numel() == 0:
        return float("nan")
    return float(masked.mean().sqrt().item())


# ===========================================================================
# R² and Pearson correlation
# ===========================================================================

def r2_score(
    gt: torch.Tensor,
    rec: torch.Tensor,
    valid_mask: Optional[torch.Tensor] = None,
) -> float:
    """Coefficient of determination R² = 1 − SS_res / SS_tot.

    Computed globally over all valid pixels in the (C, H, W) tensor.
    Returns 1.0 for a perfect prediction and can be negative for very bad ones.
    """
    gt  = _ensure_chw(gt,  "gt").to(torch.float64)
    rec = _ensure_chw(rec, "rec").to(torch.float64)
    if gt.shape != rec.shape:
        raise ValueError(f"Shape mismatch: {gt.shape} vs {rec.shape}")

    valid = torch.isfinite(gt)
    if valid_mask is not None:
        valid = valid & _ensure_chw(valid_mask, "valid_mask").bool()
    if not valid.any():
        return float("nan")

    gt_v   = gt[valid]
    rec_v  = torch.nan_to_num(rec, nan=0.0, posinf=0.0, neginf=0.0)[valid]
    gt_mean = gt_v.mean()
    ss_tot = ((gt_v - gt_mean) ** 2).sum()
    ss_res = ((gt_v - rec_v) ** 2).sum()
    if ss_tot == 0:
        return 1.0 if float(ss_res.item()) == 0.0 else float("nan")
    return float((1.0 - ss_res / ss_tot).item())


def pearson_coeff(
    gt: torch.Tensor,
    rec: torch.Tensor,
    valid_mask: Optional[torch.Tensor] = None,
) -> float:
    """Pearson correlation coefficient between GT and reconstruction.

    Computed globally over all valid pixels in the (C, H, W) tensor.
    Returns a value in [-1, 1].
    """
    gt  = _ensure_chw(gt,  "gt").to(torch.float64)
    rec = _ensure_chw(rec, "rec").to(torch.float64)
    if gt.shape != rec.shape:
        raise ValueError(f"Shape mismatch: {gt.shape} vs {rec.shape}")

    valid = torch.isfinite(gt) & torch.isfinite(rec)
    if valid_mask is not None:
        valid = valid & _ensure_chw(valid_mask, "valid_mask").bool()
    if not valid.any():
        return float("nan")

    gt_v  = gt[valid]
    rec_v = rec[valid]
    mu_gt  = gt_v.mean()
    mu_rec = rec_v.mean()
    dgt  = gt_v  - mu_gt
    drec = rec_v - mu_rec
    cov  = (dgt * drec).mean()
    std_gt  = dgt.pow(2).mean().sqrt()
    std_rec = drec.pow(2).mean().sqrt()
    denom = std_gt * std_rec
    if denom == 0:
        return float("nan")
    return float((cov / denom).clamp(-1.0, 1.0).item())


# ===========================================================================
# PSD-based spatial score
# ===========================================================================

def psd_score_spatial(
    gt: torch.Tensor,
    rec: torch.Tensor,
    dx: float = 1.0,
    valid_mask: Optional[torch.Tensor] = None,
    level: float = 0.5,
    detrend: bool = True,
    apply_window: bool = True,
) -> Dict[str, float]:
    """PSD-based spatial score along the W (horizontal) axis.

    Adapts the ocean-data PSD leaderboard metric to the 2-D image setting.
    For each (C, H) profile along W:

      1. Optional mean-removal (detrend) and Hann windowing.
      2. 1-D rfft → one-sided power spectrum.

    All profiles are averaged to yield ``PSD_signal`` (GT) and ``PSD_err``
    (GT − REC).  Then::

        psd_score(f) = 1 − PSD_err(f) / PSD_signal(f)

    This mirrors ``psd_based_scores`` from the SSH leaderboard but operates
    on a single spatial axis instead of (time, lon).

    The *resolved spatial scale* is the smallest wavelength (in ``dx`` units)
    at which ``psd_score ≥ level``, i.e. the highest spatial frequency that
    is still resolved at the requested accuracy.

    Parameters
    ----------
    gt, rec      : (C, H, W) or (1, C, H, W) tensors.
    dx           : physical pixel spacing along W (e.g. km, °lon). Default 1.0.
    valid_mask   : optional boolean mask; profiles with any invalid pixel
                   along W are skipped entirely.
    level        : PSD score threshold for resolved-scale computation (default 0.5).
    detrend      : subtract per-profile mean before FFT (recommended).
    apply_window : apply a Hann window before FFT (recommended).

    Returns
    -------
    dict with keys:

    * ``score_global``   — ``1 − sqrt(PSD_err_total / PSD_signal_total)``,
                           a global normalised-RMSE in the spectral domain.
    * ``resolved_scale`` — smallest spatial wavelength (``dx`` units) at which
                           ``psd_score ≥ level``.  ``nan`` if never reached.
    * ``psd_score_mean`` — mean of ``psd_score`` over all positive frequencies.
    * ``n_profiles``     — number of (C, H) profiles used.
    """
    gt  = _ensure_chw(gt,  "gt").to(torch.float64)
    rec = _ensure_chw(rec, "rec").to(torch.float64)
    if gt.shape != rec.shape:
        raise ValueError(f"Shape mismatch: {gt.shape} vs {rec.shape}")

    C, H, W = gt.shape
    gt_np  = gt.detach().cpu().numpy()
    rec_np = rec.detach().cpu().numpy()

    if valid_mask is not None:
        vm_np = _ensure_chw(valid_mask, "valid_mask").bool().detach().cpu().numpy()
    else:
        vm_np = np.ones((C, H, W), dtype=bool)

    # Hann window (pre-computed, normalised so total power is preserved)
    hann      = np.hanning(W)
    hann_norm = float(np.sqrt(np.mean(hann ** 2)))
    if hann_norm == 0.0:
        hann_norm = 1.0

    n_freqs = W // 2 + 1
    psd_signal_acc = np.zeros(n_freqs, dtype=np.float64)
    psd_err_acc    = np.zeros(n_freqs, dtype=np.float64)
    n_profiles     = 0

    for c in range(C):
        for h in range(H):
            finite_row = np.isfinite(gt_np[c, h, :]) & np.isfinite(rec_np[c, h, :])
            valid_row  = vm_np[c, h, :] & finite_row
            if not valid_row.all() or int(valid_row.sum()) < 4:
                continue

            sig = gt_np[c, h, :].copy()
            err = sig - rec_np[c, h, :]

            if detrend:
                sig -= sig.mean()
                err -= err.mean()

            if apply_window:
                sig = sig * hann / hann_norm
                err = err * hann / hann_norm

            # One-sided power spectrum via rfft; normalise by W so power is
            # independent of sequence length.
            psd_signal_acc += np.abs(np.fft.rfft(sig)) ** 2 / W
            psd_err_acc    += np.abs(np.fft.rfft(err)) ** 2 / W
            n_profiles += 1

    if n_profiles == 0:
        return {
            "score_global":   float("nan"),
            "resolved_scale": float("nan"),
            "psd_score_mean": float("nan"),
            "n_profiles":     0,
        }

    psd_sig = psd_signal_acc / n_profiles
    psd_err = psd_err_acc    / n_profiles

    # Global score: 1 − sqrt(total_err_power / total_signal_power)
    total_sig = float(psd_sig.sum())
    total_err = float(psd_err.sum())
    score_global = (
        float(1.0 - np.sqrt(total_err / total_sig))
        if total_sig > 0.0 else float("nan")
    )

    # Per-frequency score on positive frequencies (skip DC bin at index 0)
    freqs     = np.fft.rfftfreq(W, d=float(dx))
    pos_mask  = freqs > 0.0
    freqs_pos = freqs[pos_mask]
    psd_sig_p = psd_sig[pos_mask]
    psd_err_p = psd_err[pos_mask]

    with np.errstate(divide="ignore", invalid="ignore"):
        psd_score = np.where(psd_sig_p > 0.0, 1.0 - psd_err_p / psd_sig_p, np.nan)

    psd_score_mean = float(np.nanmean(psd_score))

    # Resolved scale: 1/freq of the highest freq where psd_score >= level.
    # Frequencies are ascending → wavelengths are descending.
    above = np.where(np.isfinite(psd_score) & (psd_score >= float(level)))[0]
    resolved_scale = float(1.0 / freqs_pos[above[-1]]) if above.size > 0 else float("nan")

    return {
        "score_global":   float(score_global),
        "resolved_scale": float(resolved_scale),
        "psd_score_mean": float(psd_score_mean),
        "n_profiles":     int(n_profiles),
    }


# ===========================================================================
# Prominence-weighted matched 2D extremum distance + count penalty
# ===========================================================================

def _interp_nan_1d_np(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    idx = np.arange(x.size)
    ok  = np.isfinite(x)
    if ok.sum() == 0:
        return np.zeros_like(x)
    if ok.sum() == 1:
        return np.full_like(x, x[ok][0])
    return np.interp(idx, idx[ok], x[ok])


def detect_extrema_prominence(
    x: torch.Tensor,
    kind: str = "both",
    prominence: float = 0.5,
    min_distance: int = 5,
) -> torch.Tensor:
    """Detect local extrema along H using scipy.signal.find_peaks and an absolute prominence."""
    if _scipy_signal is None:
        raise ImportError("scipy.signal is required for detect_extrema_prominence")
    if kind not in ("min", "max", "both"):
        raise ValueError(f"kind must be 'min', 'max', or 'both', got '{kind}'")

    x = _ensure_chw(x)
    C, H, W = x.shape
    arr = x.detach().cpu().numpy().astype(np.float64, copy=False)
    profiles = arr.transpose(0, 2, 1).reshape(C * W, H)
    valid_all = np.isfinite(profiles)
    out_np = np.zeros((C * W, H), dtype=np.float32)
    prom = max(1e-12, float(prominence))
    dist = max(1, int(min_distance))

    for n in range(C * W):
        valid = valid_all[n]
        if int(valid.sum()) < 3:
            continue
        profile = profiles[n]
        if kind in ("max", "both"):
            peaks, _ = _scipy_signal.find_peaks(
                np.where(valid, profile, -np.inf),
                prominence=prom,
                distance=dist,
            )
            out_np[n, peaks] = 1.0
        if kind in ("min", "both"):
            peaks, _ = _scipy_signal.find_peaks(
                np.where(valid, -profile, -np.inf),
                prominence=prom,
                distance=dist,
            )
            out_np[n, peaks] = 1.0

    out = out_np.reshape(C, W, H).transpose(0, 2, 1)
    return torch.from_numpy(np.ascontiguousarray(out)).to(device=x.device)


def prominence_weighted_extrema_distance_2d(
    gt: torch.Tensor,
    rec: torch.Tensor,
    valid_mask: Optional[torch.Tensor] = None,
    kind: str = "both",
    prominence_fraction: float = 0.1,
    min_distance_h: int = 3,
    match_radius: float = float("inf"),
    count_penalty_weight: float = 1.0,
) -> Dict[str, float]:
    """Prominence-weighted matched 2D extremum distance + count penalty.

    Detects 1D extrema along the depth (H) axis for every (C, W) profile
    using scipy prominence, then treats each extremum as a point in the 2D
    (H, W) spatial plane.  GT and REC extrema are matched globally across
    all profiles with a greedy algorithm that prioritises high GT prominence.
    The score combines:

    * **mean_matched_dist_2d** — prominence-weighted mean 2D Euclidean distance
      ``sqrt((Δh)² + (Δw)²)`` for matched pairs.
    * **count_penalty** — ``|n_rec - n_gt| / (n_gt + 1)`` — relative count error.
    * **score** — ``mean_matched_dist_2d + count_penalty_weight * count_penalty``

    Parameters
    ----------
    gt, rec : (C, H, W) or (1, C, H, W) tensors in physical units.
    valid_mask : optional boolean mask.
    kind : ``"both"``, ``"min"``, or ``"max"``.
    prominence_fraction : prominence threshold as fraction of per-profile IQR.
    min_distance_h : minimum peak separation along H.
    match_radius : maximum 2D Euclidean radius for matching (``inf`` = unlimited).
    count_penalty_weight : weight of the count penalty in the combined score.

    Returns
    -------
    dict with keys ``matched_dist_mean``, ``matched_dist_std``,
    ``prominence_weighted_dist``, ``count_penalty``, ``score``,
    ``n_gt``, ``n_rec``, ``n_matched``.
    """
    if _scipy_signal is None:
        raise ImportError("scipy.signal is required for prominence_weighted_extrema_distance_2d")

    gt  = _ensure_chw(gt,  "gt")
    rec = _ensure_chw(rec, "rec")
    if gt.shape != rec.shape:
        raise ValueError(f"Shape mismatch: {tuple(gt.shape)} vs {tuple(rec.shape)}")

    if valid_mask is None:
        valid_mask_t = torch.isfinite(gt)
    else:
        valid_mask_t = _ensure_chw(valid_mask, "valid_mask").bool() & torch.isfinite(gt)

    gt_np    = gt.detach().cpu().numpy().astype(np.float64, copy=False)
    rec_np   = rec.detach().cpu().numpy().astype(np.float64, copy=False)
    valid_np = valid_mask_t.detach().cpu().numpy()

    C, H, W = gt_np.shape

    # Collect all extrema in 2D (H, W) space
    def _collect_extrema(field_np: np.ndarray) -> np.ndarray:
        """Returns array of shape (N, 3): [h_pos, w_pos, prominence]."""
        records: List[Tuple[float, float, float]] = []
        for c in range(C):
            for w in range(W):
                profile = field_np[c, :, w].copy()
                ok = valid_np[c, :, w]
                if int(ok.sum()) < 3:
                    continue
                profile[~ok] = np.nan
                profile_filled = _interp_nan_1d_np(profile)
                std_v = float(np.nanstd(profile, ddof=0))
                prom_thr = max(1e-12, float(prominence_fraction) * max(std_v, 1e-12))

                if kind in ("max", "both"):
                    peaks, props = _scipy_signal.find_peaks(
                        profile_filled,
                        prominence=prom_thr,
                        distance=max(1, int(min_distance_h)),
                    )
                    peaks = peaks[ok[peaks]]
                    proms = np.asarray(props.get("prominences", []), dtype=np.float64)
                    # proms may need masking too
                    proms_m = proms[ok[np.asarray(_scipy_signal.find_peaks(
                        profile_filled,
                        prominence=prom_thr,
                        distance=max(1, int(min_distance_h)),
                    )[0]).astype(int)] if len(proms) != len(peaks) else np.ones(len(peaks), dtype=bool)]
                    for h_pos, prom_v in zip(peaks.tolist(), proms[:len(peaks)].tolist()):
                        records.append((float(h_pos), float(w), float(prom_v)))

                if kind in ("min", "both"):
                    peaks, props = _scipy_signal.find_peaks(
                        -profile_filled,
                        prominence=prom_thr,
                        distance=max(1, int(min_distance_h)),
                    )
                    peaks = peaks[ok[peaks]]
                    proms = np.asarray(props.get("prominences", []), dtype=np.float64)
                    for h_pos, prom_v in zip(peaks.tolist(), proms[:len(peaks)].tolist()):
                        records.append((float(h_pos), float(w), float(prom_v)))

        if not records:
            return np.empty((0, 3), dtype=np.float64)
        return np.array(records, dtype=np.float64)

    gt_pts  = _collect_extrema(gt_np)   # (N_gt,  3): h, w, prom
    rec_pts = _collect_extrema(rec_np)  # (N_rec, 3): h, w, prom

    n_gt  = int(gt_pts.shape[0])
    n_rec = int(rec_pts.shape[0])

    if n_gt == 0 and n_rec == 0:
        return {
            "matched_dist_mean": float("nan"),
            "matched_dist_std": float("nan"),
            "prominence_weighted_dist": float("nan"),
            "count_penalty": 0.0,
            "score": float("nan"),
            "n_gt": 0, "n_rec": 0, "n_matched": 0,
        }

    count_penalty = float(abs(n_rec - n_gt)) / float(n_gt + 1)

    if n_gt == 0 or n_rec == 0:
        score = count_penalty_weight * count_penalty
        return {
            "matched_dist_mean": float("nan"),
            "matched_dist_std": float("nan"),
            "prominence_weighted_dist": float("nan"),
            "count_penalty": float(count_penalty),
            "score": float(score),
            "n_gt": n_gt, "n_rec": n_rec, "n_matched": 0,
        }

    # Greedy matching: descending GT prominence, O(N_gt * N_rec) but typical N is small
    gt_h   = gt_pts[:, 0]
    gt_w   = gt_pts[:, 1]
    gt_p   = gt_pts[:, 2]
    rec_h  = rec_pts[:, 0]
    rec_w  = rec_pts[:, 1]
    rec_p  = rec_pts[:, 2]

    order = np.argsort(-gt_p)  # descending prominence
    used_rec = np.zeros(n_rec, dtype=bool)

    matched_dists:   List[float] = []
    matched_proms:   List[float] = []

    for gi in order.tolist():
        dh = rec_h - gt_h[gi]
        dw = rec_w - gt_w[gi]
        dist2d = np.sqrt(dh * dh + dw * dw)
        within = (~used_rec) & (dist2d <= float(match_radius))
        if not within.any():
            continue
        candidates = np.flatnonzero(within)
        ri = int(candidates[np.argmin(dist2d[candidates])])
        used_rec[ri] = True
        matched_dists.append(float(dist2d[ri]))
        matched_proms.append(float(gt_p[gi]))

    n_matched = len(matched_dists)
    if n_matched == 0:
        score = count_penalty_weight * count_penalty
        return {
            "matched_dist_mean": float("nan"),
            "matched_dist_std": float("nan"),
            "prominence_weighted_dist": float("nan"),
            "count_penalty": float(count_penalty),
            "score": float(score),
            "n_gt": n_gt, "n_rec": n_rec, "n_matched": 0,
        }

    dists_arr = np.array(matched_dists, dtype=np.float64)
    proms_arr = np.array(matched_proms, dtype=np.float64)
    prom_sum  = float(proms_arr.sum())

    prom_weighted_dist = (
        float((dists_arr * proms_arr).sum() / prom_sum)
        if prom_sum > 0 else float(dists_arr.mean())
    )

    score = prom_weighted_dist + count_penalty_weight * count_penalty

    return {
        "matched_dist_mean":       float(dists_arr.mean()),
        "matched_dist_std":        float(dists_arr.std(ddof=0)),
        "prominence_weighted_dist": float(prom_weighted_dist),
        "count_penalty":           float(count_penalty),
        "score":                   float(score),
        "n_gt":                    int(n_gt),
        "n_rec":                   int(n_rec),
        "n_matched":               int(n_matched),
    }


 
# ===========================================================================
# Metrics report writer
# ===========================================================================
 
def write_metrics_txt(
    path: str,
    rows: List[Dict[str, Any]],
    header: Optional[Dict[str, Any]] = None,
    exclude_prefixes: Optional[Tuple[str, ...]] = None,
    exclude_keys: Optional[Tuple[str, ...]] = None,
) -> None:
    """Write a readable .txt report with aggregate stats + per-image CSV table."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
 
    if not rows:
        with path.open("w") as f:
            f.write("# Metrics report\n\n(no rows)\n")
        return
 
    exclude_prefixes = tuple(exclude_prefixes or ())
    exclude_keys_set = set(exclude_keys or ())

    def _include_key(k: str) -> bool:
        return k not in exclude_keys_set and not any(k.startswith(prefix) for prefix in exclude_prefixes)

    keys = [k for k in rows[0].keys() if _include_key(k)]
 
    def _agg(vals: List[float]) -> Tuple[float, float, float, float]:
        arr = np.asarray(vals, dtype=np.float64)
        arr = arr[np.isfinite(arr)]
        if arr.size == 0:
            return (np.nan, np.nan, np.nan, np.nan)
        return (float(arr.mean()), float(arr.std(ddof=0)), float(arr.min()), float(arr.max()))

    def _mse_to_rmse_aggregate_key_vals(k: str, vals: List[float]) -> Tuple[str, List[float]]:
        # Aggregate report convention: present MSE-like scalars as RMSE-like scalars.
        is_mse_like = ("MSE" in k) and ("MSE_over_STD" not in k)
        if not is_mse_like:
            return k, vals
        arr = np.asarray(vals, dtype=np.float64)
        arr = np.where(arr >= 0.0, np.sqrt(arr), np.nan)
        return k.replace("MSE", "RMSE"), arr.tolist()

    def _is_mse_like_key(k: str) -> bool:
        return ("MSE" in k) and ("MSE_over_STD" not in k)

    def _convert_mse_like_value(v: Any) -> Any:
        if isinstance(v, (int, float, np.floating, np.integer)):
            vf = float(v)
            return float(np.sqrt(vf)) if np.isfinite(vf) and vf >= 0.0 else float("nan")
        arr = _parse_numeric_list(v)
        if arr is not None:
            arr = np.where(arr >= 0.0, np.sqrt(arr), np.nan)
            return json.dumps([float(x) if np.isfinite(x) else float("nan") for x in arr.tolist()])
        try:
            vf = float(v)
            return float(np.sqrt(vf)) if np.isfinite(vf) and vf >= 0.0 else float("nan")
        except Exception:
            return v
 
    def _parse_numeric_list(v: Any) -> Optional[np.ndarray]:
        if v is None:
            return None
        if isinstance(v, np.ndarray):
            return v.astype(np.float64, copy=False).reshape(-1)
        if isinstance(v, (list, tuple)):
            try:
                return np.asarray(v, dtype=np.float64).reshape(-1)
            except Exception:
                return None
        if isinstance(v, str):
            s = v.strip()
            if not s:
                return None
            try:
                parsed = json.loads(s)
            except Exception:
                return None
            if isinstance(parsed, (list, tuple)):
                try:
                    return np.asarray(parsed, dtype=np.float64).reshape(-1)
                except Exception:
                    return None
        return None
 
    def _parse_confusion_list(v: Any) -> Optional[List[Dict[str, int]]]:
        parsed = v
        if isinstance(v, str):
            try:
                parsed = json.loads(v.strip())
            except Exception:
                return None
        if not isinstance(parsed, (list, tuple)):
            return None
        out: List[Dict[str, int]] = []
        for item in parsed:
            if not isinstance(item, dict):
                return None
            try:
                out.append({
                    "tp": int(item.get("tp", 0)), "fp": int(item.get("fp", 0)),
                    "fn": int(item.get("fn", 0)), "tn": int(item.get("tn", 0)),
                })
            except Exception:
                return None
        return out
 
    with path.open("w") as f:
        f.write("# Compression metrics report\n\n")
        if header:
            f.write("## Run info\n")
            for k, v in header.items():
                f.write(f"- {k}: {v}\n")
            f.write("\n")
 
        # --- aggregate scalars ---
        f.write("## Aggregate statistics\n")
        for k in keys:
            if k in ("i", "img_name"):
                continue
            try:
                vals = [float(r[k]) for r in rows]
            except Exception:
                continue
            out_key, out_vals = _mse_to_rmse_aggregate_key_vals(k, vals)
            mean, std, vmin, vmax = _agg(out_vals)
            f.write(f"{out_key}: mean={mean:.6g}, std={std:.6g}, min={vmin:.6g}, max={vmax:.6g}\n")
 
        # --- per-level helpers ---
        def _write_level_stats(
            level_keys: List[str],
            section_title: str,
            level_label: str = "H",
            write_mean_vector: bool = False,
        ) -> None:
            if not level_keys:
                return
            f.write(f"\n## {section_title}\n")
            for k in level_keys:
                per_level_values: List[List[float]] = []
                for row in rows:
                    arr = _parse_numeric_list(row.get(k))
                    if arr is None:
                        continue
                    while len(per_level_values) < int(arr.size):
                        per_level_values.append([])
                    for idx, val in enumerate(arr.tolist()):
                        if np.isfinite(val):
                            per_level_values[idx].append(float(val))
                if not per_level_values:
                    continue
                f.write(f"{k}\n")
                if write_mean_vector:
                    mean_vector = [
                        float(np.mean(vals)) if vals else float("nan")
                        for vals in per_level_values
                    ]
                    f.write(f"  full_mean_per_{level_label}: {json.dumps(mean_vector)}\n")
                for idx, vals in enumerate(per_level_values):
                    if not vals:
                        f.write(f"  {level_label}_{idx}: mean=nan, std=nan, min=nan, max=nan, n=0\n")
                        continue
                    mean, std, vmin, vmax = _agg(vals)
                    f.write(
                        f"  {level_label}_{idx}: mean={mean:.6g}, std={std:.6g}, "
                        f"min={vmin:.6g}, max={vmax:.6g}, n={len(vals)}\n"
                    )
 
        def _write_confusion_level_stats(
            cm_keys: List[str],
            section_title: str,
            level_label: str = "h",
        ) -> None:
            if not cm_keys:
                return
            f.write(f"\n## {section_title}\n")
            for k in cm_keys:
                per_level_cm: List[Dict[str, int]] = []
                for row in rows:
                    cm_list = _parse_confusion_list(row.get(k))
                    if cm_list is None:
                        continue
                    while len(per_level_cm) < len(cm_list):
                        per_level_cm.append({"tp": 0, "fp": 0, "fn": 0, "tn": 0})
                    for idx, cm in enumerate(cm_list):
                        per_level_cm[idx]["tp"] += cm["tp"]
                        per_level_cm[idx]["fp"] += cm["fp"]
                        per_level_cm[idx]["fn"] += cm["fn"]
                        per_level_cm[idx]["tn"] += cm["tn"]
                if not per_level_cm:
                    continue
                f.write(f"{k}\n")
                f.write(f"  full_sum_per_{level_label}: {json.dumps(per_level_cm)}\n")
                for idx, cm in enumerate(per_level_cm):
                    tp, fp, fn, tn = cm["tp"], cm["fp"], cm["fn"], cm["tn"]
                    precision = tp / (tp + fp + 1e-12)
                    recall    = tp / (tp + fn + 1e-12)
                    f1 = 2.0 * precision * recall / (precision + recall + 1e-12)
                    f.write(
                        f"  {level_label}_{idx}: tp={tp}, fp={fp}, fn={fn}, tn={tn}, "
                        f"precision={precision:.6g}, recall={recall:.6g}, f1={f1:.6g}\n"
                    )
 
        def _write_extrema_count_stats(
            gt_keys_rec_keys_kind: List[Tuple[str, str, str]],
        ) -> None:
            """
            Per-level GT/REC extrema counts.
            Flags levels with zero GT extrema — these contribute F1=0 to the
            unweighted mean (F1_*) but are excluded from the weighted mean (F1_*_weighted).
            """
            for gt_key, rec_key, kind in gt_keys_rec_keys_kind:
                if gt_key not in keys:
                    continue
                f.write(f"\n## Extrema counts per H level ({kind})\n")
 
                per_level_gt:  List[List[int]] = []
                per_level_rec: List[List[int]] = []
 
                for row in rows:
                    gt_arr  = _parse_numeric_list(row.get(gt_key))
                    rec_arr = _parse_numeric_list(row.get(rec_key))
                    if gt_arr is None or rec_arr is None:
                        continue
                    n = int(gt_arr.size)
                    while len(per_level_gt) < n:
                        per_level_gt.append([])
                        per_level_rec.append([])
                    for idx in range(n):
                        per_level_gt[idx].append(int(gt_arr[idx]))
                        per_level_rec[idx].append(int(rec_arr[idx]))
 
                if not per_level_gt:
                    f.write("  (no data)\n")
                    continue
 
                n_empty_gt  = sum(1 for vals in per_level_gt  if sum(vals) == 0)
                n_empty_rec = sum(1 for vals in per_level_rec if sum(vals) == 0)
                f.write(f"  levels_with_zero_gt_extrema:  {n_empty_gt} / {len(per_level_gt)}\n")
                f.write(f"  levels_with_zero_rec_extrema: {n_empty_rec} / {len(per_level_rec)}\n")
                f.write(
                    f"  NOTE: F1_{kind} (unweighted) treats empty levels as F1=0. "
                    f"F1_{kind}_weighted excludes them.\n"
                )
 
                for idx, (gt_vals, rec_vals) in enumerate(zip(per_level_gt, per_level_rec)):
                    gt_total  = sum(gt_vals)
                    rec_total = sum(rec_vals)
                    gt_mean   = float(gt_total) / len(gt_vals)  if gt_vals  else float("nan")
                    rec_mean  = float(rec_total) / len(rec_vals) if rec_vals else float("nan")
                    flag = "  *** NO GT EXTREMA — excluded from weighted F1 ***" if gt_total == 0 else ""
                    f.write(
                        f"  h_{idx}: gt_total={gt_total}, gt_mean={gt_mean:.2f}, "
                        f"rec_total={rec_total}, rec_mean={rec_mean:.2f}{flag}\n"
                    )
 
        # --- call all section writers ---
        _write_level_stats(
            [k for k in keys if k.startswith("F1_") and k.endswith("_levels")],
            section_title="F1 per H level", level_label="h",
        )
        _write_confusion_level_stats(
            [k for k in keys if k.startswith("CM_") and k.endswith("_levels")],
            section_title="F1 confusion matrix per H level", level_label="h",
        )
        _write_extrema_count_stats([
            ("GT_extrema_min_levels",  "REC_extrema_min_levels",  "min"),
            ("GT_extrema_max_levels",  "REC_extrema_max_levels",  "max"),
            ("GT_extrema_both_levels", "REC_extrema_both_levels", "both"),
        ])
        _write_level_stats(
            [k for k in keys if k in ("MSE_levels", "RMSE_levels")],
            section_title="RMSE per H level", level_label="h", write_mean_vector=True,
        )
        _write_level_stats(
            [k for k in keys if k == "MSE_over_STD_levels"],
            section_title="MSE/STD(GT) per H level", level_label="h",
        )
 
        # --- per-image CSV ---
        f.write("\n## Per-image (CSV)\n")
        keys_out = [k.replace("MSE", "RMSE") if _is_mse_like_key(k) else k for k in keys]
        f.write(",".join(keys_out) + "\n")
        for r in rows:
            out_vals: List[str] = []
            for k in keys:
                v = r.get(k, "")
                if _is_mse_like_key(k):
                    v = _convert_mse_like_value(v)
                out_vals.append(str(v))
            f.write(",".join(out_vals) + "\n")