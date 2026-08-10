
"""
Extrema-focused metrics v2 for MLIC++ profile reconstruction.

Cleaned and optimized rewrite.

Key fixes / changes:
- removes duplicate helper definitions and dead code;
- vectorizes the core gradient-sign detectors;
- vectorizes the tolerant mask-based F1 aggregation;
- fixes the confusing tolerant "tp" reporting by exposing both
  tp_precision_view and tp_recall_view and using a conservative tp=min(...);
- vectorizes v2 side-support filtering using prefix sums;
- keeps all public metrics from the mixed version above.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F

try:
    import scipy.signal as _scipy_signal
except Exception:
    _scipy_signal = None

try:
    from scipy.stats import wasserstein_distance as _wasserstein_distance
except Exception:
    _wasserstein_distance = None

TensorLike = Union[torch.Tensor, np.ndarray]


# ============================================================================
# Basic helpers
# ============================================================================

def _ensure_chw(x: torch.Tensor, name: str = "x") -> torch.Tensor:
    """Accept (C,H,W) or (1,C,H,W); always return (C,H,W)."""
    if not isinstance(x, torch.Tensor):
        raise TypeError(f"{name}: expected torch.Tensor, got {type(x)!r}")
    if x.ndim == 4:
        if x.shape[0] != 1:
            raise ValueError(f"{name}: batch size must be 1, got {x.shape[0]}")
        x = x.squeeze(0)
    if x.ndim != 3:
        raise ValueError(f"{name}: expected (C,H,W) or (1,C,H,W), got shape {tuple(x.shape)}")
    return x


def _nan_mask(x: torch.Tensor) -> torch.Tensor:
    return torch.isfinite(x)


def apply_gt_nan_mask_torch(
    gt: torch.Tensor,
    pred: torch.Tensor,
    valid_mask: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Mask from finite GT values, optionally intersected with valid_mask."""
    gt = _ensure_chw(gt, "gt")
    pred = _ensure_chw(pred, "pred")
    if gt.shape != pred.shape:
        raise ValueError(f"Shape mismatch: {tuple(gt.shape)} vs {tuple(pred.shape)}")

    combined_valid = torch.isfinite(gt)
    if valid_mask is not None:
        valid_mask = _ensure_chw(valid_mask, "valid_mask")
        if valid_mask.shape != gt.shape:
            raise ValueError(f"valid_mask shape mismatch: {tuple(valid_mask.shape)} vs {tuple(gt.shape)}")
        combined_valid = combined_valid & valid_mask.to(dtype=torch.bool, device=gt.device)

    gt_masked = torch.where(combined_valid, gt, torch.zeros_like(gt))
    pred_finite = torch.nan_to_num(pred, nan=0.0, posinf=0.0, neginf=0.0)
    pred_masked = torch.where(combined_valid, pred_finite, torch.zeros_like(pred_finite))
    return gt_masked, pred_masked, combined_valid


def _safe_div(num: float, den: float) -> float:
    return float(num) / float(den) if den > 0 else 0.0


def _f1_from_precision_recall(precision: float, recall: float) -> float:
    den = precision + recall
    return 2.0 * precision * recall / den if den > 0 else 0.0


# ============================================================================
# Extrema detector internals
# ============================================================================

def _auto_grad_eps(x: torch.Tensor) -> float:
    x = _ensure_chw(x)
    grad_all = torch.diff(x, dim=1)
    abs_grad = grad_all[torch.isfinite(grad_all)].abs()
    if abs_grad.numel() == 0:
        return 0.0
    return float(torch.quantile(abs_grad, 0.1).item()) * 0.5


def _smooth_nan_along_h(x: torch.Tensor, window: int) -> torch.Tensor:
    """NaN-aware moving average smoothing along H for a full (C,H,W) tensor."""
    if window <= 1:
        return x
    x = _ensure_chw(x)
    C, H, W = x.shape
    flat = x.permute(0, 2, 1).reshape(C * W, 1, H)
    kernel = torch.ones(1, 1, window, dtype=flat.dtype, device=flat.device)
    valid = torch.isfinite(flat).to(flat.dtype)
    flat0 = torch.where(torch.isfinite(flat), flat, torch.zeros_like(flat))
    num = F.conv1d(flat0, kernel, padding=window // 2)
    den = F.conv1d(valid, kernel, padding=window // 2)
    if num.shape[-1] != H:
        num = num[..., :H]
        den = den[..., :H]
    out = torch.full_like(flat, torch.nan)
    ok = den > 0
    out[ok] = num[ok] / den[ok]
    return out.reshape(C, W, H).permute(0, 2, 1)


def _fill_zero_signs_lastdim(s: torch.Tensor) -> torch.Tensor:
    """Forward/backward fill zeros along the last dimension in bulk."""
    if s.numel() == 0:
        return s
    orig_shape = s.shape
    L = orig_shape[-1]
    flat = s.reshape(-1, L)

    idx = torch.arange(L, device=flat.device, dtype=torch.long).view(1, L).expand(flat.shape[0], -1)
    nonzero = flat != 0

    last_idx = torch.where(nonzero, idx, torch.zeros_like(idx))
    last_idx = torch.cummax(last_idx, dim=-1).values
    out = torch.gather(flat, -1, last_idx)

    needs_backfill = out == 0
    rev = torch.flip(out, dims=[-1])
    rev_nonzero = rev != 0
    rev_idx = torch.where(rev_nonzero, idx, torch.zeros_like(idx))
    rev_idx = torch.cummax(rev_idx, dim=-1).values
    rev_filled = torch.gather(rev, -1, rev_idx)
    backfilled = torch.flip(rev_filled, dims=[-1])
    out = torch.where(needs_backfill, backfilled, out)
    return out.reshape(orig_shape)


def _suppress_close_extrema_lastdim(mask: torch.Tensor, min_separation: int) -> torch.Tensor:
    """
    Greedy left-to-right suppression along the last dimension, vectorized across rows.
    """
    if min_separation <= 1:
        return mask
    orig_shape = mask.shape
    L = orig_shape[-1]
    flat = mask.reshape(-1, L).bool().clone()
    last_kept = torch.full((flat.shape[0],), -(10**9), dtype=torch.long, device=flat.device)
    for k in range(L):
        active = flat[:, k]
        keep = active & ((k - last_kept) >= min_separation)
        flat[:, k] = keep
        last_kept = torch.where(keep, torch.full_like(last_kept, k), last_kept)
    return flat.reshape(orig_shape)


def _prepare_grad_sign_turning(
    x: torch.Tensor,
    grad_eps: Union[float, str] = "auto",
    smooth_window: int = 1,
    fill_zero_signs: bool = True,
    grad_eps_reference: Optional[torch.Tensor] = None,
    work_dtype: torch.dtype = torch.float32,
) -> Tuple[torch.Tensor, torch.Tensor, float]:
    """
    Return (grad, turning, eps) for extrema detection on (C,H,W).
    grad shape:    (C,H-1,W)
    turning shape: (C,H-2,W)
    """
    x = _ensure_chw(x)
    smooth_window = max(1, int(smooth_window))
    if smooth_window % 2 == 0:
        smooth_window += 1

    x_work = x.to(dtype=work_dtype)
    if smooth_window > 1:
        x_work = _smooth_nan_along_h(x_work, smooth_window)

    if grad_eps == "auto":
        ref = grad_eps_reference if grad_eps_reference is not None else x_work
        ref = _ensure_chw(ref, "grad_eps_reference").to(dtype=work_dtype)
        eps = _auto_grad_eps(ref)
    else:
        eps = float(grad_eps)

    grad = torch.diff(x_work, dim=1)
    if eps > 0:
        sign = torch.where(
            grad > eps,
            torch.ones_like(grad),
            torch.where(grad < -eps, -torch.ones_like(grad), torch.zeros_like(grad)),
        )
    else:
        sign = torch.sign(grad)

    if fill_zero_signs:
        sign = _fill_zero_signs_lastdim(sign.permute(0, 2, 1)).permute(0, 2, 1)

    turning = torch.diff(sign, dim=1)
    return grad, turning, eps


def _support_counts_for_candidates(
    grad_last: torch.Tensor,
    threshold: float,
    side_window: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Compute left/right counts for +/- thresholded gradients for all candidate positions.

    grad_last: (N, G) where G = H-1
    Returns:
        left_pos, right_pos, left_neg, right_neg of shape (N, H-2)
    where the candidate positions correspond to h = 1..H-2.
    """
    if grad_last.ndim != 2:
        raise ValueError(f"grad_last must be 2D, got shape {tuple(grad_last.shape)}")
    N, G = grad_last.shape
    if G < 2:
        empty = torch.zeros((N, 0), dtype=torch.int32, device=grad_last.device)
        return empty, empty, empty, empty

    side_window = max(1, int(side_window))
    h = torch.arange(1, G, device=grad_last.device, dtype=torch.long)  # 1..H-2

    left_start = torch.clamp(h - side_window, min=0)
    left_end = h
    right_start = h
    right_end = torch.clamp(h + side_window, max=G)

    pos_support = ((grad_last > threshold) & torch.isfinite(grad_last)).to(torch.int32)
    neg_support = ((grad_last < -threshold) & torch.isfinite(grad_last)).to(torch.int32)

    pos_prefix = F.pad(pos_support, (1, 0), value=0).cumsum(dim=-1)
    neg_prefix = F.pad(neg_support, (1, 0), value=0).cumsum(dim=-1)

    left_pos = pos_prefix[:, left_end] - pos_prefix[:, left_start]
    right_pos = pos_prefix[:, right_end] - pos_prefix[:, right_start]
    left_neg = neg_prefix[:, left_end] - neg_prefix[:, left_start]
    right_neg = neg_prefix[:, right_end] - neg_prefix[:, right_start]
    return left_pos, right_pos, left_neg, right_neg


def _interp_nan_1d(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    idx = np.arange(x.size)
    ok = np.isfinite(x)
    if ok.sum() == 0:
        return np.zeros_like(x)
    if ok.sum() == 1:
        return np.full_like(x, x[ok][0])
    return np.interp(idx, idx[ok], x[ok])


def _robust_profile_scale(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float64)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return 0.0
    q05, q95 = np.percentile(x, [5.0, 95.0])
    return float(max(0.0, q95 - q05))


# ============================================================================
# Gradient-sign extrema detectors
# ============================================================================

def detect_extrema(
    x: torch.Tensor,
    kind: str = "both",
    grad_eps: Union[float, str] = "auto",
    smooth_window: int = 1,
    min_separation: int = 3,
    fill_zero_signs: bool = True,
    grad_eps_reference: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Fast local-extrema detector along H of a (C,H,W) tensor."""
    if kind not in ("min", "max", "both"):
        raise ValueError(f"kind must be 'min', 'max', or 'both', got {kind!r}")

    x = _ensure_chw(x)
    C, H, W = x.shape
    if H < 3:
        return torch.zeros_like(x, dtype=torch.float32)

    _, turning, _ = _prepare_grad_sign_turning(
        x,
        grad_eps=grad_eps,
        smooth_window=smooth_window,
        fill_zero_signs=fill_zero_signs,
        grad_eps_reference=grad_eps_reference,
        work_dtype=torch.float32,
    )

    if kind == "min":
        core = turning > 0
    elif kind == "max":
        core = turning < 0
    else:
        core = turning != 0

    if min_separation > 1:
        core = _suppress_close_extrema_lastdim(core.permute(0, 2, 1), min_separation).permute(0, 2, 1)

    out = torch.zeros(C, H, W, dtype=torch.float32, device=x.device)
    out[:, 1:-1, :] = core.to(torch.float32)
    out = out * torch.isfinite(x).to(torch.float32)
    return out


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
    Stricter extrema detector with side-support checks.

    A minimum requires sufficiently negative gradients on the left and
    sufficiently positive gradients on the right. A maximum uses the converse.
    """
    if kind not in ("min", "max", "both"):
        raise ValueError(f"kind must be 'min', 'max', or 'both', got {kind!r}")

    x = _ensure_chw(x)
    C, H, W = x.shape
    if H < 3:
        return torch.zeros_like(x, dtype=torch.float32)

    side_window = max(1, int(side_window))
    min_support_count = max(1, int(min_support_count))

    grad, turning, eps = _prepare_grad_sign_turning(
        x,
        grad_eps=grad_eps,
        smooth_window=smooth_window,
        fill_zero_signs=fill_zero_signs,
        grad_eps_reference=grad_eps_reference,
        work_dtype=torch.float32,
    )

    core_min = turning > 0
    core_max = turning < 0

    # Flatten profiles to (N, G) and vectorize the support test.
    grad_last = grad.permute(0, 2, 1).reshape(-1, grad.shape[1])
    left_pos, right_pos, left_neg, right_neg = _support_counts_for_candidates(
        grad_last, threshold=eps, side_window=side_window
    )

    if kind in ("min", "both"):
        core_min_last = core_min.permute(0, 2, 1).reshape(-1, core_min.shape[1])
        keep_min = (left_neg >= min_support_count) & (right_pos >= min_support_count)
        core_min = (core_min_last & keep_min).reshape(C, W, H - 2).permute(0, 2, 1)

    if kind in ("max", "both"):
        core_max_last = core_max.permute(0, 2, 1).reshape(-1, core_max.shape[1])
        keep_max = (left_pos >= min_support_count) & (right_neg >= min_support_count)
        core_max = (core_max_last & keep_max).reshape(C, W, H - 2).permute(0, 2, 1)

    if kind == "min":
        core = core_min
    elif kind == "max":
        core = core_max
    else:
        core = core_min | core_max

    if min_separation > 1:
        core = _suppress_close_extrema_lastdim(core.permute(0, 2, 1), min_separation).permute(0, 2, 1)

    out = torch.zeros(C, H, W, dtype=torch.float32, device=x.device)
    out[:, 1:-1, :] = core.to(torch.float32)
    out = out * torch.isfinite(x).to(torch.float32)
    return out


def detect_extrema_all_v2(
    x: torch.Tensor,
    grad_eps: Union[float, str] = "auto",
    smooth_window: int = 1,
    min_separation: int = 3,
    fill_zero_signs: bool = True,
    grad_eps_reference: Optional[torch.Tensor] = None,
    side_window: int = 5,
    min_support_count: int = 1,
) -> Dict[str, torch.Tensor]:
    kwargs = dict(
        grad_eps=grad_eps,
        smooth_window=smooth_window,
        min_separation=min_separation,
        fill_zero_signs=fill_zero_signs,
        grad_eps_reference=grad_eps_reference,
        side_window=side_window,
        min_support_count=min_support_count,
    )
    return {
        "min": detect_extrema_v2(x, kind="min", **kwargs),
        "max": detect_extrema_v2(x, kind="max", **kwargs),
        "both": detect_extrema_v2(x, kind="both", **kwargs),
    }


# ============================================================================
# Prominence-based extrema detection
# ============================================================================

def detect_extrema_prominence(
    x: torch.Tensor,
    kind: str = "both",
    prominence_fraction: float = 0.1,
    min_width: int = 1,
    min_distance: int = 3,
) -> torch.Tensor:
    if _scipy_signal is None:
        raise ImportError("scipy.signal is required for detect_extrema_prominence")
    if kind not in ("min", "max", "both"):
        raise ValueError(f"kind must be 'min', 'max', or 'both', got {kind!r}")

    x = _ensure_chw(x)
    C, H, W = x.shape
    arr = x.detach().cpu().numpy().astype(np.float64, copy=False)
    profiles = arr.transpose(0, 2, 1).reshape(C * W, H)
    valid_all = np.isfinite(profiles)
    valid_count = valid_all.sum(axis=1)
    std_vals = np.nanstd(profiles, axis=1)
    proms = np.maximum(1e-12, float(prominence_fraction) * std_vals)
    out_np = np.zeros((C * W, H), dtype=np.float32)

    active = np.flatnonzero(valid_count >= 3)
    for n in active.tolist():
        profile = profiles[n]
        valid = valid_all[n]
        profile_filled = _interp_nan_1d(profile)
        prom = float(proms[n])

        if kind in ("max", "both"):
            peaks, _ = _scipy_signal.find_peaks(
                profile_filled,
                prominence=prom,
                width=max(1, int(min_width)),
                distance=max(1, int(min_distance)),
            )
            peaks = peaks[valid[peaks]]
            out_np[n, peaks] = 1.0

        if kind in ("min", "both"):
            peaks, _ = _scipy_signal.find_peaks(
                -profile_filled,
                prominence=prom,
                width=max(1, int(min_width)),
                distance=max(1, int(min_distance)),
            )
            peaks = peaks[valid[peaks]]
            out_np[n, peaks] = 1.0

    out = out_np.reshape(C, W, H).transpose(0, 2, 1)
    return torch.from_numpy(np.ascontiguousarray(out)).to(device=x.device)


# ============================================================================
# Wasserstein distance between extrema distributions
# ============================================================================

def extrema_wasserstein(
    gt_mask: torch.Tensor,
    rec_mask: torch.Tensor,
    valid: Optional[torch.Tensor] = None,
) -> Dict[str, Any]:
    if _wasserstein_distance is None:
        raise ImportError("scipy.stats.wasserstein_distance is required for extrema_wasserstein")

    gt_mask = _ensure_chw(gt_mask, "gt_mask")
    rec_mask = _ensure_chw(rec_mask, "rec_mask")
    if gt_mask.shape != rec_mask.shape:
        raise ValueError(f"Shape mismatch: {tuple(gt_mask.shape)} vs {tuple(rec_mask.shape)}")
    C, H, W = gt_mask.shape

    if valid is not None:
        valid = _ensure_chw(valid, "valid")
        gt_bin = (gt_mask > 0) & valid.bool()
        rec_bin = (rec_mask > 0) & valid.bool()
    else:
        gt_bin = gt_mask > 0
        rec_bin = rec_mask > 0

    gt_np = gt_bin.detach().cpu().numpy()
    rec_np = rec_bin.detach().cpu().numpy()
    h_indices = np.arange(H, dtype=np.float64)

    gt_profiles = gt_np.transpose(0, 2, 1).reshape(C * W, H)
    rec_profiles = rec_np.transpose(0, 2, 1).reshape(C * W, H)

    distances: List[float] = []
    per_h_contributions: List[List[float]] = [[] for _ in range(H)]

    gt_nonempty = gt_profiles.any(axis=1)
    rec_nonempty = rec_profiles.any(axis=1)
    active = np.flatnonzero(gt_nonempty & rec_nonempty)

    for n in active.tolist():
        gt_positions = h_indices[gt_profiles[n]]
        rec_positions = h_indices[rec_profiles[n]]
        d = float(_wasserstein_distance(gt_positions, rec_positions))
        distances.append(d)
        for h_idx in gt_positions.astype(int):
            per_h_contributions[h_idx].append(d)

    wasserstein_per_h = [float(np.mean(vals)) if vals else float("nan") for vals in per_h_contributions]
    return {
        "wasserstein_mean": float(np.mean(distances)) if distances else float("nan"),
        "wasserstein_std": float(np.std(distances)) if distances else float("nan"),
        "wasserstein_per_h": wasserstein_per_h,
        "n_profiles_used": int(len(distances)),
        "n_profiles_total": int(C * W),
    }


# ============================================================================
# Low-pass filter
# ============================================================================

def lowpass_filter_torch_along_axis(
    arr: torch.Tensor,
    axis: int = -1,
    order: int = 2,
    wn: float = 0.107,
) -> torch.Tensor:
    if _scipy_signal is None:
        return arr
    arr_np = arr.detach().cpu().numpy().astype(np.float64, copy=False)
    b, a = _scipy_signal.butter(N=int(order), Wn=float(wn), btype="low", analog=False)
    filtered_np = _scipy_signal.filtfilt(b, a, arr_np, axis=axis)
    return torch.from_numpy(np.ascontiguousarray(filtered_np)).to(device=arr.device, dtype=arr.dtype)


# ============================================================================
# Tolerant mask-based F1 (legacy but kept for comparison)
# ============================================================================

def _dilate_along_h(mask: torch.Tensor, kernel_size: int) -> torch.Tensor:
    mask = _ensure_chw(mask, "mask")
    C, H, W = mask.shape
    pad = kernel_size // 2
    kernel = torch.ones(1, 1, kernel_size, device=mask.device, dtype=torch.float32)
    flat = mask.permute(0, 2, 1).reshape(C * W, 1, H).to(torch.float32)
    expanded = F.conv1d(flat, kernel, padding=pad)
    if expanded.shape[-1] != H:
        expanded = expanded[..., :H]
    return (expanded > 0).reshape(C, W, H).permute(0, 2, 1)


def _build_tolerant_confusion_list(
    tp_precision_view: torch.Tensor,
    tp_recall_view: torch.Tensor,
    fp: torch.Tensor,
    fn: torch.Tensor,
    tn: torch.Tensor,
) -> List[Dict[str, int]]:
    out: List[Dict[str, int]] = []
    for tp_p, tp_r, fp_v, fn_v, tn_v in zip(
        tp_precision_view.detach().cpu().tolist(),
        tp_recall_view.detach().cpu().tolist(),
        fp.detach().cpu().tolist(),
        fn.detach().cpu().tolist(),
        tn.detach().cpu().tolist(),
    ):
        tp_p_i = int(tp_p)
        tp_r_i = int(tp_r)
        out.append(
            {
                "tp": int(min(tp_p_i, tp_r_i)),
                "tp_precision_view": tp_p_i,
                "tp_recall_view": tp_r_i,
                "fp": int(fp_v),
                "fn": int(fn_v),
                "tn": int(tn_v),
            }
        )
    return out


def _f1_from_masks(
    gt_mask: torch.Tensor,
    rec_mask: torch.Tensor,
    valid: torch.Tensor,
    kernel_size: int,
) -> Dict[str, Any]:
    if kernel_size < 1:
        raise ValueError(f"kernel_size must be >= 1, got {kernel_size}")

    gt_mask = _ensure_chw(gt_mask, "gt_mask")
    rec_mask = _ensure_chw(rec_mask, "rec_mask")
    valid = _ensure_chw(valid, "valid")

    gt_bin = (gt_mask > 0) & valid
    rec_bin = (rec_mask > 0) & valid
    gt_win = _dilate_along_h(gt_bin.float(), kernel_size)
    rec_win = _dilate_along_h(rec_bin.float(), kernel_size)

    reduce_dims = (0, 2)
    tp_prec_h = (gt_win & rec_bin).sum(dim=reduce_dims).to(torch.float64)
    tp_rec_h = (gt_bin & rec_win).sum(dim=reduce_dims).to(torch.float64)
    fp_h = ((~gt_win) & rec_bin & valid).sum(dim=reduce_dims).to(torch.float64)
    fn_h = (gt_bin & (~rec_win) & valid).sum(dim=reduce_dims).to(torch.float64)
    tn_h = (valid & (~gt_bin) & (~rec_bin)).sum(dim=reduce_dims).to(torch.float64)

    precision_h = torch.where(tp_prec_h + fp_h > 0, tp_prec_h / (tp_prec_h + fp_h), torch.zeros_like(tp_prec_h))
    recall_h = torch.where(tp_rec_h + fn_h > 0, tp_rec_h / (tp_rec_h + fn_h), torch.zeros_like(tp_rec_h))
    f1_h = torch.where(
        precision_h + recall_h > 0,
        2.0 * precision_h * recall_h / (precision_h + recall_h),
        torch.zeros_like(precision_h),
    )

    gt_extrema_per_h = gt_bin.sum(dim=reduce_dims).to(torch.int64)
    rec_extrema_per_h = rec_bin.sum(dim=reduce_dims).to(torch.int64)
    weights = gt_extrema_per_h.to(torch.float64)
    total_weight = float(weights.sum().item())

    return {
        "f1_mean": float(torch.mean(f1_h).item()) if f1_h.numel() > 0 else float("nan"),
        "f1_mean_weighted_h": float((f1_h * weights).sum().item() / total_weight) if total_weight > 0 else float("nan"),
        "f1_per_level": [float(v) for v in f1_h.detach().cpu().tolist()],
        "confusion_per_level": _build_tolerant_confusion_list(tp_prec_h, tp_rec_h, fp_h, fn_h, tn_h),
        "gt_extrema_per_h": [int(v) for v in gt_extrema_per_h.detach().cpu().tolist()],
        "rec_extrema_per_h": [int(v) for v in rec_extrema_per_h.detach().cpu().tolist()],
    }


def _get_f1_score_torch_detailed(
    min_max_idx_truth: torch.Tensor,
    min_max_idx_ae: torch.Tensor,
    axs: int = -2,
    kernel_size: int = 5,
) -> Dict[str, Any]:
    truth = _ensure_chw(min_max_idx_truth, "min_max_idx_truth")
    pred = _ensure_chw(min_max_idx_ae, "min_max_idx_ae")
    if truth.shape != pred.shape:
        raise ValueError(f"Shape mismatch: {tuple(truth.shape)} vs {tuple(pred.shape)}")
    if kernel_size < 1:
        raise ValueError(f"kernel_size must be >= 1, got {kernel_size}")

    ndim = truth.ndim
    if axs < 0:
        axs = ndim + axs

    truth = truth > 0
    pred = pred > 0
    truth_h_last = truth.movedim(axs, -1)
    pred_h_last = pred.movedim(axs, -1)
    h_size = truth_h_last.shape[-1]

    flat_truth = truth_h_last.reshape(-1, 1, h_size).to(torch.float32)
    flat_pred = pred_h_last.reshape(-1, 1, h_size).to(torch.float32)
    kernel = torch.ones(1, 1, kernel_size, device=truth.device, dtype=torch.float32)
    pad = kernel_size // 2

    truth_expanded = F.conv1d(flat_truth, kernel, padding=pad)
    pred_expanded = F.conv1d(flat_pred, kernel, padding=pad)
    if truth_expanded.shape[-1] != h_size:
        truth_expanded = truth_expanded[..., :h_size]
        pred_expanded = pred_expanded[..., :h_size]

    truth_win = (truth_expanded > 0).reshape(*truth_h_last.shape)
    pred_win = (pred_expanded > 0).reshape(*pred_h_last.shape)

    tp_map_prec = truth_win & pred_h_last
    tp_map_rec = truth_h_last & pred_win
    fp_map = (~truth_win) & pred_h_last
    fn_map = truth_h_last & (~pred_win)

    def _reduce_along(dim: int):
        tp_prec = tp_map_prec.sum(dim=dim).to(torch.float64).sum(dim=0)
        tp_rec = tp_map_rec.sum(dim=dim).to(torch.float64).sum(dim=0)
        fp = fp_map.sum(dim=dim).to(torch.float64).sum(dim=0)
        fn = fn_map.sum(dim=dim).to(torch.float64).sum(dim=0)
        prec = torch.where(tp_prec + fp > 0, tp_prec / (tp_prec + fp), torch.zeros_like(tp_prec))
        rec = torch.where(tp_rec + fn > 0, tp_rec / (tp_rec + fn), torch.zeros_like(tp_rec))
        f1 = torch.where(prec + rec > 0, 2.0 * prec * rec / (prec + rec), torch.zeros_like(prec))
        return f1, tp_prec, tp_rec, fp, fn

    f1_per_w, tp_prec_w, tp_rec_w, fp_w, fn_w = _reduce_along(dim=-1)
    f1_per_h, tp_prec_h, tp_rec_h, fp_h, fn_h = _reduce_along(dim=-2)

    total_per_h = int(tp_map_prec.shape[0] * tp_map_prec.shape[1])
    total_per_w = int(tp_map_prec.shape[0] * tp_map_prec.shape[2])

    # conservative TNs derived from the precision-view support space
    tn_h = torch.clamp(torch.full_like(tp_prec_h, float(total_per_h)) - torch.minimum(tp_prec_h, tp_rec_h) - fp_h - fn_h, min=0.0)
    tn_w = torch.clamp(torch.full_like(tp_prec_w, float(total_per_w)) - torch.minimum(tp_prec_w, tp_rec_w) - fp_w - fn_w, min=0.0)

    gt_extrema_per_h = truth_h_last.sum(dim=(0, 1)).long()
    rec_extrema_per_h = pred_h_last.sum(dim=(0, 1)).long()
    weights = gt_extrema_per_h.to(torch.float64)
    total_weight = float(weights.sum().item())

    return {
        "f1_per_w": [float(v) for v in f1_per_w.detach().cpu().tolist()],
        "confusion_per_w": _build_tolerant_confusion_list(tp_prec_w, tp_rec_w, fp_w, fn_w, tn_w),
        "f1_per_level": [float(v) for v in f1_per_h.detach().cpu().tolist()],
        "f1_per_h": [float(v) for v in f1_per_h.detach().cpu().tolist()],
        "confusion_per_level": _build_tolerant_confusion_list(tp_prec_h, tp_rec_h, fp_h, fn_h, tn_h),
        "confusion_per_h": _build_tolerant_confusion_list(tp_prec_h, tp_rec_h, fp_h, fn_h, tn_h),
        "f1_mean": float(f1_per_w.mean().item()) if f1_per_w.numel() > 0 else float("nan"),
        "f1_mean_h": float(f1_per_h.mean().item()) if f1_per_h.numel() > 0 else float("nan"),
        "gt_extrema_per_h": [int(v) for v in gt_extrema_per_h.detach().cpu().tolist()],
        "rec_extrema_per_h": [int(v) for v in rec_extrema_per_h.detach().cpu().tolist()],
        "f1_mean_weighted_h": float((f1_per_h * weights).sum().item() / total_weight) if total_weight > 0 else float("nan"),
    }


def f1_and_confusion_per_level(
    truth_onehot: torch.Tensor,
    pred_onehot: torch.Tensor,
    kernel_size: int = 5,
) -> Dict[str, Any]:
    truth_onehot = _ensure_chw(truth_onehot, "truth_onehot")
    pred_onehot = _ensure_chw(pred_onehot, "pred_onehot")
    if truth_onehot.shape != pred_onehot.shape:
        raise ValueError(f"Shape mismatch: {tuple(truth_onehot.shape)} vs {tuple(pred_onehot.shape)}")
    return _get_f1_score_torch_detailed(truth_onehot, pred_onehot, axs=-2, kernel_size=kernel_size)


def f1_and_confusion_per_level_split(
    truth_min_onehot: torch.Tensor,
    truth_max_onehot: torch.Tensor,
    pred_onehot: torch.Tensor,
    kernel_size: int = 5,
) -> Dict[str, Any]:
    """Tolerant F1 split by extremum type (min / max), then combined.

    Computes the tolerant F1 score independently for local minima and local
    maxima, then reports both individual scores and their average as the
    combined score.

    Parameters
    ----------
    truth_min_onehot : (C, H, W) binary mask  — GT local minima positions.
    truth_max_onehot : (C, H, W) binary mask  — GT local maxima positions.
    pred_onehot      : (C, H, W) binary mask  — predicted extrema positions
                       (may be type-unaware, covering both min and max).
    kernel_size : int
        Tolerance window: a prediction within kernel_size // 2 levels of a GT
        extremum counts as a TP.

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
    truth_min_onehot = _ensure_chw(truth_min_onehot, "truth_min_onehot")
    truth_max_onehot = _ensure_chw(truth_max_onehot, "truth_max_onehot")
    pred_onehot = _ensure_chw(pred_onehot, "pred_onehot")

    if truth_min_onehot.shape != pred_onehot.shape:
        raise ValueError(
            f"Shape mismatch: truth_min {tuple(truth_min_onehot.shape)} vs pred {tuple(pred_onehot.shape)}"
        )
    if truth_max_onehot.shape != pred_onehot.shape:
        raise ValueError(
            f"Shape mismatch: truth_max {tuple(truth_max_onehot.shape)} vs pred {tuple(pred_onehot.shape)}"
        )

    res_min = _get_f1_score_torch_detailed(truth_min_onehot, pred_onehot, axs=-2, kernel_size=kernel_size)
    res_max = _get_f1_score_torch_detailed(truth_max_onehot, pred_onehot, axs=-2, kernel_size=kernel_size)

    def _nanmean2(a: float, b: float) -> float:
        vals = [v for v in (a, b) if not (v != v)]  # exclude NaN
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


# ============================================================================
# Event-based matching metrics
# ============================================================================

def _match_event_indices_1d(
    gt_idx: Union[torch.Tensor, np.ndarray, Sequence[int]],
    pred_idx: Union[torch.Tensor, np.ndarray, Sequence[int]],
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
    return event_match_stats_per_level(truth_onehot, pred_onehot, kernel_size=kernel_size, axs=-2)


# ============================================================================
# Prominence-aware matching metrics
# ============================================================================

def _extract_prominent_extrema_1d(
    x: np.ndarray,
    kind: str = "both",
    prominence_fraction: float = 0.1,
    min_distance: int = 3,
    min_width: int = 1,
    prominence_reference_scale: Optional[float] = None,
) -> Dict[str, np.ndarray]:
    if _scipy_signal is None:
        raise ImportError("scipy.signal is required for prominence-aware metrics")

    x = np.asarray(x, dtype=np.float64)
    valid = np.isfinite(x)
    x_filled = _interp_nan_1d(x)
    scale = float(prominence_reference_scale) if prominence_reference_scale is not None else _robust_profile_scale(x)
    prom_thr = max(1e-12, float(prominence_fraction) * max(scale, 1e-12))

    out = {
        "idx": np.empty((0,), dtype=np.int64),
        "prom": np.empty((0,), dtype=np.float64),
        "amp": np.empty((0,), dtype=np.float64),
        "sign": np.empty((0,), dtype=np.int64),
    }

    def _append(found_idx: np.ndarray, found_prom: np.ndarray, sign: int) -> None:
        if found_idx.size == 0:
            return
        found_idx = found_idx[valid[found_idx]]
        if found_idx.size == 0:
            return
        # need to re-mask prominences as well
        if found_prom.size != found_idx.size:
            # caller passed unmasked prominences
            mask = valid[np.asarray(found_idx, dtype=np.int64)]
            found_prom = found_prom[mask]
        out["idx"] = np.concatenate([out["idx"], found_idx.astype(np.int64, copy=False)])
        out["prom"] = np.concatenate([out["prom"], found_prom.astype(np.float64, copy=False)])
        out["amp"] = np.concatenate([out["amp"], x_filled[found_idx].astype(np.float64, copy=False)])
        out["sign"] = np.concatenate([out["sign"], sign * np.ones(found_idx.size, dtype=np.int64)])

    if kind in ("max", "both"):
        peaks, props = _scipy_signal.find_peaks(
            x_filled,
            prominence=prom_thr,
            distance=max(1, int(min_distance)),
            width=max(1, int(min_width)),
        )
        prom = np.asarray(props.get("prominences", np.zeros(len(peaks))), dtype=np.float64)
        valid_peaks = valid[peaks]
        _append(peaks[valid_peaks], prom[valid_peaks], +1)

    if kind in ("min", "both"):
        peaks, props = _scipy_signal.find_peaks(
            -x_filled,
            prominence=prom_thr,
            distance=max(1, int(min_distance)),
            width=max(1, int(min_width)),
        )
        prom = np.asarray(props.get("prominences", np.zeros(len(peaks))), dtype=np.float64)
        valid_peaks = valid[peaks]
        _append(peaks[valid_peaks], prom[valid_peaks], -1)

    if out["idx"].size > 0:
        order = np.argsort(out["idx"])
        for key in out:
            out[key] = out[key][order]
    return out


def _greedy_match_extrema(
    gt_idx: np.ndarray,
    rec_idx: np.ndarray,
    gt_prom: np.ndarray,
    rec_prom: np.ndarray,
    gt_sign: np.ndarray,
    rec_sign: np.ndarray,
    max_distance: int,
) -> List[Tuple[int, int]]:
    matches: List[Tuple[int, int]] = []
    used_rec: set[int] = set()
    order = np.argsort(-gt_prom) if gt_prom.size > 0 else np.empty((0,), dtype=np.int64)
    for gi in order.tolist():
        lo = np.searchsorted(rec_idx, gt_idx[gi] - max_distance, side="left")
        hi = np.searchsorted(rec_idx, gt_idx[gi] + max_distance, side="right")
        candidates: List[Tuple[int, float, int]] = []
        for rj in range(lo, hi):
            if rj in used_rec or gt_sign[gi] != rec_sign[rj]:
                continue
            d = abs(int(gt_idx[gi]) - int(rec_idx[rj]))
            if d <= max_distance:
                candidates.append((d, -float(rec_prom[rj]), rj))
        if not candidates:
            continue
        candidates.sort()
        rj = candidates[0][2]
        used_rec.add(rj)
        matches.append((int(gi), int(rj)))
    return matches


def prominence_matching_metrics(
    gt: torch.Tensor,
    rec: torch.Tensor,
    valid_mask: Optional[torch.Tensor] = None,
    kind: str = "both",
    prominence_fraction: float = 0.1,
    min_width: int = 1,
    min_distance: int = 3,
    match_radius: int = 2,
) -> Dict[str, float]:
    gt = _ensure_chw(gt, "gt")
    rec = _ensure_chw(rec, "rec")
    if gt.shape != rec.shape:
        raise ValueError(f"Shape mismatch: {tuple(gt.shape)} vs {tuple(rec.shape)}")
    if valid_mask is None:
        valid_mask = torch.isfinite(gt)
    else:
        valid_mask = _ensure_chw(valid_mask, "valid_mask").bool() & torch.isfinite(gt)

    gt_np = gt.detach().cpu().numpy().astype(np.float64, copy=False)
    rec_np = rec.detach().cpu().numpy().astype(np.float64, copy=False)
    valid_np = valid_mask.detach().cpu().numpy().astype(bool, copy=False)

    weighted_recalls: List[float] = []
    weighted_precisions: List[float] = []
    loc_errs: List[float] = []
    prom_rel_errs: List[float] = []
    amp_errs: List[float] = []
    count_biases: List[float] = []
    gt_counts: List[float] = []
    rec_counts: List[float] = []

    C, _, W = gt_np.shape
    for c in range(C):
        for w in range(W):
            ok = valid_np[c, :, w]
            if int(ok.sum()) < 3:
                continue

            gt_prof = gt_np[c, :, w].copy()
            rec_prof = rec_np[c, :, w].copy()
            gt_prof[~ok] = np.nan
            rec_prof[~ok] = np.nan

            ref_scale = _robust_profile_scale(gt_prof)
            gt_ext = _extract_prominent_extrema_1d(
                gt_prof,
                kind=kind,
                prominence_fraction=prominence_fraction,
                min_distance=min_distance,
                min_width=min_width,
                prominence_reference_scale=ref_scale,
            )
            rec_ext = _extract_prominent_extrema_1d(
                rec_prof,
                kind=kind,
                prominence_fraction=prominence_fraction,
                min_distance=min_distance,
                min_width=min_width,
                prominence_reference_scale=ref_scale,
            )

            gt_idx = gt_ext["idx"]
            gt_prom = gt_ext["prom"]
            gt_amp = gt_ext["amp"]
            gt_sign = gt_ext["sign"]
            rec_idx = rec_ext["idx"]
            rec_prom = rec_ext["prom"]
            rec_amp = rec_ext["amp"]
            rec_sign = rec_ext["sign"]

            gt_counts.append(float(gt_idx.size))
            rec_counts.append(float(rec_idx.size))
            count_biases.append(float(rec_idx.size - gt_idx.size))

            gt_mass = float(gt_prom.sum())
            rec_mass = float(rec_prom.sum())
            if gt_idx.size == 0 and rec_idx.size == 0:
                weighted_recalls.append(1.0)
                weighted_precisions.append(1.0)
                continue

            matches = _greedy_match_extrema(
                gt_idx, rec_idx, gt_prom, rec_prom, gt_sign, rec_sign, max_distance=max(0, int(match_radius))
            )
            matched_gt_mass = float(sum(gt_prom[gi] for gi, _ in matches))
            matched_rec_mass = float(sum(rec_prom[rj] for _, rj in matches))
            weighted_recalls.append(matched_gt_mass / (gt_mass + 1e-12) if gt_mass > 0 else 1.0)
            weighted_precisions.append(matched_rec_mass / (rec_mass + 1e-12) if rec_mass > 0 else 1.0)

            for gi, rj in matches:
                loc_errs.append(float(abs(int(gt_idx[gi]) - int(rec_idx[rj]))))
                prom_rel_errs.append(float(abs(gt_prom[gi] - rec_prom[rj]) / (abs(gt_prom[gi]) + 1e-12)))
                amp_errs.append(float(abs(gt_amp[gi] - rec_amp[rj])))

    wr = float(np.mean(weighted_recalls)) if weighted_recalls else float("nan")
    wp = float(np.mean(weighted_precisions)) if weighted_precisions else float("nan")
    wf1 = _f1_from_precision_recall(wp, wr) if np.isfinite(wr) and np.isfinite(wp) else float("nan")
    return {
        "weighted_recall": wr,
        "weighted_precision": wp,
        "weighted_f1": wf1,
        "loc_mae": float(np.mean(loc_errs)) if loc_errs else float("nan"),
        "prom_rel_mae": float(np.mean(prom_rel_errs)) if prom_rel_errs else float("nan"),
        "amp_mae": float(np.mean(amp_errs)) if amp_errs else float("nan"),
        "count_bias": float(np.mean(count_biases)) if count_biases else float("nan"),
        "gt_count_mean": float(np.mean(gt_counts)) if gt_counts else float("nan"),
        "rec_count_mean": float(np.mean(rec_counts)) if rec_counts else float("nan"),
    }



def _nanmean_finite(values: Sequence[float]) -> float:
    vals = [float(v) for v in values if np.isfinite(float(v))]
    return float(np.mean(vals)) if vals else float("nan")


def prominence_matching_metrics_split(
    gt: torch.Tensor,
    rec: torch.Tensor,
    valid_mask: Optional[torch.Tensor] = None,
    prominence_fraction: float = 0.1,
    min_width: int = 1,
    min_distance: int = 3,
    match_radius: int = 2,
    include_both: bool = True,
) -> Dict[str, Any]:
    """Prominence-weighted matching split like the standard F1 metrics.

    This computes the prominence-weighted matching independently for local
    minima and local maxima, then defines the balanced ``combined`` score as
    the finite nan-mean of the min and max scores.  The optional ``both`` score
    is kept for backward compatibility: it pools minima and maxima before
    matching, so it is not type-balanced.

    Returned flat keys include, for example:
      - ``weighted_f1_min``
      - ``weighted_f1_max``
      - ``weighted_f1_combined``
      - ``weighted_f1_both`` when ``include_both=True``
    and the analogous precision/recall/error/count-bias keys.
    """

    def _one(kind: str) -> Dict[str, float]:
        out = prominence_matching_metrics(
            gt,
            rec,
            valid_mask=valid_mask,
            kind=kind,
            prominence_fraction=prominence_fraction,
            min_width=min_width,
            min_distance=min_distance,
            match_radius=match_radius,
        )
        return {
            "weighted_precision": float(out.get("weighted_precision", float("nan"))),
            "weighted_recall": float(out.get("weighted_recall", float("nan"))),
            "weighted_f1": float(out.get("weighted_f1", float("nan"))),
            "loc_mae": float(out.get("loc_mae", float("nan"))),
            "prom_rel_mae": float(out.get("prom_rel_mae", float("nan"))),
            "amp_mae": float(out.get("amp_mae", float("nan"))),
            "count_bias": float(out.get("count_bias", float("nan"))),
            "gt_count_mean": float(out.get("gt_count_mean", float("nan"))),
            "rec_count_mean": float(out.get("rec_count_mean", float("nan"))),
        }

    res_min = _one("min")
    res_max = _one("max")
    res_combined = {
        key: _nanmean_finite([res_min.get(key, float("nan")), res_max.get(key, float("nan"))])
        for key in res_min.keys()
    }

    result: Dict[str, Any] = {
        "min": res_min,
        "max": res_max,
        "combined": res_combined,
    }

    for suffix, block in (("min", res_min), ("max", res_max), ("combined", res_combined)):
        for key, value in block.items():
            result[f"{key}_{suffix}"] = float(value)

    if include_both:
        res_both = _one("both")
        result["both"] = res_both
        for key, value in res_both.items():
            result[f"{key}_both"] = float(value)

    return result


# ============================================================================
# Convenience wrappers
# ============================================================================

def f1_extrema(
    gt: torch.Tensor,
    rec: torch.Tensor,
    kind: str = "both",
    kernel_size: int = 5,
    grad_eps: Union[float, str] = "auto",
    smooth_window: int = 1,
    min_separation: int = 3,
    fill_zero_signs: bool = True,
    detector_version: str = "v2",
    side_window: int = 5,
    min_support_count: int = 1,
) -> Dict[str, Any]:
    gt = _ensure_chw(gt, "gt")
    rec = _ensure_chw(rec, "rec")
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

    return _f1_from_masks(gt_mask, rec_mask, valid, kernel_size)


def f1_extrema_prominence(
    gt: torch.Tensor,
    rec: torch.Tensor,
    kind: str = "both",
    kernel_size: int = 5,
    prominence_fraction: float = 0.1,
    min_width: int = 1,
    min_distance: int = 3,
) -> Dict[str, Any]:
    gt = _ensure_chw(gt, "gt")
    rec = _ensure_chw(rec, "rec")
    valid = _nan_mask(gt)
    gt_mask = detect_extrema_prominence(
        gt,
        kind=kind,
        prominence_fraction=prominence_fraction,
        min_width=min_width,
        min_distance=min_distance,
    )
    rec_mask = detect_extrema_prominence(
        rec,
        kind=kind,
        prominence_fraction=prominence_fraction,
        min_width=min_width,
        min_distance=min_distance,
    )
    return _f1_from_masks(gt_mask, rec_mask, valid, kernel_size)


# ============================================================================
# Report writer
# ============================================================================

def write_minmax_report_txt(
    path: str,
    rows: List[Dict[str, Any]],
    header: Optional[Dict[str, Any]] = None,
) -> None:
    """Write a concise readable report for scalar extrema metrics."""
    path_obj = Path(path)
    path_obj.parent.mkdir(parents=True, exist_ok=True)

    if not rows:
        path_obj.write_text("# Extrema metrics report\n\n(no rows)\n", encoding="utf-8")
        return

    keys = list(rows[0].keys())

    def _parse_scalar(v: Any) -> Optional[float]:
        try:
            val = float(v)
        except Exception:
            return None
        return val if np.isfinite(val) else None

    def _parse_numeric_list(v: Any) -> Optional[np.ndarray]:
        if isinstance(v, np.ndarray):
            arr = v.astype(np.float64, copy=False).reshape(-1)
            return arr if arr.size > 0 else None
        if isinstance(v, (list, tuple)):
            try:
                arr = np.asarray(v, dtype=np.float64).reshape(-1)
                return arr if arr.size > 0 else None
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
                    arr = np.asarray(parsed, dtype=np.float64).reshape(-1)
                    return arr if arr.size > 0 else None
                except Exception:
                    return None
        return None

    def _mse_to_rmse_aggregate_key_vals(k: str, vals: List[float]) -> Tuple[str, np.ndarray]:
        is_mse_like = ("MSE" in k) and ("MSE_over_STD" not in k)
        arr = np.asarray(vals, dtype=np.float64)
        if not is_mse_like:
            return k, arr
        arr = np.where(arr >= 0.0, np.sqrt(arr), np.nan)
        return k.replace("MSE", "RMSE"), arr

    def _is_mse_like_key(k: str) -> bool:
        return ("MSE" in k) and ("MSE_over_STD" not in k)

    def _convert_mse_like_value(v: Any) -> Any:
        if isinstance(v, (int, float, np.floating, np.integer)):
            vf = float(v)
            return float(np.sqrt(vf)) if np.isfinite(vf) and vf >= 0.0 else float("nan")
        arr = _parse_numeric_list(v)
        if arr is not None:
            arr = np.where(arr >= 0.0, np.sqrt(arr), np.nan)
            return [float(x) if np.isfinite(x) else float("nan") for x in arr.tolist()]
        try:
            vf = float(v)
            return float(np.sqrt(vf)) if np.isfinite(vf) and vf >= 0.0 else float("nan")
        except Exception:
            return v

    with path_obj.open("w", encoding="utf-8") as f:
        f.write("# Extrema metrics v2 report\n\n")
        if header:
            f.write("## Run info\n")
            for k, v in header.items():
                f.write(f"- {k}: {v}\n")
            f.write("\n")

        f.write("## Aggregate scalar metrics\n")
        for k in keys:
            vals = [_parse_scalar(r.get(k)) for r in rows]
            vals = [v for v in vals if v is not None]
            if not vals:
                continue
            out_key, arr = _mse_to_rmse_aggregate_key_vals(k, vals)
            arr = arr[np.isfinite(arr)]
            if arr.size == 0:
                continue
            f.write(
                f"- {out_key}: mean={arr.mean():.6g}, std={arr.std(ddof=0):.6g}, min={arr.min():.6g}, max={arr.max():.6g}\n"
            )

        list_keys = [k for k in keys if k.endswith("_levels") or k.endswith("_per_h")]
        if list_keys:
            f.write("\n## Aggregate list metrics\n")
            for k in list_keys:
                parsed = [_parse_numeric_list(r.get(k)) for r in rows]
                parsed = [a for a in parsed if a is not None]
                if not parsed:
                    continue
                n = max(a.size for a in parsed)
                stacked = np.full((len(parsed), n), np.nan, dtype=np.float64)
                for i, arr in enumerate(parsed):
                    stacked[i, : arr.size] = arr
                means = np.nanmean(stacked, axis=0)
                payload = [None if not np.isfinite(v) else float(v) for v in means.tolist()]
                f.write(f"- {k}: {json.dumps(payload)}\n")

        f.write("\n## Per-image rows (JSON lines)\n")
        for row in rows:
            row_out: Dict[str, Any] = {}
            for k, v in row.items():
                out_k = k.replace("MSE", "RMSE") if _is_mse_like_key(k) else k
                out_v = _convert_mse_like_value(v) if _is_mse_like_key(k) else v
                row_out[out_k] = out_v
            f.write(json.dumps(row_out, ensure_ascii=False) + "\n")


__all__ = [
    "apply_gt_nan_mask_torch",
    "detect_extrema",
    "detect_extrema_v2",
    "detect_extrema_all_v2",
    "detect_extrema_prominence",
    "extrema_wasserstein",
    "lowpass_filter_torch_along_axis",
    "f1_and_confusion_per_level",
    "event_match_stats_per_level",
    "f1_and_match_stats_per_level",
    "prominence_matching_metrics",
    "prominence_matching_metrics_split",
    "f1_extrema",
    "f1_extrema_prominence",
    "write_minmax_report_txt",
]