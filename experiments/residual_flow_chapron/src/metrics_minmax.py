"""
Extrema detection and F1 scoring metrics for MLIC++.

Focused module containing:
- detect_extrema: robust gradient-sign-inversion detector (with grad_eps, zero-fill, min_separation)
- detect_extrema_prominence: scipy.signal.find_peaks based detector using prominence
- extrema_wasserstein: continuous distance between extrema distributions
- F1 scoring engine (tolerant matching via 1D convolution)
- Legacy detectors (get_min_or_max_idx, get_min_max_idx) kept for comparison
"""

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

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

try:
    from scipy.ndimage import convolve as _nd_convolve
except Exception:
    _nd_convolve = None


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


# ===========================================================================
# Extrema detection helpers (gradient-sign-inversion with robustness)
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


# ===========================================================================
# Batched (vectorized) helpers — loop over H (~155) instead of C*W (~12000)
# ===========================================================================

def _smooth_nan_batched(x: torch.Tensor, window: int) -> torch.Tensor:
    """NaN-aware smoothing along H for (C, H, W). Fully vectorized via conv1d."""
    C, H, W = x.shape
    kernel = torch.ones(1, 1, window, dtype=x.dtype, device=x.device)
    pad_size = window // 2
    valid = torch.isfinite(x).to(x.dtype)
    x0 = torch.where(torch.isfinite(x), x, torch.zeros_like(x))
    # (C, H, W) -> (C*W, 1, H) to convolve along H
    x_flat = x0.permute(0, 2, 1).reshape(C * W, 1, H)
    v_flat = valid.permute(0, 2, 1).reshape(C * W, 1, H)
    num = F.conv1d(x_flat, kernel, padding=pad_size)[:, 0, :H]
    den = F.conv1d(v_flat, kernel, padding=pad_size)[:, 0, :H]
    out_flat = torch.full_like(num, float("nan"))
    ok = den > 0
    out_flat[ok] = num[ok] / den[ok]
    return out_flat.reshape(C, W, H).permute(0, 2, 1)


def _fill_zero_signs_batched(s: torch.Tensor) -> torch.Tensor:
    """Forward-backward zero-fill on sign tensor. s: (C, L, W).
    Loops over L (~155) instead of C*W profiles."""
    out = s.clone()
    _, L, _ = out.shape
    # Forward fill along dim=1
    for k in range(1, L):
        mask = out[:, k, :] == 0
        out[:, k, :] = torch.where(mask, out[:, k - 1, :], out[:, k, :])
    # Backward fill along dim=1
    for k in range(L - 2, -1, -1):
        mask = out[:, k, :] == 0
        out[:, k, :] = torch.where(mask, out[:, k + 1, :], out[:, k, :])
    return out

def _safe_div(num: float, den: float) -> float:
    return float(num) / float(den) if den > 0 else 0.0

def _f1_from_precision_recall(precision: float, recall: float) -> float:
    den = precision + recall
    return 2.0 * precision * recall / den if den > 0 else 0.0

def _match_event_indices_1d(
    gt_idx: torch.Tensor,
    pred_idx: torch.Tensor,
    tolerance: int,
) -> list[tuple[int, int, int]]:
    """
    One-to-one greedy matching between GT and predicted event indices.

    Returns
    -------
    matches : list of tuples (gt_pos, pred_pos, abs_offset)
        gt_pos / pred_pos are integer positions in H.
    """
    if gt_idx.numel() == 0 or pred_idx.numel() == 0:
        return []

    gt_list = [int(v) for v in gt_idx.detach().cpu().tolist()]
    pred_list = [int(v) for v in pred_idx.detach().cpu().tolist()]

    candidates: list[tuple[int, int, int]] = []
    for ig, g in enumerate(gt_list):
        for ip, p in enumerate(pred_list):
            d = abs(g - p)
            if d <= tolerance:
                candidates.append((d, ig, ip))

    # Closest pairs first
    candidates.sort(key=lambda t: (t[0], t[1], t[2]))

    used_gt = [False] * len(gt_list)
    used_pred = [False] * len(pred_list)
    matches: list[tuple[int, int, int]] = []

    for d, ig, ip in candidates:
        if used_gt[ig] or used_pred[ip]:
            continue
        used_gt[ig] = True
        used_pred[ip] = True
        matches.append((gt_list[ig], pred_list[ip], d))

    return matches


def event_match_stats_per_level(
    truth_onehot: torch.Tensor,
    pred_onehot: torch.Tensor,
    kernel_size: int = 5,
    axs: int = -2,
) -> dict[str, Any]:
    """
    Event-level replacement for the old confusion matrix.

    Parameters
    ----------
    truth_onehot, pred_onehot : torch.Tensor
        Binary/event masks with shape (C,H,W) or compatible.
    kernel_size : int
        Matching tolerance along H. A GT event and a predicted event are
        considered a match if |h_gt - h_pred| <= kernel_size // 2.
    axs : int
        Axis of H.

    Returns
    -------
    dict with:
      - match_stats_per_h
      - gt_count_per_h
      - pred_count_per_h
      - matched_gt_per_h
      - matched_pred_per_h
      - overall_event_stats
      - event_f1_mean_h
      - event_f1_mean_weighted_h
    """
    if truth_onehot.shape != pred_onehot.shape:
        raise ValueError(f"Shape mismatch: {truth_onehot.shape} vs {pred_onehot.shape}")
    if truth_onehot.ndim < 2:
        raise ValueError(f"Expected at least 2 dimensions, got {truth_onehot.ndim}")
    if kernel_size < 1:
        raise ValueError(f"kernel_size must be >= 1, got {kernel_size}")

    ndim = truth_onehot.ndim
    if axs < 0:
        axs = ndim + axs

    truth = (truth_onehot > 0)
    pred = (pred_onehot > 0)

    # Put H last -> (..., H)
    truth_h_last = truth.movedim(axs, -1)
    pred_h_last = pred.movedim(axs, -1)

    if truth_h_last.shape != pred_h_last.shape:
        raise ValueError("Internal shape mismatch after movedim")

    *prefix, H = truth_h_last.shape
    n_profiles = int(np.prod(prefix)) if len(prefix) > 0 else 1

    truth_flat = truth_h_last.reshape(n_profiles, H)
    pred_flat = pred_h_last.reshape(n_profiles, H)

    tolerance = kernel_size // 2

    gt_count_per_h = torch.zeros(H, dtype=torch.int64)
    pred_count_per_h = torch.zeros(H, dtype=torch.int64)
    matched_gt_per_h = torch.zeros(H, dtype=torch.int64)
    matched_pred_per_h = torch.zeros(H, dtype=torch.int64)
    offset_sum_per_h = torch.zeros(H, dtype=torch.float64)
    offset_count_per_h = torch.zeros(H, dtype=torch.int64)

    total_gt = 0
    total_pred = 0
    total_matches = 0
    total_abs_offset = 0.0

    for prof_idx in range(n_profiles):
        gt_idx = torch.nonzero(truth_flat[prof_idx], as_tuple=False).view(-1)
        pred_idx = torch.nonzero(pred_flat[prof_idx], as_tuple=False).view(-1)

        for g in gt_idx.tolist():
            gt_count_per_h[g] += 1
        for p in pred_idx.tolist():
            pred_count_per_h[p] += 1

        total_gt += int(gt_idx.numel())
        total_pred += int(pred_idx.numel())

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
        gt_count = int(gt_count_per_h[h].item())
        pred_count = int(pred_count_per_h[h].item())
        matched_gt = int(matched_gt_per_h[h].item())
        matched_pred = int(matched_pred_per_h[h].item())

        fn = gt_count - matched_gt
        fp = pred_count - matched_pred

        recall_h = _safe_div(matched_gt, gt_count)
        precision_h = _safe_div(matched_pred, pred_count)
        f1_h = _f1_from_precision_recall(precision_h, recall_h)

        mean_abs_offset_h = (
            float(offset_sum_per_h[h].item()) / float(offset_count_per_h[h].item())
            if int(offset_count_per_h[h].item()) > 0
            else float("nan")
        )

        event_f1_per_h.append(f1_h)
        match_stats_per_h.append({
            "gt_count": gt_count,
            "pred_count": pred_count,
            "matched_gt": matched_gt,
            "matched_pred": matched_pred,
            "fn": fn,
            "fp": fp,
            "recall": float(recall_h),
            "precision": float(precision_h),
            "f1": float(f1_h),
            "mean_abs_offset": mean_abs_offset_h,
        })

    overall_precision = _safe_div(total_matches, total_pred)
    overall_recall = _safe_div(total_matches, total_gt)
    overall_f1 = _f1_from_precision_recall(overall_precision, overall_recall)
    overall_mean_abs_offset = total_abs_offset / total_matches if total_matches > 0 else float("nan")

    weights = gt_count_per_h.to(torch.float64)
    total_weight = float(weights.sum().item())
    event_f1_mean_weighted_h = (
        float((torch.tensor(event_f1_per_h, dtype=torch.float64) * weights).sum().item() / total_weight)
        if total_weight > 0 else float("nan")
    )

    return {
        "match_stats_per_h": match_stats_per_h,
        "gt_count_per_h": [int(v) for v in gt_count_per_h.tolist()],
        "pred_count_per_h": [int(v) for v in pred_count_per_h.tolist()],
        "matched_gt_per_h": [int(v) for v in matched_gt_per_h.tolist()],
        "matched_pred_per_h": [int(v) for v in matched_pred_per_h.tolist()],
        "event_f1_per_h": [float(v) for v in event_f1_per_h],
        "event_f1_mean_h": float(np.mean(event_f1_per_h)) if len(event_f1_per_h) > 0 else float("nan"),
        "event_f1_mean_weighted_h": event_f1_mean_weighted_h,
        "overall_event_stats": {
            "gt_count": int(total_gt),
            "pred_count": int(total_pred),
            "tp": int(total_matches),
            "fp": int(total_pred - total_matches),
            "fn": int(total_gt - total_matches),
            "precision": float(overall_precision),
            "recall": float(overall_recall),
            "f1": float(overall_f1),
            "mean_abs_offset": float(overall_mean_abs_offset),
        },
    }
    
def _suppress_close_extrema_batched(
    mask: torch.Tensor, min_separation: int
) -> torch.Tensor:
    """Greedy left-to-right suppression. mask: (C, L, W) bool.
    Loops over L (~155) instead of C*W profiles."""
    if min_separation <= 1:
        return mask
    C, L, W = mask.shape
    out = mask.clone()
    last_kept = torch.full((C, W), -(10**9), dtype=torch.long, device=mask.device)
    for k in range(L):
        active = out[:, k, :]
        too_close = (k - last_kept) < min_separation
        out[:, k, :] = active & ~too_close
        last_kept = torch.where(out[:, k, :], k, last_kept)
    return out


def _compute_turning_points(
    x: torch.Tensor,
    grad_eps: Union[float, str],
    smooth_window: int,
    fill_zero_signs: bool,
    grad_eps_reference: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Shared computation for detect_extrema / detect_extrema_all.
    x: (C, H, W). Returns turning: (C, H-2, W).
    grad_eps_reference: if provided and grad_eps=="auto", compute the
        threshold from this tensor (typically GT) instead of x."""
    C, H, W = x.shape
    smooth_window = max(1, int(smooth_window))
    if smooth_window % 2 == 0:
        smooth_window += 1
    x_work = x.to(torch.float64).clone()
    if smooth_window > 1:
        x_work = _smooth_nan_batched(x_work, smooth_window)
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
        sign = _fill_zero_signs_batched(sign)
    return torch.diff(sign, dim=1)  # (C, H-2, W)

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

    Uses gradient-sign inversion with robustness features:
    - grad_eps: ignores tiny gradients below this threshold ("auto" = 0.5 * P10(|grad|))
    - fill_zero_signs: propagates last nonzero sign through zero-gradient plateaus
    - min_separation: suppresses extrema closer than this many levels apart
    - grad_eps_reference: if provided and grad_eps=="auto", compute the
        threshold from this tensor (typically GT) instead of x.
    """
    if kind not in ("min", "max", "both"):
        raise ValueError(f"kind must be 'min', 'max', or 'both', got '{kind}'")
    x = _ensure_chw(x)
    C, H, W = x.shape
    if H < 3:
        return torch.zeros_like(x, dtype=torch.float32)
    turning = _compute_turning_points(x, grad_eps, smooth_window, fill_zero_signs, grad_eps_reference)
    if kind == "min":
        core = turning > 0
    elif kind == "max":
        core = turning < 0
    else:
        core = turning != 0
    if min_separation > 1:
        core = _suppress_close_extrema_batched(core, min_separation)
    out = torch.zeros(C, H, W, dtype=torch.float32, device=x.device)
    out[:, 1:-1, :] = core.to(torch.float32)
    return out * torch.isfinite(x).to(torch.float32)


def detect_extrema_all(
    x: torch.Tensor,
    grad_eps: Union[float, str] = "auto",
    smooth_window: int = 1,
    min_separation: int = 3,
    fill_zero_signs: bool = True,
    grad_eps_reference: Optional[torch.Tensor] = None,
) -> Dict[str, torch.Tensor]:
    """Single-pass extrema detection returning min, max, and both masks.

    Computes grad → sign → turning once, then splits into min/max/both.
    ~3x faster than calling detect_extrema three times.
    """
    x = _ensure_chw(x)
    C, H, W = x.shape
    zeros = torch.zeros(C, H, W, dtype=torch.float32, device=x.device)
    if H < 3:
        return {"min": zeros.clone(), "max": zeros.clone(), "both": zeros.clone()}
    turning = _compute_turning_points(x, grad_eps, smooth_window, fill_zero_signs, grad_eps_reference)
    valid_f = torch.isfinite(x).to(torch.float32)
    results: Dict[str, torch.Tensor] = {}
    for kname, cond in [("min", turning > 0), ("max", turning < 0), ("both", turning != 0)]:
        core = cond
        if min_separation > 1:
            core = _suppress_close_extrema_batched(core, min_separation)
        out = torch.zeros(C, H, W, dtype=torch.float32, device=x.device)
        out[:, 1:-1, :] = core.to(torch.float32)
        results[kname] = out * valid_f
    return results


# ===========================================================================
# Prominence-based extrema detection (scipy.signal.find_peaks)
# ===========================================================================

def detect_extrema_prominence(
    x: torch.Tensor,
    kind: str = "both",
    prominence_fraction: float = 0.1,
    min_width: int = 1,
    min_distance: int = 3,
) -> torch.Tensor:
    """Detect local extrema along H axis using scipy.signal.find_peaks with prominence.

    Parameters
    ----------
    x : (C, H, W) or (1, C, H, W) tensor
    kind : "min", "max", or "both"
    prominence_fraction : prominence threshold as a fraction of per-profile std
    min_width : minimum peak width (samples)
    min_distance : minimum distance between peaks (samples)

    Returns
    -------
    (C, H, W) float32 tensor with 1.0 at detected extrema, 0.0 elsewhere
    """
    if _scipy_signal is None:
        raise ImportError("scipy.signal is required for detect_extrema_prominence")
    if kind not in ("min", "max", "both"):
        raise ValueError(f"kind must be 'min', 'max', or 'both', got '{kind}'")
    x = _ensure_chw(x)
    C, H, W = x.shape
    # Work entirely in numpy; single flat loop over N=C*W profiles
    arr = x.detach().cpu().numpy().astype(np.float64)
    profiles = arr.transpose(0, 2, 1).reshape(C * W, H)  # (N, H)
    valid_all = np.isfinite(profiles)                      # (N, H)
    valid_count = valid_all.sum(axis=1)                    # (N,)
    std_vals = np.nanstd(profiles, axis=1)                 # (N,) — vectorized
    proms = np.maximum(1e-12, prominence_fraction * std_vals)
    out_np = np.zeros((C * W, H), dtype=np.float32)

    for n in range(C * W):
        if valid_count[n] < 3:
            continue
        profile = profiles[n]
        v = valid_all[n]
        prom = float(proms[n])
        if kind in ("max", "both"):
            peaks, _ = _scipy_signal.find_peaks(
                np.where(v, profile, -np.inf),
                prominence=prom, width=min_width, distance=min_distance,
            )
            out_np[n, peaks] = 1.0
        if kind in ("min", "both"):
            peaks, _ = _scipy_signal.find_peaks(
                np.where(v, -profile, -np.inf),
                prominence=prom, width=min_width, distance=min_distance,
            )
            out_np[n, peaks] = 1.0

    out = out_np.reshape(C, W, H).transpose(0, 2, 1)  # back to (C, H, W)
    return torch.from_numpy(np.ascontiguousarray(out)).to(device=x.device)


# ===========================================================================
# Wasserstein distance between extrema distributions
# ===========================================================================

def extrema_wasserstein(
    gt_mask: torch.Tensor,
    rec_mask: torch.Tensor,
    valid: Optional[torch.Tensor] = None,
) -> Dict[str, Any]:
    """Compute Wasserstein distance between GT and REC extrema positions along H.

    Treats extrema H-positions as 1D distributions for each (C, W) profile,
    then averages over all profiles that have at least one extremum in both GT and REC.

    Parameters
    ----------
    gt_mask : (C, H, W) binary tensor of GT extrema positions
    rec_mask : (C, H, W) binary tensor of REC extrema positions
    valid : optional (C, H, W) boolean mask

    Returns
    -------
    dict with keys:
        wasserstein_mean : float — mean Wasserstein distance over valid profiles
        wasserstein_std : float — std of Wasserstein distances
        wasserstein_per_h : list[float] — mean Wasserstein per H level (aggregated)
        n_profiles_used : int — profiles with extrema in both GT and REC
        n_profiles_total : int — total number of profiles
    """
    if _wasserstein_distance is None:
        raise ImportError("scipy.stats.wasserstein_distance is required for extrema_wasserstein")

    gt_mask = _ensure_chw(gt_mask)
    rec_mask = _ensure_chw(rec_mask)
    C, H, W = gt_mask.shape

    if valid is not None:
        valid = _ensure_chw(valid)
        gt_bin = (gt_mask > 0) & valid.bool()
        rec_bin = (rec_mask > 0) & valid.bool()
    else:
        gt_bin = gt_mask > 0
        rec_bin = rec_mask > 0

    gt_np = gt_bin.detach().cpu().numpy()
    rec_np = rec_bin.detach().cpu().numpy()
    h_indices = np.arange(H, dtype=np.float64)

    distances: List[float] = []
    per_h_contributions: List[List[float]] = [[] for _ in range(H)]
    n_total = C * W

    for c in range(C):
        for w in range(W):
            gt_positions = h_indices[gt_np[c, :, w]]
            rec_positions = h_indices[rec_np[c, :, w]]
            if gt_positions.size == 0 or rec_positions.size == 0:
                continue
            d = float(_wasserstein_distance(gt_positions, rec_positions))
            distances.append(d)
            # Attribute distance to the H levels where GT extrema occur
            for h_idx in gt_positions.astype(int):
                per_h_contributions[h_idx].append(d)

    wasserstein_per_h = [
        float(np.mean(vals)) if vals else float("nan")
        for vals in per_h_contributions
    ]

    return {
        "wasserstein_mean": float(np.mean(distances)) if distances else float("nan"),
        "wasserstein_std": float(np.std(distances)) if distances else float("nan"),
        "wasserstein_per_h": wasserstein_per_h,
        "n_profiles_used": len(distances),
        "n_profiles_total": n_total,
    }


# ===========================================================================
# Lowpass filter
# ===========================================================================

def lowpass_filter_torch_along_axis(
    arr: torch.Tensor,
    axis: int = -1,
    order: int = 2,
    wn: float = 0.107,
) -> torch.Tensor:
    """Apply zero-phase Butterworth low-pass filtering along `axis`."""
    if _scipy_signal is None:
        return arr
    arr_np = arr.detach().cpu().numpy().astype(np.float64, copy=False)
    b, a = _scipy_signal.butter(N=order, Wn=wn, btype="low", analog=False)
    filtered_np = _scipy_signal.filtfilt(b, a, arr_np, axis=axis)
    filtered_np = np.ascontiguousarray(filtered_np)
    return torch.from_numpy(filtered_np).to(device=arr.device, dtype=arr.dtype)


# ===========================================================================
# Legacy extrema detectors (kept for comparison)
# ===========================================================================

def get_min_max_idx(
    arr: Union[np.ndarray, torch.Tensor],
    axs: int = 1,
    pad: bool = True,
) -> Union[np.ndarray, torch.Tensor]:
    """Gradient-sign inversion extrema detection (min+max combined). No robustness."""
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
    """Gradient-sign inversion extrema detector split by type. No robustness."""
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
# F1 internals
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
    gt_bin = (gt_mask > 0) & valid
    rec_bin = (rec_mask > 0) & valid
    gt_win = _dilate_along_h(gt_bin.float(), kernel_size)
    rec_win = _dilate_along_h(rec_bin.float(), kernel_size)
    f1_per_level: List[float] = []
    confusion: List[Dict] = []
    gt_extrema_per_h: List[int] = []
    rec_extrema_per_h: List[int] = []
    for h in range(H):
        t = gt_bin[:, h, :]
        p = rec_bin[:, h, :]
        tw = gt_win[:, h, :]
        pw = rec_win[:, h, :]
        valid_h = valid[:, h, :]
        tp_prec = int((tw & p).sum().item())
        tp_rec = int((t & pw).sum().item())
        fp = int((~tw & p & valid_h).sum().item())
        fn = int((t & ~pw & valid_h).sum().item())
        tn = int((valid_h & ~t & ~p).sum().item())
        precision = tp_prec / (tp_prec + fp + 1e-12)
        recall = tp_rec / (tp_rec + fn + 1e-12)
        f1 = 2.0 * precision * recall / (precision + recall + 1e-12)
        f1_per_level.append(float(f1))
        confusion.append({"tp": tp_prec + tp_rec, "fp": fp, "fn": fn, "tn": tn})
        gt_extrema_per_h.append(int(t.sum().item()))
        rec_extrema_per_h.append(int(p.sum().item()))

    f1_arr = np.array(f1_per_level, dtype=np.float64)
    f1_mean = float(np.mean(f1_arr)) if f1_arr.size > 0 else float("nan")

    # Weighted mean: weight by GT extrema count, exclude levels with 0 GT extrema
    weights = np.array(gt_extrema_per_h, dtype=np.float64)
    total_weight = float(weights.sum())
    if total_weight > 0:
        f1_mean_weighted = float((f1_arr * weights).sum() / total_weight)
    else:
        f1_mean_weighted = float("nan")

    return {
        "f1_mean": f1_mean,
        "f1_mean_weighted_h": f1_mean_weighted,
        "f1_per_level": f1_per_level,
        "confusion_per_level": confusion,
        "gt_extrema_per_h": gt_extrema_per_h,
        "rec_extrema_per_h": rec_extrema_per_h,
    }


# ===========================================================================
# Core F1 engine (detailed, from original metrics.py)
# ===========================================================================
from typing import Any, Dict, List, Tuple
import math
import torch


def _safe_div(num: float, den: float) -> float:
    return float(num) / float(den) if den > 0 else 0.0


def _f1_from_pr(precision: float, recall: float) -> float:
    den = precision + recall
    return 2.0 * precision * recall / den if den > 0 else 0.0


def _match_profile_extrema_1d(
    gt_idx: torch.Tensor,
    pred_idx: torch.Tensor,
    tolerance: int,
) -> List[Tuple[int, int, int]]:
    """
    Ordered one-to-one matching between GT and predicted extrema indices.

    Parameters
    ----------
    gt_idx : 1D sorted tensor of GT extrema indices
    pred_idx : 1D sorted tensor of predicted extrema indices
    tolerance : maximum allowed |pred - gt|

    Returns
    -------
    matches : list of (gt_h, pred_h, abs_offset)
    """
    if gt_idx.numel() == 0 or pred_idx.numel() == 0:
        return []

    gt = [int(v) for v in gt_idx.tolist()]
    pred = [int(v) for v in pred_idx.tolist()]

    matches: List[Tuple[int, int, int]] = []
    i = 0
    j = 0

    while i < len(gt) and j < len(pred):
        g = gt[i]
        p = pred[j]

        if p < g - tolerance:
            j += 1
        elif p > g + tolerance:
            i += 1
        else:
            matches.append((g, p, abs(g - p)))
            i += 1
            j += 1

    return matches


def extrema_f1_along_h(
    gt_mask: torch.Tensor,
    pred_mask: torch.Tensor,
    tolerance: int = 2,
) -> Dict[str, Any]:
    """
    Event-based extrema score along H for tensors of shape (C, H, W).

    Parameters
    ----------
    gt_mask : torch.Tensor
        Binary/extrema mask of shape (C, H, W). Nonzero values are treated as extrema.
    pred_mask : torch.Tensor
        Binary/extrema mask of shape (C, H, W). Nonzero values are treated as extrema.
    tolerance : int
        A predicted extremum matches a GT extremum if |h_pred - h_gt| <= tolerance
        within the same (C, W) profile.

    Returns
    -------
    dict with:
      - precision, recall, f1
      - tp, fp, fn
      - gt_total, pred_total
      - gt_count_per_h, pred_count_per_h
      - matched_gt_per_h, matched_pred_per_h
      - recall_per_h, precision_per_h
      - mean_abs_offset, mean_abs_offset_per_h
    """
    if gt_mask.shape != pred_mask.shape:
        raise ValueError(f"Shape mismatch: {gt_mask.shape} vs {pred_mask.shape}")
    if gt_mask.ndim != 3:
        raise ValueError(f"Expected shape (C,H,W), got {gt_mask.shape}")
    if tolerance < 0:
        raise ValueError(f"tolerance must be >= 0, got {tolerance}")

    gt = (gt_mask > 0)
    pred = (pred_mask > 0)

    C, H, W = gt.shape

    gt_count_per_h = torch.zeros(H, dtype=torch.int64)
    pred_count_per_h = torch.zeros(H, dtype=torch.int64)
    matched_gt_per_h = torch.zeros(H, dtype=torch.int64)
    matched_pred_per_h = torch.zeros(H, dtype=torch.int64)

    offset_sum_per_h = torch.zeros(H, dtype=torch.float64)
    offset_num_per_h = torch.zeros(H, dtype=torch.int64)

    gt_total = 0
    pred_total = 0
    tp_total = 0
    total_abs_offset = 0.0

    for c in range(C):
        for w in range(W):
            gt_idx = torch.nonzero(gt[c, :, w], as_tuple=False).view(-1)
            pred_idx = torch.nonzero(pred[c, :, w], as_tuple=False).view(-1)

            if gt_idx.numel() > 0:
                gt_count_per_h.index_add_(
                    0,
                    gt_idx,
                    torch.ones_like(gt_idx, dtype=torch.int64),
                )
            if pred_idx.numel() > 0:
                pred_count_per_h.index_add_(
                    0,
                    pred_idx,
                    torch.ones_like(pred_idx, dtype=torch.int64),
                )

            gt_total += int(gt_idx.numel())
            pred_total += int(pred_idx.numel())

            matches = _match_profile_extrema_1d(gt_idx, pred_idx, tolerance=tolerance)
            tp_total += len(matches)

            for g_h, p_h, d in matches:
                matched_gt_per_h[g_h] += 1
                matched_pred_per_h[p_h] += 1
                offset_sum_per_h[g_h] += float(d)
                offset_num_per_h[g_h] += 1
                total_abs_offset += float(d)

    fp_total = pred_total - tp_total
    fn_total = gt_total - tp_total

    precision = _safe_div(tp_total, pred_total)
    recall = _safe_div(tp_total, gt_total)
    f1 = _f1_from_pr(precision, recall)

    recall_per_h: List[float] = []
    precision_per_h: List[float] = []
    mean_abs_offset_per_h: List[float] = []

    for h in range(H):
        gt_h = int(gt_count_per_h[h].item())
        pred_h = int(pred_count_per_h[h].item())
        matched_gt_h = int(matched_gt_per_h[h].item())
        matched_pred_h = int(matched_pred_per_h[h].item())

        recall_per_h.append(
            float(matched_gt_h) / float(gt_h) if gt_h > 0 else float("nan")
        )
        precision_per_h.append(
            float(matched_pred_h) / float(pred_h) if pred_h > 0 else float("nan")
        )
        mean_abs_offset_per_h.append(
            float(offset_sum_per_h[h].item()) / float(offset_num_per_h[h].item())
            if int(offset_num_per_h[h].item()) > 0 else float("nan")
        )

    mean_abs_offset = total_abs_offset / tp_total if tp_total > 0 else float("nan")

    return {
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "tp": int(tp_total),
        "fp": int(fp_total),
        "fn": int(fn_total),
        "gt_total": int(gt_total),
        "pred_total": int(pred_total),
        "tolerance": int(tolerance),
        "gt_count_per_h": [int(v) for v in gt_count_per_h.tolist()],
        "pred_count_per_h": [int(v) for v in pred_count_per_h.tolist()],
        "matched_gt_per_h": [int(v) for v in matched_gt_per_h.tolist()],
        "matched_pred_per_h": [int(v) for v in matched_pred_per_h.tolist()],
        "recall_per_h": recall_per_h,
        "precision_per_h": precision_per_h,
        "mean_abs_offset": float(mean_abs_offset),
        "mean_abs_offset_per_h": mean_abs_offset_per_h,
    } 
    
def _get_f1_score_torch_detailed(
    min_max_idx_truth: torch.Tensor,
    min_max_idx_ae: torch.Tensor,
    axs: int = -2,
    kernel_size: int = 10,
) -> Dict[str, Any]:
    """
    Tolerant F1 score with per-level breakdown, confusion matrices,
    GT/REC extrema counts per level, and a GT-extrema-weighted F1 mean.
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
    pred = (min_max_idx_ae > 0)

    truth_h_last = truth.movedim(axs, -1)
    pred_h_last = pred.movedim(axs, -1)
    h_size = truth_h_last.shape[-1]

    flat_truth = truth_h_last.reshape(-1, 1, h_size).to(torch.float32)
    flat_pred = pred_h_last.reshape(-1, 1, h_size).to(torch.float32)

    kernel = torch.ones(1, 1, kernel_size, device=min_max_idx_truth.device, dtype=torch.float32)
    pad = kernel_size // 2

    truth_expanded = F.conv1d(flat_truth, kernel, padding=pad)
    pred_expanded = F.conv1d(flat_pred, kernel, padding=pad)

    if kernel_size % 2 == 0:
        truth_expanded = truth_expanded[:, :, :h_size]
        pred_expanded = pred_expanded[:, :, :h_size]

    truth_win = (truth_expanded > 0).reshape(*truth_h_last.shape)
    pred_win = (pred_expanded > 0).reshape(*pred_h_last.shape)

    tp_map = (truth_win & pred_h_last)
    fp_map = (~truth_win & pred_h_last)
    fn_map = (truth_h_last & ~pred_win)

    def _f1_from_maps(tp_m, fp_m, fn_m, dim):
        tp_ = tp_m.sum(dim=dim).to(torch.float64).sum(dim=0)
        fp_ = fp_m.sum(dim=dim).to(torch.float64).sum(dim=0)
        fn_ = fn_m.sum(dim=dim).to(torch.float64).sum(dim=0)
        prec_den = tp_ + fp_
        rec_den = tp_ + fn_
        prec = torch.where(prec_den == 0, torch.zeros_like(tp_), tp_ / prec_den)
        rec = torch.where(rec_den == 0, torch.zeros_like(tp_), tp_ / rec_den)
        f1_den = prec + rec
        f1 = torch.where(f1_den == 0, torch.zeros_like(f1_den), 2.0 * prec * rec / f1_den)
        return f1, tp_, fp_, fn_

    f1_per_w, tp_w, fp_w, fn_w = _f1_from_maps(tp_map, fp_map, fn_map, dim=-1)
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

    total_per_h = int(tp_map.shape[0] * tp_map.shape[1])
    total_per_w = int(tp_map.shape[0] * tp_map.shape[2])

    gt_extrema_per_h = truth_h_last.sum(dim=(0, 1)).long()
    rec_extrema_per_h = pred_h_last.sum(dim=(0, 1)).long()

    gt_counts = [int(v) for v in gt_extrema_per_h.cpu().tolist()]
    rec_counts = [int(v) for v in rec_extrema_per_h.cpu().tolist()]

    weights = gt_extrema_per_h.to(torch.float64)
    total_weight = float(weights.sum().item())
    if total_weight > 0:
        f1_mean_weighted_h = float(
            (f1_per_h.to(torch.float64) * weights).sum().item() / total_weight
        )
    else:
        f1_mean_weighted_h = float("nan")

    return {
        "f1_per_w": [float(v) for v in f1_per_w.flatten().cpu().tolist()],
        "confusion_per_w": _confusion_list(tp_w, fp_w, fn_w, total_per_w),
        "f1_per_level": [float(v) for v in f1_per_h.flatten().cpu().tolist()],
        "f1_per_h": [float(v) for v in f1_per_h.flatten().cpu().tolist()],
        "confusion_per_level": _confusion_list(tp_h, fp_h, fn_h, total_per_h),
        "confusion_per_h": _confusion_list(tp_h, fp_h, fn_h, total_per_h),
        "f1_mean": float(f1_per_w.mean().item()) if f1_per_w.numel() > 0 else float("nan"),
        "f1_mean_h": float(f1_per_h.mean().item()) if f1_per_h.numel() > 0 else float("nan"),
        "gt_extrema_per_h": gt_counts,
        "rec_extrema_per_h": rec_counts,
        "f1_mean_weighted_h": f1_mean_weighted_h,
    }


# ===========================================================================
# Public F1 API
# ===========================================================================

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
        pred_onehot = pred_onehot.unsqueeze(0)
    if truth_onehot.ndim == 4:
        if truth_onehot.shape[0] != 1:
            raise ValueError("Batch size > 1 not supported; pass one sample at a time.")
        truth_onehot = truth_onehot.squeeze(0)
        pred_onehot = pred_onehot.squeeze(0)
    if truth_onehot.ndim != 3:
        raise ValueError(f"Expected (C,H,W) tensor, got shape {truth_onehot.shape}")
    out = _get_f1_score_torch_detailed(
        min_max_idx_truth=truth_onehot,
        min_max_idx_ae=pred_onehot,
        axs=-2,
        kernel_size=kernel_size,
    )
    return float(out["f1_mean"])

def f1_score_v2(
    truth_onehot: torch.Tensor,
    pred_onehot: torch.Tensor,
    kernel_size: int = 5,
) -> float:
    """
    Event-based tolerant F1 for extrema along H.

    Parameters
    ----------
    truth_onehot : torch.Tensor
        Extrema mask, shape (C,H,W), or (H,W), or (1,C,H,W).
    pred_onehot : torch.Tensor
        Extrema mask, same shape as truth_onehot.
    kernel_size : int
        Matching window size along H. Converted to tolerance = kernel_size // 2.

    Returns
    -------
    float
        Global event-based F1 score.
    """
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

def f1_and_confusion_per_level(
    truth_onehot: torch.Tensor,
    pred_onehot: torch.Tensor,
    kernel_size: int = 5,
) -> dict[str, Any]:
    """Tolerant F1 + confusion matrix + extrema counts per H level."""
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
) -> dict[str, Any]:
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
    pred_onehot = _prep(pred_onehot, "pred_onehot")

    if truth_min_onehot.shape != pred_onehot.shape:
        raise ValueError(
            f"Shape mismatch: truth_min {truth_min_onehot.shape} vs pred {pred_onehot.shape}"
        )
    if truth_max_onehot.shape != pred_onehot.shape:
        raise ValueError(
            f"Shape mismatch: truth_max {truth_max_onehot.shape} vs pred {pred_onehot.shape}"
        )

    res_min = _get_f1_score_torch_detailed(
        min_max_idx_truth=truth_min_onehot,
        min_max_idx_ae=pred_onehot,
        axs=-2,
        kernel_size=kernel_size,
    )
    res_max = _get_f1_score_torch_detailed(
        min_max_idx_truth=truth_max_onehot,
        min_max_idx_ae=pred_onehot,
        axs=-2,
        kernel_size=kernel_size,
    )

    def _nanmean2(a: float, b: float) -> float:
        vals = [v for v in (a, b) if not (v != v)]  # exclude NaN
        return float(np.mean(vals)) if vals else float("nan")

    nan = float("nan")
    return {
        "min": res_min,
        "max": res_max,
        "f1_min_mean":         res_min.get("f1_mean",            nan),
        "f1_max_mean":         res_max.get("f1_mean",            nan),
        "f1_combined_mean":    _nanmean2(res_min.get("f1_mean",  nan), res_max.get("f1_mean",  nan)),
        "f1_min_mean_h":       res_min.get("f1_mean_h",          nan),
        "f1_max_mean_h":       res_max.get("f1_mean_h",          nan),
        "f1_combined_mean_h":  _nanmean2(res_min.get("f1_mean_h",nan), res_max.get("f1_mean_h",nan)),
        "f1_min_weighted_h":      res_min.get("f1_mean_weighted_h", nan),
        "f1_max_weighted_h":      res_max.get("f1_mean_weighted_h", nan),
        "f1_combined_weighted_h": _nanmean2(
            res_min.get("f1_mean_weighted_h", nan),
            res_max.get("f1_mean_weighted_h", nan),
        ),
    }

def f1_and_match_stats_per_level(
    truth_onehot: torch.Tensor,
    pred_onehot: torch.Tensor,
    kernel_size: int = 5,
) -> dict[str, Any]:
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

    return event_match_stats_per_level(
        truth_onehot=truth_onehot,
        pred_onehot=pred_onehot,
        kernel_size=kernel_size,
        axs=-2,
    )
    
def f1_extrema(
    gt: torch.Tensor,
    rec: torch.Tensor,
    kind: str = "both",
    kernel_size: int = 5,
    filtered_rec: Optional[torch.Tensor] = None,
    grad_eps: Union[float, str] = "auto",
    smooth_window: int = 1,
    min_separation: int = 1,
    fill_zero_signs: bool = True,
) -> dict[str, Any]:
    """Tolerant F1 using detect_extrema (robust detector).

    When grad_eps="auto", the threshold is always derived from GT and applied
    to both GT and reconstruction for consistent comparison.
    """
    gt = _ensure_chw(gt)
    rec = _ensure_chw(rec)
    resolved_eps = _auto_grad_eps(gt) if grad_eps == "auto" else float(grad_eps)
    extrema_kwargs = dict(
        kind=kind, grad_eps=resolved_eps, smooth_window=smooth_window,
        min_separation=min_separation, fill_zero_signs=fill_zero_signs,
    )
    valid = _nan_mask(gt)
    gt_mask = detect_extrema_v2(gt, **extrema_kwargs)
    rec_mask = detect_extrema_v2(rec, **extrema_kwargs)
    result = _f1_from_masks(gt_mask, rec_mask, valid, kernel_size)
    if filtered_rec is not None:
        filtered_rec = _ensure_chw(filtered_rec)
        rec_mask_f = detect_extrema_v2(filtered_rec, **extrema_kwargs)
        res_f = _f1_from_masks(gt_mask, rec_mask_f, valid, kernel_size)
        result["f1_mean_filtered"] = res_f["f1_mean"]
        result["f1_per_level_filtered"] = res_f["f1_per_level"]
        result["confusion_filtered"] = res_f["confusion_per_level"]
    return result


def f1_extrema_prominence(
    gt: torch.Tensor,
    rec: torch.Tensor,
    kind: str = "both",
    kernel_size: int = 5,
    prominence_fraction: float = 0.1,
    min_width: int = 1,
    min_distance: int = 3,
    filtered_rec: Optional[torch.Tensor] = None,
) -> Dict:
    """Tolerant F1 using detect_extrema_prominence (find_peaks-based)."""
    gt = _ensure_chw(gt)
    rec = _ensure_chw(rec)
    valid = _nan_mask(gt)
    prom_kwargs = dict(
        kind=kind,
        prominence_fraction=prominence_fraction,
        min_width=min_width,
        min_distance=min_distance,
    )
    gt_mask = detect_extrema_prominence(gt, **prom_kwargs)
    rec_mask = detect_extrema_prominence(rec, **prom_kwargs)
    result = _f1_from_masks(gt_mask, rec_mask, valid, kernel_size)
    # Prefix keys to distinguish from gradient-based results
    result = {f"prom_{k}": v for k, v in result.items()}
    if filtered_rec is not None:
        filtered_rec = _ensure_chw(filtered_rec)
        rec_mask_f = detect_extrema_prominence(filtered_rec, **prom_kwargs)
        res_f = _f1_from_masks(gt_mask, rec_mask_f, valid, kernel_size)
        result["prom_f1_mean_filtered"] = res_f["f1_mean"]
        result["prom_f1_per_level_filtered"] = res_f["f1_per_level"]
    return result


# ===========================================================================
# Report writer (extrema-focused)
# ===========================================================================

def write_minmax_report_txt(
    path: str,
    rows: List[Dict[str, Any]],
    header: Optional[Dict[str, Any]] = None,
) -> None:
    """Write a readable .txt report focused on extrema detection and F1 metrics."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    if not rows:
        with path.open("w") as f:
            f.write("# Extrema metrics report\n\n(no rows)\n")
        return

    keys = list(rows[0].keys())

    def _agg(vals: List[float]) -> Tuple[float, float, float, float]:
        arr = np.asarray(vals, dtype=np.float64)
        arr = arr[np.isfinite(arr)]
        if arr.size == 0:
            return (np.nan, np.nan, np.nan, np.nan)
        return (float(arr.mean()), float(arr.std(ddof=0)), float(arr.min()), float(arr.max()))

    def _mse_to_rmse_aggregate_key_vals(k: str, vals: List[float]) -> Tuple[str, List[float]]:
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
        f.write("# Extrema detection & F1 metrics report\n\n")
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

        # --- per-level F1 ---
        level_keys = [k for k in keys if k.endswith("_levels") and "F1" in k]
        if level_keys:
            f.write("\n## F1 per H level\n")
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
                for idx, vals in enumerate(per_level_values):
                    if not vals:
                        f.write(f"  h_{idx}: mean=nan, std=nan, min=nan, max=nan, n=0\n")
                        continue
                    mean, std, vmin, vmax = _agg(vals)
                    f.write(
                        f"  h_{idx}: mean={mean:.6g}, std={std:.6g}, "
                        f"min={vmin:.6g}, max={vmax:.6g}, n={len(vals)}\n"
                    )

        # --- confusion matrices ---
        cm_keys = [k for k in keys if k.startswith("CM_") and k.endswith("_levels")]
        if cm_keys:
            f.write("\n## F1 confusion matrix per H level\n")
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
                for idx, cm in enumerate(per_level_cm):
                    tp, fp, fn, tn = cm["tp"], cm["fp"], cm["fn"], cm["tn"]
                    precision = tp / (tp + fp + 1e-12)
                    recall = tp / (tp + fn + 1e-12)
                    f1 = 2.0 * precision * recall / (precision + recall + 1e-12)
                    f.write(
                        f"  h_{idx}: tp={tp}, fp={fp}, fn={fn}, tn={tn}, "
                        f"precision={precision:.6g}, recall={recall:.6g}, f1={f1:.6g}\n"
                    )

        # --- extrema counts ---
        extrema_pairs = [
            ("GT_extrema_min_levels", "REC_extrema_min_levels", "min"),
            ("GT_extrema_max_levels", "REC_extrema_max_levels", "max"),
            ("GT_extrema_both_levels", "REC_extrema_both_levels", "both"),
            ("GT_extrema_prom_min_levels", "REC_extrema_prom_min_levels", "prom_min"),
            ("GT_extrema_prom_max_levels", "REC_extrema_prom_max_levels", "prom_max"),
            ("GT_extrema_prom_both_levels", "REC_extrema_prom_both_levels", "prom_both"),
        ]
        for gt_key, rec_key, kind in extrema_pairs:
            if gt_key not in keys:
                continue
            f.write(f"\n## Extrema counts per H level ({kind})\n")
            per_level_gt: List[List[int]] = []
            per_level_rec: List[List[int]] = []
            for row in rows:
                gt_arr = _parse_numeric_list(row.get(gt_key))
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
            n_empty_gt = sum(1 for vals in per_level_gt if sum(vals) == 0)
            f.write(f"  levels_with_zero_gt_extrema: {n_empty_gt} / {len(per_level_gt)}\n")
            for idx, (gt_vals, rec_vals) in enumerate(zip(per_level_gt, per_level_rec)):
                gt_total = sum(gt_vals)
                rec_total = sum(rec_vals)
                gt_mean = float(gt_total) / len(gt_vals) if gt_vals else float("nan")
                rec_mean = float(rec_total) / len(rec_vals) if rec_vals else float("nan")
                flag = "  *** NO GT EXTREMA ***" if gt_total == 0 else ""
                f.write(
                    f"  h_{idx}: gt_total={gt_total}, gt_mean={gt_mean:.2f}, "
                    f"rec_total={rec_total}, rec_mean={rec_mean:.2f}{flag}\n"
                )

        # --- Wasserstein per H ---
        wass_keys = [k for k in keys if "wasserstein" in k.lower() and k.endswith("_per_h")]
        if wass_keys:
            f.write("\n## Wasserstein distance per H level\n")
            for k in wass_keys:
                per_level_values_w: List[List[float]] = []
                for row in rows:
                    arr = _parse_numeric_list(row.get(k))
                    if arr is None:
                        continue
                    while len(per_level_values_w) < int(arr.size):
                        per_level_values_w.append([])
                    for idx, val in enumerate(arr.tolist()):
                        if np.isfinite(val):
                            per_level_values_w[idx].append(float(val))
                if not per_level_values_w:
                    continue
                f.write(f"{k}\n")
                for idx, vals in enumerate(per_level_values_w):
                    if not vals:
                        f.write(f"  h_{idx}: mean=nan, n=0\n")
                        continue
                    mean, std, vmin, vmax = _agg(vals)
                    f.write(
                        f"  h_{idx}: mean={mean:.6g}, std={std:.6g}, "
                        f"min={vmin:.6g}, max={vmax:.6g}, n={len(vals)}\n"
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