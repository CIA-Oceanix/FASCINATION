"""
Residual post-decoder experiment for MLIC++ extrema F1.

Experiment
----------
1. Train a normal MLIC++ checkpoint separately with ``playground/train.py``.
2. Freeze that checkpoint.
3. For each training field, decode ``x_hat`` and learn the residual
   ``r = x - x_hat`` with one of:

   - deterministic residual U-Net
   - residual DDPM-style diffusion model
   - residual flow matching model

4. Compare frozen MLIC++ vs MLIC++ + residual model using the existing robust
   extrema detector and tolerant F1 metric.

The residual models operate in the same normalized tensor space as MLIC++.
F1 is computed after denormalization, so it matches the physical-field
evaluation used in the other playground scripts.

Example
-------
python playground/train_residual_f1_experiment.py \\
  --dataset /path/to/slices_no_nan/lon \\
  --var-name celerity \\
  --slice-dim lon --use-slices --normalize-mode minmax01 \\
  --codec-checkpoint ./experiments/mlic_rd/checkpoints/checkpoint_best_loss.pth.tar \\
  --experiment residual_f1_lon \\
  --methods unet diffusion flow \\
  --epochs 50 --batch-size 4 --test-batch-size 1
"""

from __future__ import annotations


import os 
import sys

from numpy import astype

running_path = "/Odyssey/private/o23gauvr/code/"
os.chdir(running_path)
sys.path.insert(0,running_path)




import argparse
import csv
import logging
import math
import random
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import xarray as xr
from torch.utils.data import DataLoader

try:
    from tqdm.auto import tqdm
except Exception:  # pragma: no cover - tqdm is optional in the run env.
    tqdm = None

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from FASCINATION.experiments.residual_flow_chapron.src.NetCDFFolder import NetCDFFolder2D
from MLIC.MLIC.config.config import model_config
from MLIC.MLIC.models import *
from FASCINATION.experiments.residual_flow_chapron.src.metrics import (
    compute_dtw_and_extrema_metrics,
    compute_extrema_dtw_metrics,
    denorm_bchw,
    detect_extrema_prominence,
    lowpass_filter_torch_along_axis,
)
from FASCINATION.experiments.residual_flow_chapron.src.metrics_minmax import detect_extrema, extrema_wasserstein, f1_and_confusion_per_level
from FASCINATION.experiments.residual_flow_chapron.src.metrics_minmax_v2 import prominence_matching_metrics


METHODS = ("unet", "diffusion", "flow")


def setup_run_logger(log_path: Path) -> logging.Logger:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("residual_f1")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")

    file_handler = logging.FileHandler(log_path)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(console_handler)
    return logger


def progress_iter(iterable, total: Optional[int], desc: str, args: argparse.Namespace):
    if bool(getattr(args, "tqdm", False)) and tqdm is not None:
        return tqdm(iterable, total=total, desc=desc, dynamic_ncols=True, leave=False)
    return iterable


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Train residual models on frozen MLIC++ reconstructions and compare extrema F1."
    )
    p.add_argument("--dataset", required=True, help="Root with train/val/test NetCDF split folders.")
    p.add_argument("--var-name", "--var_name", dest="var_name", default="celerity")
    p.add_argument("--codec-checkpoint", required=True, help="Frozen MLIC++ checkpoint.")
    p.add_argument("--experiment", default="mlic_residual_f1")
    p.add_argument("--log-file", default=None,
                   help="Optional log path. Default: experiments/<experiment>/train_residual_f1.log")
    p.add_argument("--methods", nargs="+", default=list(METHODS), choices=METHODS)
    p.add_argument("--cascade-after-unet", action="store_true",
                   help="After selecting the best UNet residual alpha, train diffusion/flow on "
                        "the remaining error x - (x_hat + alpha_unet * r_unet).")
    p.add_argument("--cascade-methods", nargs="+", default=["diffusion", "flow"],
                   choices=["diffusion", "flow"],
                   help="Second-stage residual methods to train when --cascade-after-unet is set.")
    p.add_argument("--eval-split", default="test", choices=["train", "val", "test"])

    p.add_argument("--normalize-mode", default="minmax01",
                   choices=["minmax01", "minmax", "zscore", "standard", "robust", "iqr",
                            "median_iqr", "zscore_squash", "zscore_squash_q1", "none"])
    p.add_argument("--use-slices", action="store_true")
    p.add_argument("--slice-dim", "--slice_dim", dest="slice_dim",
                   default="lat", choices=["lat", "lon", "z"])
    p.add_argument("--slice-stride", "--slice_stride", dest="slice_stride", type=int, default=3)
    p.add_argument("--depth-z-name", default="z")
    p.add_argument("--clip-tol", type=float, default=None)
    p.add_argument("--clip-quantiles", type=float, nargs=2, default=None)
    p.add_argument("--soft-clip-tanh", action="store_true")
    p.add_argument("--soft-clip-scale", type=float, default=None)
    p.add_argument("--squash-lower-quantile", type=float, default=1.0)

    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--force-retrain", action="store_true",
                   help="Ignore existing residual checkpoints and retrain methods from scratch.")
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--test-batch-size", type=int, default=1)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--tqdm", action="store_true",
                   help="Show tqdm progress bars in addition to logger progress lines.")
    p.add_argument("--scale-log-interval", type=int, default=4,
                   help="Log one residual-scale progress line every N batches.")
    p.add_argument("--train-log-interval", type=int, default=25,
                   help="Log one training progress line every N batches.")
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--weight-decay", type=float, default=1e-5)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=483)
    p.add_argument("--gpu-id", type=int, default=0)
    p.add_argument("--no-cuda", action="store_true")

    p.add_argument("--base-channels", type=int, default=64)
    p.add_argument("--levels", type=int, default=3)
    p.add_argument("--time-emb-dim", type=int, default=128)
    p.add_argument("--diffusion-steps", type=int, default=1000)
    p.add_argument("--sample-steps", type=int, default=50)
    p.add_argument("--ensemble-size", type=int, default=4,
                   help="Number of stochastic residual samples for final diffusion/flow evaluation.")
    p.add_argument("--epoch-ensemble-size", type=int, default=1,
                   help="Number of samples used during per-epoch validation.")
    p.add_argument("--residual-weights", type=float, nargs="+",
                   default=[0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5],
                   help="Alpha sweep for x_hat = x_tilde + alpha * residual.")
    p.add_argument("--residual-scale", default="auto",
                   help="Train target is residual / scale; sampled residual is multiplied back. "
                        "Use 'auto' to estimate the frozen-codec residual std.")
    p.add_argument("--residual-scale-batches", type=int, default=32,
                   help="Number of train batches used when --residual-scale auto.")
    p.add_argument("--clamp-output", action="store_true",
                   help="Clamp corrected normalized output to [0,1]. Useful for minmax01 codecs.")

    p.add_argument("--selection-metric", default="f1_min_max_combined_mean",
                   choices=[
                       "f1_min_max_combined_mean",
                       "f1_min_max_combined_mean_weighted_h",
                       "f1_min_mean", "f1_max_mean", "f1_combined_mean",
                       "f1_combine_mean",
                       "f1_min_mean_weighted_h", "f1_max_mean_weighted_h",
                       "f1_combined_mean_weighted_h",
                       "f1_combine_mean_weighted_h",
                   ],
                   help="Metric used to keep the best residual checkpoint.")
    p.add_argument("--optimize-extrema-kinds", nargs="+", default=["all"],
                   choices=["min", "max", "both", "combined", "combine", "all"],
                   help="GT extrema kinds emphasized in the residual training loss.")
    p.add_argument("--extrema-loss-weight", type=float, default=1.0,
                   help="Extra residual-loss weight on GT extrema locations. Set 0 to disable.")
    p.add_argument("--extrema-dilation-radius", type=int, default=2,
                   help="Dilate GT extrema mask by this many pixels before weighting residual loss.")
    p.add_argument("--f1-kernel-size", type=int, default=5)
    p.add_argument("--grad-eps", default="auto")
    p.add_argument("--smooth-window", type=int, default=1)
    p.add_argument("--min-separation", type=int, default=3)
    p.add_argument("--max-eval-batches", type=int, default=None)
    p.add_argument("--eval-log-interval", type=int, default=25,
                   help="Log one progress line every N evaluation batches when a logger is available.")
    p.add_argument("--full-metrics", dest="full_metrics", action="store_true", default=True,
                   help="Compute prominence F1, filtered F1, DTW, extrema DTW, and Wasserstein in final residual metrics.")
    p.add_argument("--no-full-metrics", dest="full_metrics", action="store_false",
                   help="Disable the extra final residual metrics.")
    p.add_argument("--full-metric-member-spread", dest="full_metric_member_spread",
                   action="store_true", default=True,
                   help="For stochastic residuals, compute member std/range for full metrics too.")
    p.add_argument("--no-full-metric-member-spread", dest="full_metric_member_spread",
                   action="store_false",
                   help="Only compute member std/range for the core residual F1/RMSE metrics.")
    p.add_argument("--dtw-window", type=int, default=3)
    p.add_argument("--dtw-metrics", dest="dtw_metrics", action="store_true", default=True,
                   help="Compute full-profile DTW when requested and extrema-DTW in final residual metrics.")
    p.add_argument("--no-dtw-metrics", dest="dtw_metrics", action="store_false",
                   help="Skip DTW/extrema-DTW while keeping the other full metrics such as prominence F1.")
    p.add_argument("--compute-profile-dtw", action="store_true",
                   help="Also compute full-profile DTW in final residual metrics. Slow; extrema DTW is kept without this flag.")
    p.add_argument("--extrema-prominence", type=float, default=0.2)
    p.add_argument("--extrema-prominence-ratio", type=float, default=0.1)
    p.add_argument("--wasserstein-metrics", dest="wasserstein_metrics",
                   action="store_true", default=True)
    p.add_argument("--no-wasserstein-metrics", dest="wasserstein_metrics",
                   action="store_false")

    p.add_argument("--inference-only", action="store_true",
                   help="Load existing residual checkpoints, run evaluation/export, and skip training.")
    p.add_argument("--export-nc", action="store_true",
                   help="Write NetCDF inference samples after loading/training residual methods.")
    p.add_argument("--infer-split", default=None, choices=["train", "val", "test"],
                   help="Split used for NetCDF export. Default: --eval-split.")
    p.add_argument("--max-infer-samples", type=int, default=16,
                   help="Maximum number of samples to export to NetCDF.")
    p.add_argument("--random-infer-samples", action="store_true",
                   help="Export a reproducible random subset instead of the first samples.")
    p.add_argument("--export-log-interval", type=int, default=1,
                   help="Log one export progress line every N saved NetCDF files.")
    p.add_argument("--save-ensemble-members", action="store_true",
                   help="Also save all stochastic ensemble members in each NetCDF file.")

    return p.parse_args()


def normalize_extrema_kind(kind: str) -> str:
    return "both" if kind in ("combined", "combine") else kind


def normalize_args(args: argparse.Namespace) -> argparse.Namespace:
    args.optimize_extrema_kinds = [
        "combined" if kind == "combine" else kind
        for kind in args.optimize_extrema_kinds
    ]
    metric_aliases = {
        "f1_combine_mean": "f1_combined_mean",
        "f1_combine_mean_weighted_h": "f1_combined_mean_weighted_h",
    }
    args.selection_metric = metric_aliases.get(args.selection_metric, args.selection_metric)
    return args


def clip_percentiles_from_args(args: argparse.Namespace) -> Optional[Tuple[float, float]]:
    if args.clip_quantiles is not None:
        qmin, qmax = float(args.clip_quantiles[0]), float(args.clip_quantiles[1])
        if not (0.0 <= qmin < qmax <= 100.0):
            raise ValueError("--clip-quantiles must satisfy 0 <= QMIN < QMAX <= 100")
        return qmin, qmax
    if args.clip_tol is not None:
        tol = float(args.clip_tol)
        if not (0.0 <= tol < 50.0):
            raise ValueError("--clip-tol must be in [0, 50)")
        return tol, 100.0 - tol
    return None


def make_dataset(args: argparse.Namespace, split: str, return_meta: bool = False) -> NetCDFFolder2D:
    normalize_mode = None if args.normalize_mode == "none" else args.normalize_mode
    lower_squash_q = (
        args.squash_lower_quantile
        if normalize_mode in ("zscore_squash", "zscore_squash_q1")
        else None
    )
    return NetCDFFolder2D(
        root=args.dataset,
        var_name=args.var_name,
        split=split,
        rgb=not args.use_slices,
        slice_dim=args.slice_dim,
        slice_mode="random",
        slice_stride=args.slice_stride,
        select_time=None,
        nan_value=0.0,
        fill_nan_for_nn=-1.0,
        normalize=normalize_mode,
        clip_percentiles=clip_percentiles_from_args(args),
        soft_clip_tanh=args.soft_clip_tanh,
        soft_clip_scale=args.soft_clip_scale,
        lower_squash_percentile=lower_squash_q,
        patch_size=157 if split == "train" else None,
        return_meta=return_meta,
        z_coord_name=args.depth_z_name,
    )


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def strip_module_prefix(state: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    if not any(k.startswith("module.") for k in state):
        return state
    return {k.removeprefix("module."): v for k, v in state.items()}


def load_codec(checkpoint_path: str, device: torch.device) -> MLICPlusPlus:
    config = model_config()
    config.use_new_gs_lrp = False
    codec = MLICPlusPlus(config=config).to(device)
    ckpt = torch.load(checkpoint_path, map_location="cpu")
    state = ckpt.get("state_dict", ckpt)
    state = strip_module_prefix(state)
    try:
        codec.load_state_dict(state, strict=True)
    except TypeError:
        codec.load_state_dict(state)
    codec.eval()
    for p in codec.parameters():
        p.requires_grad_(False)
    return codec


@torch.no_grad()
def frozen_decode(codec: nn.Module, x: torch.Tensor) -> torch.Tensor:
    h, w = x.shape[-2:]
    pad_h = (64 - (h % 64)) % 64
    pad_w = (64 - (w % 64)) % 64
    x_in = F.pad(x, (0, pad_w, 0, pad_h), mode="reflect") if (pad_h or pad_w) else x
    out = codec(x_in)
    return out["x_hat"][:, :, :h, :w].detach()


def batch_item(batch):
    """Return TrainingItem from either item or (item, meta) DataLoader batches."""
    if hasattr(batch, "input") and hasattr(batch, "valid_mask"):
        return batch
    if isinstance(batch, (tuple, list)) and len(batch) > 0:
        first = batch[0]
        if hasattr(first, "input") and hasattr(first, "valid_mask"):
            return first
    raise TypeError(f"Unexpected batch type: {type(batch)!r}")


@torch.no_grad()
def estimate_residual_scale(
    codec: nn.Module,
    loader: DataLoader,
    device: torch.device,
    max_batches: int,
    args: argparse.Namespace,
    logger: Optional[logging.Logger] = None,
) -> float:
    vals = []
    limit = max(1, int(max_batches))
    total_batches = min(len(loader), limit)
    log_interval = max(0, int(getattr(args, "scale_log_interval", 0) or 0))
    if logger is not None:
        logger.info("[scale] estimating residual scale batches=%d", total_batches)
    for bi, batch in enumerate(progress_iter(loader, total=total_batches, desc="scale", args=args)):
        if bi >= limit:
            break
        item = batch_item(batch)
        x = item.input.to(device, non_blocking=True)
        mask = item.valid_mask.to(device, non_blocking=True).to(dtype=torch.bool)
        x_hat = frozen_decode(codec, x)
        residual = x - x_hat
        if int(mask.sum().item()) > 0:
            vals.append(residual[mask].detach().float().cpu())
        done_batches = bi + 1
        if logger is not None and log_interval > 0:
            if done_batches % log_interval == 0 or done_batches >= total_batches:
                logger.info("[scale] batch %d/%d collected_chunks=%d", done_batches, total_batches, len(vals))
    if not vals:
        if logger is not None:
            logger.warning("[scale] no valid residual values collected; using scale=1")
        return 1.0
    all_vals = torch.cat(vals)
    scale = float(all_vals.std(unbiased=False).item())
    rms = float(torch.sqrt((all_vals * all_vals).mean()).item())
    out = max(scale, rms, 1e-6)
    if logger is not None:
        logger.info("[scale] done std=%.8g rms=%.8g selected=%.8g n=%d", scale, rms, out, int(all_vals.numel()))
    return out


def sinusoidal_embedding(t: torch.Tensor, dim: int) -> torch.Tensor:
    half = dim // 2
    freqs = torch.exp(
        torch.arange(half, device=t.device, dtype=t.dtype)
        * (-math.log(10000.0) / max(half - 1, 1))
    )
    args = t[:, None] * freqs[None, :]
    emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
    if dim % 2:
        emb = F.pad(emb, (0, 1))
    return emb


class ResBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, time_dim: int = 0):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.norm1 = nn.GroupNorm(min(8, out_ch), out_ch)
        self.norm2 = nn.GroupNorm(min(8, out_ch), out_ch)
        self.skip = nn.Identity() if in_ch == out_ch else nn.Conv2d(in_ch, out_ch, 1)
        self.time = nn.Linear(time_dim, out_ch) if time_dim > 0 else None

    def forward(self, x: torch.Tensor, temb: Optional[torch.Tensor] = None) -> torch.Tensor:
        h = F.silu(self.norm1(self.conv1(x)))
        if self.time is not None and temb is not None:
            h = h + self.time(temb)[:, :, None, None]
        h = self.norm2(self.conv2(h))
        return F.silu(h + self.skip(x))


class ResidualUNet(nn.Module):
    def __init__(
        self,
        channels: int,
        cond_channels: int,
        base_channels: int = 64,
        levels: int = 3,
        time_emb_dim: int = 0,
    ):
        super().__init__()
        self.time_emb_dim = int(time_emb_dim)
        in_ch = channels + cond_channels
        self.time_mlp = None
        if self.time_emb_dim > 0:
            self.time_mlp = nn.Sequential(
                nn.Linear(self.time_emb_dim, self.time_emb_dim),
                nn.SiLU(),
                nn.Linear(self.time_emb_dim, self.time_emb_dim),
            )

        self.downs = nn.ModuleList()
        self.pools = nn.ModuleList()
        ch = in_ch
        skip_channels: List[int] = []
        for i in range(levels):
            out_ch = base_channels * (2 ** i)
            self.downs.append(ResBlock(ch, out_ch, self.time_emb_dim))
            self.pools.append(nn.AvgPool2d(2))
            skip_channels.append(out_ch)
            ch = out_ch

        self.mid = ResBlock(ch, ch * 2, self.time_emb_dim)
        ch = ch * 2

        self.ups = nn.ModuleList()
        for skip_ch in reversed(skip_channels):
            self.ups.append(nn.ModuleDict({
                "up": nn.ConvTranspose2d(ch, skip_ch, 2, stride=2),
                "block": ResBlock(skip_ch * 2, skip_ch, self.time_emb_dim),
            }))
            ch = skip_ch

        self.out = nn.Conv2d(ch, channels, 1)

    def forward(
        self,
        x: torch.Tensor,
        cond: torch.Tensor,
        t: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        temb = None
        if self.time_mlp is not None:
            if t is None:
                t = torch.zeros(x.shape[0], device=x.device, dtype=x.dtype)
            temb = self.time_mlp(sinusoidal_embedding(t.to(dtype=x.dtype), self.time_emb_dim))

        h = torch.cat([x, cond], dim=1)
        skips = []
        for block, pool in zip(self.downs, self.pools):
            h = block(h, temb)
            skips.append(h)
            h = pool(h)
        h = self.mid(h, temb)
        for layer, skip in zip(self.ups, reversed(skips)):
            h = layer["up"](h)
            if h.shape[-2:] != skip.shape[-2:]:
                h = F.interpolate(h, size=skip.shape[-2:], mode="bilinear", align_corners=False)
            h = layer["block"](torch.cat([h, skip], dim=1), temb)
        return self.out(h)


class DiffusionSchedule:
    def __init__(self, steps: int, device: torch.device):
        beta = torch.linspace(1e-4, 2e-2, steps, device=device)
        alpha = 1.0 - beta
        alpha_bar = torch.cumprod(alpha, dim=0)
        self.steps = int(steps)
        self.beta = beta
        self.alpha = alpha
        self.alpha_bar = alpha_bar

    def gather(self, values: torch.Tensor, idx: torch.Tensor, like: torch.Tensor) -> torch.Tensor:
        return values[idx].view(like.shape[0], 1, 1, 1)


def masked_mse(
    pred: torch.Tensor,
    target: torch.Tensor,
    mask: Optional[torch.Tensor],
    weight: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    if mask is None and weight is None:
        return F.mse_loss(pred, target)
    if mask is None:
        m = torch.ones_like(pred)
    else:
        m = mask.to(dtype=pred.dtype, device=pred.device)
        while m.ndim < pred.ndim:
            m = m.unsqueeze(1)
    if weight is not None:
        w = weight.to(dtype=pred.dtype, device=pred.device)
        while w.ndim < pred.ndim:
            w = w.unsqueeze(1)
        m = m * w
    denom = m.sum().clamp(min=1.0)
    return ((pred - target) ** 2 * m).sum() / denom


def _loss_extrema_kinds(args: argparse.Namespace) -> List[str]:
    requested = set(args.optimize_extrema_kinds)
    if "all" in requested or "both" in requested or "combined" in requested:
        return ["min", "max"]
    return sorted(requested)


@torch.no_grad()
def build_extrema_loss_weight(
    gt_norm: torch.Tensor,
    valid_mask: torch.Tensor,
    dataset: NetCDFFolder2D,
    args: argparse.Namespace,
) -> torch.Tensor:
    """Return per-pixel residual-loss weights focused on GT min/max events."""
    base = torch.ones_like(gt_norm)
    if float(args.extrema_loss_weight) <= 0.0:
        return base

    norm_mode = None if args.normalize_mode == "none" else args.normalize_mode
    gt_phys = denorm_bchw(gt_norm, dataset.norm_stats, norm_mode)
    event = torch.zeros_like(gt_norm)
    kinds = _loss_extrema_kinds(args)
    for i in range(gt_phys.shape[0]):
        gt = gt_phys[i]
        valid = valid_mask[i].to(dtype=torch.bool, device=gt.device)
        gt = torch.where(valid, gt, torch.full_like(gt, float("nan")))
        sample_event = torch.zeros_like(gt)
        for kind in kinds:
            sample_event = torch.maximum(
                sample_event,
                detect_extrema(
                    gt,
                    kind=kind,
                    grad_eps=args.grad_eps,
                    smooth_window=args.smooth_window,
                    min_separation=args.min_separation,
                ),
            )
        event[i] = sample_event

    radius = int(args.extrema_dilation_radius)
    if radius > 0:
        k = 2 * radius + 1
        event = F.max_pool2d(event, kernel_size=k, stride=1, padding=radius)
    event = event * valid_mask.to(dtype=event.dtype, device=event.device)
    return base + float(args.extrema_loss_weight) * event


def train_epoch(
    method: str,
    model: ResidualUNet,
    codec: nn.Module,
    loader: DataLoader,
    dataset: NetCDFFolder2D,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    args: argparse.Namespace,
    schedule: Optional[DiffusionSchedule],
    base_model: Optional[ResidualUNet] = None,
    base_residual_weight: float = 0.0,
    logger: Optional[logging.Logger] = None,
    progress_label: Optional[str] = None,
) -> float:
    model.train()
    total = 0.0
    n = 0
    total_batches = len(loader)
    label = progress_label or f"train:{method}"
    log_interval = max(0, int(getattr(args, "train_log_interval", 0) or 0))
    for bi, batch in enumerate(progress_iter(loader, total=total_batches, desc=label, args=args)):
        item = batch_item(batch)
        x = item.input.to(device, non_blocking=True)
        mask = item.valid_mask.to(device, non_blocking=True)
        with torch.no_grad():
            x_hat = frozen_decode(codec, x)
            base_hat = x_hat
            if base_model is not None and float(base_residual_weight) != 0.0:
                base_residual = sample_residual("unet", base_model, x_hat, args, schedule)
                base_hat = x_hat + float(base_residual_weight) * base_residual
                if args.clamp_output:
                    base_hat = base_hat.clamp(0.0, 1.0)
            residual = (x - base_hat) / float(args.residual_scale)
            loss_weight = build_extrema_loss_weight(x, mask, dataset, args)

        optimizer.zero_grad(set_to_none=True)
        if method == "unet":
            pred = model(torch.zeros_like(residual), base_hat)
            loss = masked_mse(pred, residual, mask, loss_weight)
        elif method == "diffusion":
            assert schedule is not None
            idx = torch.randint(0, schedule.steps, (x.shape[0],), device=device)
            eps = torch.randn_like(residual)
            abar = schedule.gather(schedule.alpha_bar, idx, residual)
            noisy = abar.sqrt() * residual + (1.0 - abar).sqrt() * eps
            t = idx.float() / max(schedule.steps - 1, 1)
            pred_eps = model(noisy, base_hat, t)
            loss = masked_mse(pred_eps, eps, mask, loss_weight)
        elif method == "flow":
            noise = torch.randn_like(residual)
            t = torch.rand(x.shape[0], device=device)
            xt = (1.0 - t[:, None, None, None]) * noise + t[:, None, None, None] * residual
            target_v = residual - noise
            pred_v = model(xt, base_hat, t)
            loss = masked_mse(pred_v, target_v, mask, loss_weight)
        else:
            raise ValueError(method)

        loss.backward()
        if args.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        optimizer.step()
        total += float(loss.item()) * x.shape[0]
        n += x.shape[0]
        done_batches = bi + 1
        if logger is not None and log_interval > 0:
            if done_batches % log_interval == 0 or done_batches >= total_batches:
                mean_loss = total / max(n, 1)
                logger.info(
                    "[%s] train batch %d/%d loss=%.6f mean_loss=%.6f",
                    label,
                    done_batches,
                    total_batches,
                    float(loss.item()),
                    mean_loss,
                )
    return total / max(n, 1)


@torch.no_grad()
def sample_residual(
    method: str,
    model: Optional[ResidualUNet],
    x_hat: torch.Tensor,
    args: argparse.Namespace,
    schedule: Optional[DiffusionSchedule],
) -> torch.Tensor:
    if method == "mlic":
        return torch.zeros_like(x_hat)
    if model is None:
        raise ValueError("model is required for residual methods")
    model.eval()
    if method == "unet":
        return model(torch.zeros_like(x_hat), x_hat) * float(args.residual_scale)
    if method == "diffusion":
        assert schedule is not None
        x = torch.randn_like(x_hat)
        step_ids = torch.linspace(schedule.steps - 1, 0, args.sample_steps, device=x_hat.device).long()
        for si, step in enumerate(step_ids):
            step_i = int(step.item())
            prev_i = int(step_ids[si + 1].item()) if si + 1 < len(step_ids) else -1
            idx = torch.full((x_hat.shape[0],), step_i, device=x_hat.device, dtype=torch.long)
            t = idx.float() / max(schedule.steps - 1, 1)
            eps = model(x, x_hat, t)
            abar = schedule.gather(schedule.alpha_bar, idx, x)
            x0 = (x - (1.0 - abar).sqrt() * eps) / abar.sqrt().clamp(min=1e-8)
            if prev_i >= 0:
                prev_idx = torch.full((x_hat.shape[0],), prev_i, device=x_hat.device, dtype=torch.long)
                abar_prev = schedule.gather(schedule.alpha_bar, prev_idx, x)
                x = abar_prev.sqrt() * x0 + (1.0 - abar_prev).sqrt() * eps
            else:
                x = x0
        return x * float(args.residual_scale)
    if method == "flow":
        x = torch.randn_like(x_hat)
        n_steps = max(int(args.sample_steps), 1)
        dt = 1.0 / n_steps
        for i in range(n_steps):
            t = torch.full((x_hat.shape[0],), i / n_steps, device=x_hat.device, dtype=x_hat.dtype)
            v = model(x, x_hat, t)
            x = x + dt * v
        return x * float(args.residual_scale)
    raise ValueError(method)


def batch_f1_rows(
    gt_norm: torch.Tensor,
    pred_norm: torch.Tensor,
    valid_mask: torch.Tensor,
    dataset: NetCDFFolder2D,
    args: argparse.Namespace,
    full_metrics: bool = False,
) -> List[Dict[str, float]]:
    norm_mode = None if args.normalize_mode == "none" else args.normalize_mode
    gt_phys = denorm_bchw(gt_norm, dataset.norm_stats, norm_mode)
    pred_phys = denorm_bchw(pred_norm, dataset.norm_stats, norm_mode)
    rows: List[Dict[str, float]] = []
    for i in range(gt_phys.shape[0]):
        gt = gt_phys[i]
        pred = pred_phys[i]
        valid = valid_mask[i].to(dtype=torch.bool, device=gt.device)
        gt = torch.where(valid, gt, torch.full_like(gt, float("nan")))
        pred = torch.where(valid, pred, torch.full_like(pred, float("nan")))
        diff = torch.where(valid, pred - gt, torch.zeros_like(pred - gt))
        denom = valid.sum().clamp(min=1)
        rmse = torch.sqrt((diff * diff).sum() / denom)
        row = {
            "rmse_phys": float(rmse.item()),
        }
        robust_masks: Dict[str, Tuple[torch.Tensor, torch.Tensor]] = {}
        for label, kind in (("min", "min"), ("max", "max"), ("combined", "both")):
            gt_ext = detect_extrema(
                gt,
                kind=kind,
                grad_eps=args.grad_eps,
                smooth_window=args.smooth_window,
                min_separation=args.min_separation,
            )
            pred_ext = detect_extrema(
                pred,
                kind=kind,
                grad_eps=args.grad_eps,
                smooth_window=args.smooth_window,
                min_separation=args.min_separation,
                grad_eps_reference=gt,
            )
            f1 = f1_and_confusion_per_level(gt_ext, pred_ext, kernel_size=args.f1_kernel_size)
            row[f"f1_{label}_mean"] = float(f1["f1_mean"])
            row[f"f1_{label}_mean_weighted_h"] = float(
                f1.get("f1_mean_weighted_h", float("nan"))
            )
            row[f"gt_{label}_extrema_count"] = float(gt_ext.sum().item())
            row[f"pred_{label}_extrema_count"] = float(pred_ext.sum().item())
            robust_masks[label] = (gt_ext, pred_ext)
        if full_metrics:
            add_full_metric_fields(row, gt, pred, valid, robust_masks, args)
        rows.append(row)
    return rows


def _nanmean_values(*values: float) -> float:
    arr = np.asarray(values, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    return float(arr.mean()) if arr.size else float("nan")


def _safe_wasserstein(gt_mask: torch.Tensor, pred_mask: torch.Tensor, valid: torch.Tensor) -> float:
    try:
        out = extrema_wasserstein(gt_mask, pred_mask, valid=valid)
        return float(out.get("wasserstein_mean", float("nan")))
    except Exception:
        return float("nan")


def _add_wasserstein_fields(
    row: Dict[str, float],
    prefix: str,
    masks: Dict[str, Tuple[torch.Tensor, torch.Tensor]],
    valid: torch.Tensor,
    args: argparse.Namespace,
) -> None:
    if not bool(args.wasserstein_metrics):
        return
    values: Dict[str, float] = {}
    for label in ("min", "max", "combined"):
        gt_mask, pred_mask = masks[label]
        out_label = "both" if label == "combined" else label
        values[out_label] = _safe_wasserstein(gt_mask, pred_mask, valid)
        row[f"{prefix}_{out_label}"] = values[out_label]
    row[f"{prefix}_combined"] = _nanmean_values(values.get("min", float("nan")), values.get("max", float("nan")))


def _add_prominence_fields(
    row: Dict[str, float],
    gt: torch.Tensor,
    pred: torch.Tensor,
    valid: torch.Tensor,
    args: argparse.Namespace,
) -> Dict[str, Tuple[torch.Tensor, torch.Tensor]]:
    prom_masks: Dict[str, Tuple[torch.Tensor, torch.Tensor]] = {}
    for label, kind in (("min", "min"), ("max", "max"), ("both", "both")):
        gt_prom = detect_extrema_prominence(
            gt,
            kind=kind,
            prominence=args.extrema_prominence,
            min_distance=5,
        ) * valid
        pred_prom = detect_extrema_prominence(
            pred,
            kind=kind,
            prominence=args.extrema_prominence,
            min_distance=5,
        ) * valid
        prom = f1_and_confusion_per_level(gt_prom, pred_prom, kernel_size=args.f1_kernel_size)
        row[f"f1_prom_{label}_mean"] = float(prom["f1_mean"])
        row[f"f1_prom_{label}_mean_weighted_h"] = float(prom.get("f1_mean_weighted_h", float("nan")))
        row[f"gt_prom_{label}_extrema_count"] = float(gt_prom.sum().item())
        row[f"pred_prom_{label}_extrema_count"] = float(pred_prom.sum().item())
        prom_masks["combined" if label == "both" else label] = (gt_prom, pred_prom)
    row["f1_prom_combined_mean"] = _nanmean_values(
        row.get("f1_prom_min_mean", float("nan")),
        row.get("f1_prom_max_mean", float("nan")),
    )
    row["f1_prom_combined_mean_weighted_h"] = _nanmean_values(
        row.get("f1_prom_min_mean_weighted_h", float("nan")),
        row.get("f1_prom_max_mean_weighted_h", float("nan")),
    )
    try:
        prom_match = prominence_matching_metrics(
            gt,
            pred,
            valid_mask=valid,
            kind="both",
            prominence_fraction=(
                float(args.extrema_prominence_ratio)
                if args.extrema_prominence_ratio is not None
                else 0.1
            ),
            min_width=1,
            min_distance=5,
            match_radius=max(1, int(args.f1_kernel_size // 2)),
        )
        row["prom_weighted_precision_both"] = float(prom_match.get("weighted_precision", float("nan")))
        row["prom_weighted_recall_both"] = float(prom_match.get("weighted_recall", float("nan")))
        row["prom_weighted_f1_both"] = float(prom_match.get("weighted_f1", float("nan")))
        row["prom_loc_mae_both"] = float(prom_match.get("loc_mae", float("nan")))
        row["prom_prom_rel_mae_both"] = float(prom_match.get("prom_rel_mae", float("nan")))
        row["prom_amp_mae_both"] = float(prom_match.get("amp_mae", float("nan")))
        row["prom_count_bias_both"] = float(prom_match.get("count_bias", float("nan")))
    except Exception:
        row["prom_weighted_precision_both"] = float("nan")
        row["prom_weighted_recall_both"] = float("nan")
        row["prom_weighted_f1_both"] = float("nan")
        row["prom_loc_mae_both"] = float("nan")
        row["prom_prom_rel_mae_both"] = float("nan")
        row["prom_amp_mae_both"] = float("nan")
        row["prom_count_bias_both"] = float("nan")
    return prom_masks


def _add_filtered_fields(
    row: Dict[str, float],
    gt: torch.Tensor,
    pred: torch.Tensor,
    valid: torch.Tensor,
    robust_masks: Dict[str, Tuple[torch.Tensor, torch.Tensor]],
    args: argparse.Namespace,
) -> Dict[str, Tuple[torch.Tensor, torch.Tensor]]:
    pred_filtered = lowpass_filter_torch_along_axis(pred, axis=-2, order=2, wn=0.107)
    filtered_masks: Dict[str, Tuple[torch.Tensor, torch.Tensor]] = {}
    for label, kind in (("min", "min"), ("max", "max"), ("combined", "both")):
        gt_ext, _ = robust_masks[label]
        pred_ext = detect_extrema(
            pred_filtered,
            kind=kind,
            grad_eps=args.grad_eps,
            smooth_window=args.smooth_window,
            min_separation=args.min_separation,
            grad_eps_reference=gt,
        )
        pred_ext = pred_ext * valid
        f1 = f1_and_confusion_per_level(gt_ext, pred_ext, kernel_size=args.f1_kernel_size)
        row[f"f1_{label}_filtered_mean"] = float(f1["f1_mean"])
        row[f"f1_{label}_filtered_mean_weighted_h"] = float(f1.get("f1_mean_weighted_h", float("nan")))
        row[f"pred_{label}_filtered_extrema_count"] = float(pred_ext.sum().item())
        filtered_masks[label] = (gt_ext, pred_ext)
    return filtered_masks


def _add_dtw_fields(
    row: Dict[str, float],
    gt: torch.Tensor,
    pred: torch.Tensor,
    args: argparse.Namespace,
) -> None:
    profile_dtw_keys = (
        "dtw_mean_profile",
        "dtw_std_profile",
        "extrema_depth_err_mean",
        "extrema_amp_err_mean",
        "dtw_profiles_used",
        "dtw_profiles_total",
    )
    extrema_prefixes = (
        "extrema_dtw",
        "extrema_dtw_filtered",
        "extrema_dtw_prom_ratio",
    )
    if not bool(getattr(args, "dtw_metrics", True)):
        for key in profile_dtw_keys:
            row[key] = float("nan")
        for prefix in extrema_prefixes:
            for label in ("min", "max", "both", "combined"):
                row[f"{prefix}_{label}"] = float("nan")
        return

    if gt.ndim != 3 or pred.ndim != 3:
        return
    gt_np = gt.detach().cpu().numpy().astype(np.float64, copy=False)
    pred_np = pred.detach().cpu().numpy().astype(np.float64, copy=False)
    pred_filtered_np = lowpass_filter_torch_along_axis(
        pred,
        axis=-2,
        order=2,
        wn=0.107,
    ).detach().cpu().numpy().astype(np.float64, copy=False)

    if getattr(args, "compute_profile_dtw", False):
        try:
            dtw = compute_dtw_and_extrema_metrics(
                gt_3d=gt_np,
                rec_3d=pred_np,
                dims3=None,
                window=args.dtw_window,
                prominence=args.extrema_prominence,
            )
            row["dtw_mean_profile"] = float(dtw["dtw_mean"])
            row["dtw_std_profile"] = float(dtw["dtw_std"])
            row["extrema_depth_err_mean"] = float(dtw["extrema_depth_err_mean"])
            row["extrema_amp_err_mean"] = float(dtw["extrema_amp_err_mean"])
            row["dtw_profiles_used"] = float(dtw["num_profiles_used"])
            row["dtw_profiles_total"] = float(dtw["num_profiles_total"])
        except Exception:
            for key in profile_dtw_keys:
                row[key] = float("nan")
    else:
        for key in profile_dtw_keys:
            row[key] = float("nan")

    def _extrema_dtw(prefix: str, rec_np: np.ndarray, prominence_ratio: Optional[float] = None) -> None:
        try:
            out = compute_extrema_dtw_metrics(
                gt_3d=gt_np,
                rec_3d=rec_np,
                dims3=None,
                window=args.dtw_window,
                prominence=args.extrema_prominence,
                prominence_ratio=prominence_ratio,
            )
            row[f"{prefix}_min"] = float(out["extrema_dtw_min"])
            row[f"{prefix}_max"] = float(out["extrema_dtw_max"])
            row[f"{prefix}_both"] = float(out["extrema_dtw_both"])
            row[f"{prefix}_combined"] = float(out["extrema_dtw_combined"])
        except Exception:
            for label in ("min", "max", "both", "combined"):
                row[f"{prefix}_{label}"] = float("nan")

    _extrema_dtw("extrema_dtw", pred_np)
    _extrema_dtw("extrema_dtw_filtered", pred_filtered_np)
    if args.extrema_prominence_ratio is not None:
        _extrema_dtw("extrema_dtw_prom_ratio", pred_np, prominence_ratio=float(args.extrema_prominence_ratio))


def add_full_metric_fields(
    row: Dict[str, float],
    gt: torch.Tensor,
    pred: torch.Tensor,
    valid: torch.Tensor,
    robust_masks: Dict[str, Tuple[torch.Tensor, torch.Tensor]],
    args: argparse.Namespace,
) -> None:
    prom_masks = _add_prominence_fields(row, gt, pred, valid, args)
    filtered_masks = _add_filtered_fields(row, gt, pred, valid, robust_masks, args)
    _add_wasserstein_fields(row, "wasserstein_extrema", robust_masks, valid, args)
    _add_wasserstein_fields(row, "wasserstein_prom", prom_masks, valid, args)
    _add_wasserstein_fields(row, "wasserstein_filtered", filtered_masks, valid, args)
    _add_dtw_fields(row, gt, pred, args)


def mean_rows(rows: Iterable[Dict[str, float]]) -> Dict[str, float]:
    rows = list(rows)
    keys = sorted({k for row in rows for k in row})
    out: Dict[str, float] = {}
    for key in keys:
        vals = np.asarray([row[key] for row in rows if key in row], dtype=np.float64)
        vals = vals[np.isfinite(vals)]
        out[key] = float(vals.mean()) if vals.size else float("nan")
    return out


def add_f1_aggregate_fields(summary: Dict[str, float]) -> None:
    plain = [
        summary.get("f1_min_mean", float("nan")),
        summary.get("f1_max_mean", float("nan")),
        summary.get("f1_combined_mean", float("nan")),
    ]
    weighted = [
        summary.get("f1_min_mean_weighted_h", float("nan")),
        summary.get("f1_max_mean_weighted_h", float("nan")),
        summary.get("f1_combined_mean_weighted_h", float("nan")),
    ]
    summary["f1_min_max_combined_mean"] = float(np.nanmean(plain))
    summary["f1_min_max_combined_mean_weighted_h"] = float(np.nanmean(weighted))


def ensemble_diagnostics(
    samples: torch.Tensor,
    target: torch.Tensor,
    valid_mask: torch.Tensor,
    dataset: NetCDFFolder2D,
    args: argparse.Namespace,
) -> Dict[str, float]:
    """Basic ensemble spread and calibration diagnostics.

    samples: (K, B, C, H, W) normalized-space predictions.
    target:  (B, C, H, W) normalized-space truth.
    """
    if samples.shape[0] <= 1:
        return {
            "ensemble_size": float(samples.shape[0]),
            "ensemble_spread_norm": 0.0,
            "spread_skill_ratio_norm": 0.0,
            "calib_1sigma_coverage_norm": float("nan"),
            "calib_2sigma_coverage_norm": float("nan"),
            "ensemble_spread_phys": 0.0,
        }

    mask = valid_mask.to(dtype=torch.bool, device=samples.device)
    mean = samples.mean(dim=0)
    std = samples.std(dim=0, unbiased=False)
    diff = torch.where(mask, mean - target, torch.zeros_like(mean))
    denom = mask.sum().clamp(min=1)
    rmse = torch.sqrt((diff * diff).sum() / denom)
    spread = std[mask].mean() if int(mask.sum().item()) > 0 else torch.tensor(float("nan"), device=samples.device)
    abs_err = (mean - target).abs()
    eps = torch.finfo(samples.dtype).eps
    cov1 = ((abs_err <= std + eps) & mask).sum().float() / denom.float()
    cov2 = ((abs_err <= 2.0 * std + eps) & mask).sum().float() / denom.float()

    norm_mode = None if args.normalize_mode == "none" else args.normalize_mode
    samples_phys = denorm_bchw(
        samples.reshape(-1, *samples.shape[2:]),
        dataset.norm_stats,
        norm_mode,
    ).reshape_as(samples)
    spread_phys = samples_phys.std(dim=0, unbiased=False)
    spread_phys_mean = spread_phys[mask].mean() if int(mask.sum().item()) > 0 else torch.tensor(float("nan"), device=samples.device)
    return {
        "ensemble_size": float(samples.shape[0]),
        "ensemble_spread_norm": float(spread.item()),
        "spread_skill_ratio_norm": float((spread / rmse.clamp(min=eps)).item()),
        "calib_1sigma_coverage_norm": float(cov1.item()),
        "calib_2sigma_coverage_norm": float(cov2.item()),
        "ensemble_spread_phys": float(spread_phys_mean.item()),
    }


def ensemble_score_diagnostics(
    samples: torch.Tensor,
    target: torch.Tensor,
    valid_mask: torch.Tensor,
    dataset: NetCDFFolder2D,
    args: argparse.Namespace,
    full_metrics: bool = False,
) -> Dict[str, float]:
    """F1/RMSE spread across stochastic ensemble members.

    Each ensemble member is scored independently against the target. For each
    input sample in the batch, we compute member mean/std/min/max/range, then
    average those diagnostics across the batch.
    """
    member_batch_rows: List[List[Dict[str, float]]] = []
    for member_idx in range(samples.shape[0]):
        rows = batch_f1_rows(
            target,
            samples[member_idx],
            valid_mask,
            dataset,
            args,
            full_metrics=full_metrics,
        )
        for row in rows:
            add_f1_aggregate_fields(row)
        member_batch_rows.append(rows)

    if not member_batch_rows:
        return {}

    score_keys = [
        "rmse_phys",
        "f1_min_mean",
        "f1_max_mean",
        "f1_combined_mean",
        "f1_min_max_combined_mean",
        "f1_min_mean_weighted_h",
        "f1_max_mean_weighted_h",
        "f1_combined_mean_weighted_h",
        "f1_min_max_combined_mean_weighted_h",
    ]
    if full_metrics:
        score_keys.extend([
            "f1_prom_min_mean",
            "f1_prom_max_mean",
            "f1_prom_both_mean",
            "f1_prom_combined_mean",
            "f1_prom_min_mean_weighted_h",
            "f1_prom_max_mean_weighted_h",
            "f1_prom_both_mean_weighted_h",
            "f1_prom_combined_mean_weighted_h",
            "prom_weighted_precision_both",
            "prom_weighted_recall_both",
            "prom_weighted_f1_both",
            "prom_loc_mae_both",
            "prom_prom_rel_mae_both",
            "prom_amp_mae_both",
            "prom_count_bias_both",
            "f1_min_filtered_mean",
            "f1_max_filtered_mean",
            "f1_combined_filtered_mean",
            "f1_min_filtered_mean_weighted_h",
            "f1_max_filtered_mean_weighted_h",
            "f1_combined_filtered_mean_weighted_h",
            "dtw_mean_profile",
            "extrema_depth_err_mean",
            "extrema_amp_err_mean",
            "extrema_dtw_min",
            "extrema_dtw_max",
            "extrema_dtw_both",
            "extrema_dtw_combined",
            "extrema_dtw_prom_ratio_min",
            "extrema_dtw_prom_ratio_max",
            "extrema_dtw_prom_ratio_both",
            "extrema_dtw_prom_ratio_combined",
            "extrema_dtw_filtered_min",
            "extrema_dtw_filtered_max",
            "extrema_dtw_filtered_both",
            "extrema_dtw_filtered_combined",
            "wasserstein_extrema_min",
            "wasserstein_extrema_max",
            "wasserstein_extrema_both",
            "wasserstein_extrema_combined",
            "wasserstein_prom_min",
            "wasserstein_prom_max",
            "wasserstein_prom_both",
            "wasserstein_prom_combined",
            "wasserstein_filtered_min",
            "wasserstein_filtered_max",
            "wasserstein_filtered_both",
            "wasserstein_filtered_combined",
        ])
    batch_size = len(member_batch_rows[0])
    out_rows: List[Dict[str, float]] = []
    for batch_idx in range(batch_size):
        row: Dict[str, float] = {}
        for key in score_keys:
            values = np.asarray(
                [
                    member_rows[batch_idx].get(key, float("nan"))
                    for member_rows in member_batch_rows
                ],
                dtype=np.float64,
            )
            values = values[np.isfinite(values)]
            if not values.size:
                for stat in ("mean", "std", "min", "max", "range"):
                    row[f"{key}_member_{stat}"] = float("nan")
                continue
            row[f"{key}_member_mean"] = float(values.mean())
            row[f"{key}_member_std"] = float(values.std(ddof=0))
            row[f"{key}_member_min"] = float(values.min())
            row[f"{key}_member_max"] = float(values.max())
            row[f"{key}_member_range"] = float(values.max() - values.min())
        out_rows.append(row)
    return mean_rows(out_rows)


@torch.no_grad()
def evaluate_method(
    method: str,
    model: Optional[ResidualUNet],
    codec: nn.Module,
    loader: DataLoader,
    dataset: NetCDFFolder2D,
    device: torch.device,
    args: argparse.Namespace,
    schedule: Optional[DiffusionSchedule],
    residual_weight: float = 1.0,
    ensemble_size: int = 1,
    base_model: Optional[ResidualUNet] = None,
    base_residual_weight: float = 0.0,
    full_metrics: bool = False,
    logger: Optional[logging.Logger] = None,
    progress_label: Optional[str] = None,
) -> Dict[str, float]:
    rows: List[Dict[str, float]] = []
    residual_mses: List[float] = []
    diag_rows: List[Dict[str, float]] = []
    k_samples = max(1, int(ensemble_size))
    if method in ("mlic", "unet"):
        k_samples = 1
    total_batches = len(loader)
    if args.max_eval_batches is not None:
        total_batches = min(total_batches, int(args.max_eval_batches))
    label = progress_label or f"eval:{method}"
    log_interval = max(0, int(getattr(args, "eval_log_interval", 0) or 0))
    if logger is not None:
        logger.info(
            "[%s] starting evaluation batches=%d full_metrics=%s ensemble=%d alpha=%.6g",
            label,
            total_batches,
            bool(full_metrics),
            k_samples,
            float(residual_weight),
        )
    for bi, batch in enumerate(progress_iter(loader, total=total_batches, desc=label, args=args)):
        if args.max_eval_batches is not None and bi >= args.max_eval_batches:
            break
        item = batch_item(batch)
        x = item.input.to(device, non_blocking=True)
        mask = item.valid_mask.to(device, non_blocking=True)
        x_hat = frozen_decode(codec, x)
        base_hat = x_hat
        if base_model is not None and float(base_residual_weight) != 0.0:
            base_residual = sample_residual("unet", base_model, x_hat, args, schedule)
            base_hat = x_hat + float(base_residual_weight) * base_residual
            if args.clamp_output:
                base_hat = base_hat.clamp(0.0, 1.0)
        residuals = [
            sample_residual(method, model, base_hat, args, schedule)
            for _ in range(k_samples)
        ]
        residual_stack = torch.stack(residuals, dim=0)
        corrected_samples = base_hat.unsqueeze(0) + float(residual_weight) * residual_stack
        if args.clamp_output:
            corrected_samples = corrected_samples.clamp(0.0, 1.0)
        corrected_mean = corrected_samples.mean(dim=0)
        rows.extend(batch_f1_rows(x, corrected_mean, mask, dataset, args, full_metrics=full_metrics))
        residual_mses.append(float(masked_mse(corrected_mean, x, mask).item()))
        diag_rows.append(ensemble_diagnostics(corrected_samples, x, mask, dataset, args))
        diag_rows.append(
            ensemble_score_diagnostics(
                corrected_samples,
                x,
                mask,
                dataset,
                args,
                full_metrics=full_metrics and bool(args.full_metric_member_spread),
            )
        )
        done_batches = bi + 1
        if logger is not None and log_interval > 0:
            if done_batches % log_interval == 0 or done_batches >= total_batches:
                logger.info("[%s] evaluation batch %d/%d", label, done_batches, total_batches)
    summary = mean_rows(rows)
    diag_summary = mean_rows(diag_rows)
    summary.update(diag_summary)
    add_f1_aggregate_fields(summary)
    summary["normalized_mse"] = float(np.mean(residual_mses)) if residual_mses else float("nan")
    summary["n_samples"] = float(len(rows))
    summary["residual_weight"] = float(residual_weight)
    return summary


@torch.no_grad()
def evaluate_weight_sweep(
    method: str,
    model: Optional[ResidualUNet],
    codec: nn.Module,
    loader: DataLoader,
    dataset: NetCDFFolder2D,
    device: torch.device,
    args: argparse.Namespace,
    schedule: Optional[DiffusionSchedule],
    ensemble_size: int,
    base_model: Optional[ResidualUNet] = None,
    base_residual_weight: float = 0.0,
    full_metrics: bool = False,
    logger: Optional[logging.Logger] = None,
    progress_label: Optional[str] = None,
) -> Tuple[Dict[str, float], List[Dict[str, float]]]:
    summaries = []
    for alpha in args.residual_weights:
        if logger is not None:
            sweep_label = progress_label or f"final:{method}"
            logger.info("[%s] starting residual-weight alpha=%g", sweep_label, float(alpha))
        summary = evaluate_method(
            method,
            model,
            codec,
            loader,
            dataset,
            device,
            args,
            schedule,
            residual_weight=float(alpha),
            ensemble_size=ensemble_size,
            base_model=base_model,
            base_residual_weight=base_residual_weight,
            full_metrics=full_metrics,
            logger=logger,
            progress_label=(f"{progress_label} alpha={float(alpha):g}" if progress_label else None),
        )
        summaries.append(summary)
        if logger is not None:
            logger.info(
                "[%s] alpha=%g %s=%.6f",
                progress_label or f"final:{method}",
                float(alpha),
                args.selection_metric,
                summary.get(args.selection_metric, float("nan")),
            )
    best = max(summaries, key=lambda row: row.get(args.selection_metric, float("-inf")))
    return best, summaries


def save_checkpoint(path: Path, method: str, model: ResidualUNet, args: argparse.Namespace, epoch: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "method": method,
        "epoch": epoch,
        "state_dict": model.state_dict(),
        "channels": None,
        "args": vars(args),
    }, path)


def load_residual_checkpoint(path: Path, model: ResidualUNet, device: torch.device) -> bool:
    if not path.exists():
        return False
    ckpt = torch.load(path, map_location=device)
    model.load_state_dict(ckpt["state_dict"], strict=True)
    return True


def build_residual_model(method: str, channels: int, args: argparse.Namespace, device: torch.device) -> ResidualUNet:
    time_dim = args.time_emb_dim if method in ("diffusion", "flow") else 0
    return ResidualUNet(
        channels=channels,
        cond_channels=channels,
        base_channels=args.base_channels,
        levels=args.levels,
        time_emb_dim=time_dim,
    ).to(device)


def write_summary_csv(path: Path, summaries: Dict[str, Dict[str, float]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    keys = ["method"]
    for row in summaries.values():
        for key in row:
            if key not in keys:
                keys.append(key)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        for method, row in summaries.items():
            out = {"method": method}
            out.update(row)
            writer.writerow(out)


def add_delta_fields(summary: Dict[str, float], baseline: Dict[str, float]) -> None:
    for label in ("min", "max", "combined"):
        key = f"f1_{label}_mean"
        wkey = f"f1_{label}_mean_weighted_h"
        summary[f"delta_{key}_vs_mlic"] = summary.get(key, float("nan")) - baseline.get(key, float("nan"))
        summary[f"delta_{wkey}_vs_mlic"] = summary.get(wkey, float("nan")) - baseline.get(wkey, float("nan"))
    for key in ("f1_min_max_combined_mean", "f1_min_max_combined_mean_weighted_h"):
        summary[f"delta_{key}_vs_mlic"] = summary.get(key, float("nan")) - baseline.get(key, float("nan"))


def _meta_scalar(value: Any) -> Any:
    if isinstance(value, (list, tuple)) and len(value) == 1:
        return _meta_scalar(value[0])
    if torch.is_tensor(value):
        if value.numel() == 1:
            return value.item()
        return value.detach().cpu().numpy()
    if isinstance(value, np.ndarray) and value.shape == ():
        return value.item()
    return value


def _meta_to_dims_coords(meta: Dict[str, Any], chw_shape: Tuple[int, int, int]) -> Tuple[Tuple[str, ...], Dict[str, np.ndarray]]:
    dims_raw = _meta_scalar(meta.get("dims"))
    coords_raw = _meta_scalar(meta.get("coords", {}))
    if dims_raw is None:
        dims = ("channel", "y", "x")
    else:
        dims = tuple(str(d) for d in dims_raw)
        if len(dims) == 2 and chw_shape[0] == 1:
            dims = ("channel",) + dims
        elif len(dims) != 3:
            dims = ("channel", "y", "x")

    coords: Dict[str, np.ndarray] = {}
    if isinstance(coords_raw, dict):
        for name in dims:
            if name in coords_raw:
                coords[name] = np.asarray(_meta_scalar(coords_raw[name]))
    for axis, name in enumerate(dims):
        if name not in coords:
            coords[name] = np.arange(chw_shape[axis], dtype=np.int32)
    return dims, coords


def _phys_numpy(tensor_bchw: torch.Tensor, dataset: NetCDFFolder2D, args: argparse.Namespace) -> np.ndarray:
    norm_mode = None if args.normalize_mode == "none" else args.normalize_mode
    phys = denorm_bchw(tensor_bchw.detach().cpu(), dataset.norm_stats, norm_mode)
    return phys.squeeze(0).numpy().astype(np.float32, copy=False)


def write_inference_metrics_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    keys: List[str] = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


@torch.no_grad()
def export_inference_nc(
    models: Dict[str, Optional[ResidualUNet]],
    residual_weights: Dict[str, float],
    codec: nn.Module,
    dataset: NetCDFFolder2D,
    device: torch.device,
    args: argparse.Namespace,
    schedule: Optional[DiffusionSchedule],
    outdir: Path,
    logger: logging.Logger,
) -> List[Dict[str, Any]]:
    export_root = outdir / f"inference_{args.infer_split or args.eval_split}"
    nc_dir = export_root / "nc"
    nc_dir.mkdir(parents=True, exist_ok=True)
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0, collate_fn=lambda batch: batch[0])

    rows: List[Dict[str, Any]] = []
    max_samples = max(0, int(args.max_infer_samples))
    selected_indices: Optional[set[int]] = None
    if args.random_infer_samples and max_samples > 0:
        n_available = len(dataset)
        n_select = min(max_samples, n_available)
        rng = np.random.default_rng(int(args.seed))
        selected_indices = set(int(i) for i in rng.choice(n_available, size=n_select, replace=False))
        logger.info("[export] selected random sample indices: %s", ",".join(str(i) for i in sorted(selected_indices)))
    logger.info(
        "[export] scanning split=%s total=%d max_exports=%d methods=%s",
        args.infer_split or args.eval_split,
        len(dataset),
        max_samples,
        ",".join(["mlic"] + [m for m in models if m != "mlic"]),
    )
    n_exported = 0
    export_log_interval = max(1, int(getattr(args, "export_log_interval", 1) or 1))
    for sample_idx, raw in enumerate(progress_iter(loader, total=len(dataset), desc="export", args=args)):
        if selected_indices is not None and sample_idx not in selected_indices:
            continue
        if n_exported >= max_samples:
            break
        item, meta = raw if isinstance(raw, (tuple, list)) and len(raw) == 2 else (raw, {})
        x = item.input.unsqueeze(0).to(device, non_blocking=True)
        mask = item.valid_mask.unsqueeze(0).to(device, non_blocking=True)
        x_hat = frozen_decode(codec, x)

        dims, coords = _meta_to_dims_coords(meta, tuple(x.shape[1:]))
        source_path = str(_meta_scalar(meta.get("path", "")))
        source_stem = Path(source_path).stem if source_path else f"sample_{sample_idx:05d}"
        group_idx = _meta_scalar(meta.get("slice_group_index", None))
        group_suffix = f"_g{int(group_idx):02d}" if group_idx is not None else ""

        data_vars: Dict[str, Tuple[Tuple[str, ...], np.ndarray]] = {
            f"{args.var_name}_gt": (dims, _phys_numpy(x, dataset, args)),
            f"{args.var_name}_mlic": (dims, _phys_numpy(x_hat, dataset, args)),
        }

        method_order = ["mlic"] + [m for m in models if m != "mlic"]
        for method in method_order:
            model = models.get(method)
            alpha = float(residual_weights.get(method, 0.0))
            k_samples = max(1, int(args.ensemble_size))
            if method in ("mlic", "unet"):
                k_samples = 1
            cond_hat = x_hat
            sample_method = method
            if method.startswith("unet_then_"):
                base_unet = models.get("unet")
                base_alpha = float(residual_weights.get("unet", 0.0))
                if base_unet is None:
                    logger.warning("[%s] export skipped because UNet base is unavailable", method)
                    continue
                base_residual = sample_residual("unet", base_unet, x_hat, args, schedule)
                cond_hat = x_hat + base_alpha * base_residual
                if args.clamp_output:
                    cond_hat = cond_hat.clamp(0.0, 1.0)
                sample_method = method.removeprefix("unet_then_")
            residuals = [
                sample_residual(sample_method, model, cond_hat, args, schedule)
                for _ in range(k_samples)
            ]
            residual_stack = torch.stack(residuals, dim=0)
            corrected_samples = cond_hat.unsqueeze(0) + alpha * residual_stack
            if args.clamp_output:
                corrected_samples = corrected_samples.clamp(0.0, 1.0)
            corrected_mean = corrected_samples.mean(dim=0)

            metric_rows = batch_f1_rows(
                x,
                corrected_mean,
                mask,
                dataset,
                args,
                full_metrics=bool(args.full_metrics),
            )
            metric_row: Dict[str, Any] = dict(metric_rows[0])
            metric_row.update(ensemble_diagnostics(corrected_samples, x, mask, dataset, args))
            metric_row.update(
                ensemble_score_diagnostics(
                    corrected_samples,
                    x,
                    mask,
                    dataset,
                    args,
                    full_metrics=bool(args.full_metrics) and bool(args.full_metric_member_spread),
                )
            )
            add_f1_aggregate_fields(metric_row)
            metric_row.update({
                "sample_idx": sample_idx,
                "method": method,
                "source_path": source_path,
                "nc_path": str(nc_dir / f"{sample_idx:05d}_{source_stem}{group_suffix}.nc"),
                "residual_weight": alpha,
                "normalized_mse": float(masked_mse(corrected_mean, x, mask).item()),
            })
            rows.append(metric_row)

            if method == "mlic":
                continue
            corrected_phys = _phys_numpy(corrected_mean, dataset, args)
            mlic_phys = _phys_numpy(x_hat, dataset, args)
            data_vars[f"{args.var_name}_{method}_mean"] = (dims, corrected_phys)
            data_vars[f"{args.var_name}_{method}_residual_mean"] = (dims, corrected_phys - mlic_phys)
            if k_samples > 1:
                sample_phys = denorm_bchw(
                    corrected_samples.reshape(-1, *corrected_samples.shape[2:]).detach().cpu(),
                    dataset.norm_stats,
                    None if args.normalize_mode == "none" else args.normalize_mode,
                ).reshape_as(corrected_samples.detach().cpu())
                spread = sample_phys.std(dim=0, unbiased=False).squeeze(0).numpy().astype(np.float32, copy=False)
                data_vars[f"{args.var_name}_{method}_spread"] = (dims, spread)
                if args.save_ensemble_members:
                    member_dims = ("member",) + dims
                    data_vars[f"{args.var_name}_{method}_members"] = (
                        member_dims,
                        sample_phys[:, 0].numpy().astype(np.float32, copy=False),
                    )

        ds = xr.Dataset(data_vars=data_vars, coords=coords)
        ds.attrs.update({
            "source_path": source_path,
            "codec_checkpoint": str(args.codec_checkpoint),
            "experiment": str(args.experiment),
            "split": str(args.infer_split or args.eval_split),
            "residual_scale": float(args.residual_scale),
            "methods": ",".join(method_order),
        })
        nc_path = nc_dir / f"{sample_idx:05d}_{source_stem}{group_suffix}.nc"
        ds.to_netcdf(nc_path)
        n_exported += 1
        if n_exported % export_log_interval == 0 or n_exported >= max_samples:
            logger.info("[export] saved %d/%d sample_idx=%d path=%s", n_exported, max_samples, sample_idx, nc_path)

    metrics_path = export_root / f"residual_inference_metrics_{args.infer_split or args.eval_split}.csv"
    write_inference_metrics_csv(metrics_path, rows)
    logger.info("Wrote %d NetCDF inference files under %s", n_exported, nc_dir)
    logger.info("Wrote inference metrics %s", metrics_path)
    return rows


def main() -> None:
    args = normalize_args(parse_args())
    set_seed(args.seed)
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)
    device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    torch.backends.cudnn.benchmark = device.type == "cuda"

    outdir = Path("experiments") / args.experiment
    ckpt_dir = outdir / "residual_checkpoints"
    outdir.mkdir(parents=True, exist_ok=True)
    log_path = Path(args.log_file) if args.log_file else outdir / "train_residual_f1.log"
    logger = setup_run_logger(log_path)
    logger.info("Residual F1 experiment")
    logger.info("log_file=%s", log_path)
    logger.info("dataset=%s", args.dataset)
    logger.info("codec_checkpoint=%s", args.codec_checkpoint)
    logger.info("methods=%s selection_metric=%s eval_split=%s", ",".join(args.methods), args.selection_metric, args.eval_split)
    logger.info("residual_weights=%s ensemble_size=%d epoch_ensemble_size=%d", args.residual_weights, args.ensemble_size, args.epoch_ensemble_size)
    logger.info("device=%s", device)
    logger.info("progress tqdm=%s scale_log_interval=%d train_log_interval=%d eval_log_interval=%d",
                bool(args.tqdm), int(args.scale_log_interval), int(args.train_log_interval), int(args.eval_log_interval))

    logger.info("[setup] loading train dataset")
    train_dataset = make_dataset(args, "train", return_meta=False)
    logger.info("[setup] train samples=%d", len(train_dataset))
    logger.info("[setup] loading %s dataset", args.eval_split)
    eval_dataset = make_dataset(args, args.eval_split, return_meta=False)
    logger.info("[setup] %s samples=%d", args.eval_split, len(eval_dataset))
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )
    eval_loader = DataLoader(
        eval_dataset,
        batch_size=args.test_batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )

    sample = train_dataset[0]
    channels = int(sample.input.shape[0])
    logger.info("[setup] sample channels=%d shape=%s", channels, tuple(sample.input.shape))
    logger.info("[setup] loading frozen codec checkpoint")
    codec = load_codec(args.codec_checkpoint, device)
    logger.info("[setup] loaded frozen codec checkpoint")
    if str(args.residual_scale).lower() == "auto":
        args.residual_scale = estimate_residual_scale(
            codec,
            train_loader,
            device,
            args.residual_scale_batches,
            args,
            logger,
        )
        logger.info("estimated_residual_scale=%.8g", args.residual_scale)
    else:
        args.residual_scale = float(args.residual_scale)
        logger.info("residual_scale=%.8g", args.residual_scale)
    logger.info("[setup] building diffusion schedule steps=%d sample_steps=%d", args.diffusion_steps, args.sample_steps)
    schedule = DiffusionSchedule(args.diffusion_steps, device)

    summaries: Dict[str, Dict[str, float]] = {}
    weight_summaries: Dict[str, Dict[str, float]] = {}
    loaded_models: Dict[str, Optional[ResidualUNet]] = {"mlic": None}
    best_residual_weights: Dict[str, float] = {"mlic": 0.0}
    baseline = evaluate_method(
        "mlic",
        None,
        codec,
        eval_loader,
        eval_dataset,
        device,
        args,
        schedule,
        residual_weight=0.0,
        ensemble_size=1,
        full_metrics=bool(args.full_metrics),
        logger=logger,
        progress_label="eval:mlic baseline",
    )
    add_delta_fields(baseline, baseline)
    summaries["mlic"] = baseline
    weight_summaries["mlic_alpha_0"] = dict(baseline)
    logger.info(
        "[eval] mlic "
        f"min={baseline['f1_min_mean']:.4f} "
        f"max={baseline['f1_max_mean']:.4f} "
        f"combined={baseline['f1_combined_mean']:.4f} "
        f"avg3={baseline['f1_min_max_combined_mean']:.4f}"
    )

    for method in args.methods:
        model = build_residual_model(method, channels, args, device)
        ckpt_path = ckpt_dir / f"{method}.pt"
        if (not args.force_retrain) and load_residual_checkpoint(ckpt_path, model, device):
            logger.info("[%s] loaded %s", method, ckpt_path)
        elif args.inference_only:
            logger.warning("[%s] missing checkpoint %s; skipping in inference-only mode", method, ckpt_path)
            continue
        else:
            optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
            best_metric = -1.0
            for epoch in range(1, args.epochs + 1):
                loss = train_epoch(
                    method,
                    model,
                    codec,
                    train_loader,
                    train_dataset,
                    optimizer,
                    device,
                    args,
                    schedule,
                    logger=logger,
                    progress_label=f"train:{method} epoch {epoch:03d}/{args.epochs:03d}",
                )
                summary = evaluate_method(
                    method,
                    model,
                    codec,
                    eval_loader,
                    eval_dataset,
                    device,
                    args,
                    schedule,
                    residual_weight=1.0,
                    ensemble_size=args.epoch_ensemble_size,
                    logger=logger,
                    progress_label=f"eval:{method} epoch {epoch:03d}/{args.epochs:03d}",
                )
                metric_value = summary[args.selection_metric]
                logger.info(
                    f"[{method}] epoch {epoch:03d}/{args.epochs:03d} "
                    f"loss={loss:.6f} "
                    f"min={summary['f1_min_mean']:.4f} "
                    f"max={summary['f1_max_mean']:.4f} "
                    f"combined={summary['f1_combined_mean']:.4f} "
                    f"{args.selection_metric}={metric_value:.4f} "
                    f"delta={metric_value - baseline[args.selection_metric]:+.4f}"
                )
                if metric_value > best_metric:
                    best_metric = metric_value
                    save_checkpoint(ckpt_path, method, model, args, epoch)
            if not load_residual_checkpoint(ckpt_path, model, device):
                logger.warning("[%s] no checkpoint was saved; skipping final evaluation", method)
                continue

        best_summary, alpha_summaries = evaluate_weight_sweep(
            method,
            model,
            codec,
            eval_loader,
            eval_dataset,
            device,
            args,
            schedule,
            ensemble_size=args.ensemble_size,
            full_metrics=bool(args.full_metrics),
            logger=logger,
            progress_label=f"final:{method}",
        )
        for alpha_summary in alpha_summaries:
            add_delta_fields(alpha_summary, baseline)
            alpha = alpha_summary["residual_weight"]
            weight_summaries[f"{method}_alpha_{alpha:g}"] = alpha_summary

        summary = dict(best_summary)
        add_delta_fields(summary, baseline)
        summary["best_residual_weight"] = float(summary["residual_weight"])
        summaries[method] = summary
        loaded_models[method] = model
        best_residual_weights[method] = float(summary["best_residual_weight"])
        logger.info(
            f"[final] {method} "
            f"best_alpha={summary['best_residual_weight']:.3g} "
            f"min={summary['f1_min_mean']:.4f} "
            f"max={summary['f1_max_mean']:.4f} "
            f"combined={summary['f1_combined_mean']:.4f} "
            f"avg3={summary['f1_min_max_combined_mean']:.4f} "
            f"spread={summary.get('ensemble_spread_phys', float('nan')):.4f} "
            f"f1_std={summary.get('f1_min_max_combined_mean_member_std', float('nan')):.4f} "
            f"delta_avg3={summary['delta_f1_min_max_combined_mean_vs_mlic']:+.4f}"
        )

    if args.cascade_after_unet:
        unet_model = loaded_models.get("unet")
        unet_alpha = float(best_residual_weights.get("unet", 0.0))
        if unet_model is None or unet_alpha == 0.0:
            logger.warning(
                "[cascade] requested but no selected UNet residual is available; "
                "include --methods unet or run after training/loading UNet."
            )
        else:
            logger.info(
                "[cascade] training second-stage residuals after UNet correction "
                "with unet_alpha=%.6g",
                unet_alpha,
            )
            for method in args.cascade_methods:
                cascade_name = f"unet_then_{method}"
                model = build_residual_model(method, channels, args, device)
                ckpt_path = ckpt_dir / f"{cascade_name}.pt"
                if (not args.force_retrain) and load_residual_checkpoint(ckpt_path, model, device):
                    logger.info("[%s] loaded %s", cascade_name, ckpt_path)
                elif args.inference_only:
                    logger.warning(
                        "[%s] missing checkpoint %s; skipping in inference-only mode",
                        cascade_name,
                        ckpt_path,
                    )
                    continue
                else:
                    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
                    best_metric = -1.0
                    for epoch in range(1, args.epochs + 1):
                        loss = train_epoch(
                            method,
                            model,
                            codec,
                            train_loader,
                            train_dataset,
                            optimizer,
                            device,
                            args,
                            schedule,
                            base_model=unet_model,
                            base_residual_weight=unet_alpha,
                            logger=logger,
                            progress_label=f"train:{cascade_name} epoch {epoch:03d}/{args.epochs:03d}",
                        )
                        summary = evaluate_method(
                            method,
                            model,
                            codec,
                            eval_loader,
                            eval_dataset,
                            device,
                            args,
                            schedule,
                            residual_weight=1.0,
                            ensemble_size=args.epoch_ensemble_size,
                            base_model=unet_model,
                            base_residual_weight=unet_alpha,
                            logger=logger,
                            progress_label=f"eval:{cascade_name} epoch {epoch:03d}/{args.epochs:03d}",
                        )
                        metric_value = summary[args.selection_metric]
                        logger.info(
                            f"[{cascade_name}] epoch {epoch:03d}/{args.epochs:03d} "
                            f"loss={loss:.6f} "
                            f"min={summary['f1_min_mean']:.4f} "
                            f"max={summary['f1_max_mean']:.4f} "
                            f"combined={summary['f1_combined_mean']:.4f} "
                            f"{args.selection_metric}={metric_value:.4f} "
                            f"delta={metric_value - baseline[args.selection_metric]:+.4f}"
                        )
                        if metric_value > best_metric:
                            best_metric = metric_value
                            save_checkpoint(ckpt_path, method, model, args, epoch)
                    if not load_residual_checkpoint(ckpt_path, model, device):
                        logger.warning("[%s] no checkpoint was saved; skipping final evaluation", cascade_name)
                        continue

                best_summary, alpha_summaries = evaluate_weight_sweep(
                    method,
                    model,
                    codec,
                    eval_loader,
                    eval_dataset,
                    device,
                    args,
                    schedule,
                    ensemble_size=args.ensemble_size,
                    base_model=unet_model,
                    base_residual_weight=unet_alpha,
                    full_metrics=bool(args.full_metrics),
                    logger=logger,
                    progress_label=f"final:{cascade_name}",
                )
                for alpha_summary in alpha_summaries:
                    add_delta_fields(alpha_summary, baseline)
                    alpha_summary["base_method"] = "unet"
                    alpha_summary["base_residual_weight"] = unet_alpha
                    alpha = alpha_summary["residual_weight"]
                    weight_summaries[f"{cascade_name}_alpha_{alpha:g}"] = alpha_summary

                summary = dict(best_summary)
                add_delta_fields(summary, baseline)
                summary["base_method"] = "unet"
                summary["base_residual_weight"] = unet_alpha
                summary["best_residual_weight"] = float(summary["residual_weight"])
                summaries[cascade_name] = summary
                loaded_models[cascade_name] = model
                best_residual_weights[cascade_name] = float(summary["best_residual_weight"])
                logger.info(
                    f"[final] {cascade_name} "
                    f"base_alpha={unet_alpha:.3g} "
                    f"best_alpha={summary['best_residual_weight']:.3g} "
                    f"min={summary['f1_min_mean']:.4f} "
                    f"max={summary['f1_max_mean']:.4f} "
                    f"combined={summary['f1_combined_mean']:.4f} "
                    f"avg3={summary['f1_min_max_combined_mean']:.4f} "
                    f"spread={summary.get('ensemble_spread_phys', float('nan')):.4f} "
                    f"f1_std={summary.get('f1_min_max_combined_mean_member_std', float('nan')):.4f} "
                    f"delta_avg3={summary['delta_f1_min_max_combined_mean_vs_mlic']:+.4f}"
                )

    csv_path = outdir / f"residual_f1_compare_{args.eval_split}.csv"
    write_summary_csv(csv_path, summaries)
    weights_csv_path = outdir / f"residual_f1_weight_sweep_{args.eval_split}.csv"
    write_summary_csv(weights_csv_path, weight_summaries)
    logger.info("Wrote %s", csv_path)
    logger.info("Wrote %s", weights_csv_path)

    if args.export_nc:
        infer_split = args.infer_split or args.eval_split
        export_dataset = make_dataset(args, infer_split, return_meta=True)
        export_inference_nc(
            loaded_models,
            best_residual_weights,
            codec,
            export_dataset,
            device,
            args,
            schedule,
            outdir,
            logger,
        )
    logger.info("Log saved to %s", log_path)


if __name__ == "__main__":
    main()