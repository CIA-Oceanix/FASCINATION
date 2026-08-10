from __future__ import annotations

import os 
import sys


running_path = "/Odyssey/private/o23gauvr/code/"
os.chdir(running_path)
sys.path.insert(0,running_path)

import argparse
import csv
from datetime import datetime
import json
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter

from FASCINATION.src.bootstrap_codec_inference import load_truth_and_ae_unorm_cpu

from FASCINATION.src.residual import compute_residual_target, scale_residual_target


@dataclass
class MetricRule:
    mode: str  # min or max


class ResidualArrayDataset(Dataset):
    def __init__(self, residual_scaled: np.ndarray, cond_scaled: np.ndarray):
        if residual_scaled.shape != cond_scaled.shape:
            raise ValueError(
                f"residual and condition shapes differ: {residual_scaled.shape} vs {cond_scaled.shape}"
            )
        if residual_scaled.ndim != 4:
            raise ValueError(f"Expected arrays with shape (N, C, H, W), got {residual_scaled.shape}")
        self.residual = residual_scaled.astype(np.float32, copy=False)
        self.cond = cond_scaled.astype(np.float32, copy=False)

    def __len__(self) -> int:
        return self.residual.shape[0]

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return torch.from_numpy(self.residual[idx]), torch.from_numpy(self.cond[idx])


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


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
    def __init__(self, in_ch: int, out_ch: int, time_dim: int):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1)
        self.norm1 = nn.GroupNorm(min(8, out_ch), out_ch)
        self.norm2 = nn.GroupNorm(min(8, out_ch), out_ch)
        self.skip = nn.Identity() if in_ch == out_ch else nn.Conv2d(in_ch, out_ch, kernel_size=1)
        self.time = nn.Linear(time_dim, out_ch)

    def forward(self, x: torch.Tensor, temb: torch.Tensor) -> torch.Tensor:
        h = F.silu(self.norm1(self.conv1(x)))
        h = h + self.time(temb)[:, :, None, None]
        h = self.norm2(self.conv2(h))
        return F.silu(h + self.skip(x))


class FlowResidualNet(nn.Module):
    def __init__(
        self,
        channels: int,
        cond_channels: int,
        base_channels: int,
        levels: int,
        time_emb_dim: int,
    ):
        super().__init__()
        self.time_emb_dim = int(time_emb_dim)
        self.time_mlp = nn.Sequential(
            nn.Linear(self.time_emb_dim, self.time_emb_dim),
            nn.SiLU(),
            nn.Linear(self.time_emb_dim, self.time_emb_dim),
        )

        in_ch = channels + cond_channels
        self.downs = nn.ModuleList()
        self.pools = nn.ModuleList()

        ch = in_ch
        skip_channels = []
        for level in range(levels):
            out_ch = base_channels * (2 ** level)
            self.downs.append(ResBlock(ch, out_ch, self.time_emb_dim))
            self.pools.append(nn.AvgPool2d(2))
            skip_channels.append(out_ch)
            ch = out_ch

        self.mid = ResBlock(ch, ch * 2, self.time_emb_dim)
        ch = ch * 2

        self.ups = nn.ModuleList()
        for skip_ch in reversed(skip_channels):
            self.ups.append(
                nn.ModuleDict(
                    {
                        "up": nn.ConvTranspose2d(ch, skip_ch, kernel_size=2, stride=2),
                        "block": ResBlock(skip_ch * 2, skip_ch, self.time_emb_dim),
                    }
                )
            )
            ch = skip_ch

        self.out = nn.Conv2d(ch, channels, kernel_size=1)

    def forward(self, x: torch.Tensor, cond: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
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


def flow_matching_loss(
    model: FlowResidualNet,
    residual_target: torch.Tensor,
    cond: torch.Tensor,
) -> torch.Tensor:
    noise = torch.randn_like(residual_target)
    t = torch.rand(residual_target.shape[0], device=residual_target.device)
    xt = (1.0 - t[:, None, None, None]) * noise + t[:, None, None, None] * residual_target
    target_v = residual_target - noise
    pred_v = model(xt, cond, t)
    return F.mse_loss(pred_v, target_v)


@torch.no_grad()
def sample_flow_residual(
    model: FlowResidualNet,
    cond: torch.Tensor,
    sample_steps: int,
) -> torch.Tensor:
    sample_steps = max(1, int(sample_steps))
    x = torch.randn_like(cond)
    dt = 1.0 / float(sample_steps)
    for step in range(sample_steps):
        t = torch.full((cond.shape[0],), float(step) / float(sample_steps), device=cond.device, dtype=cond.dtype)
        v = model(x, cond, t)
        x = x + dt * v
    return x


@torch.no_grad()
def compute_corrcoef(pred: torch.Tensor, target: torch.Tensor) -> float:
    p = pred.reshape(pred.shape[0], -1)
    t = target.reshape(target.shape[0], -1)
    p = p - p.mean(dim=1, keepdim=True)
    t = t - t.mean(dim=1, keepdim=True)
    denom = torch.sqrt((p * p).sum(dim=1) * (t * t).sum(dim=1)).clamp(min=1e-8)
    corr = ((p * t).sum(dim=1) / denom).mean()
    return float(corr.item())


def cosine_with_warmup_lambda(current_step: int, warmup_steps: int, total_steps: int, min_lr_ratio: float) -> float:
    if current_step < warmup_steps:
        return float(current_step + 1) / float(max(1, warmup_steps))
    progress = (current_step - warmup_steps) / float(max(1, total_steps - warmup_steps))
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return min_lr_ratio + (1.0 - min_lr_ratio) * cosine


def make_scheduler(optimizer: torch.optim.Optimizer, epochs: int, warmup_epochs: int, min_lr_ratio: float) -> LambdaLR:
    return LambdaLR(
        optimizer,
        lr_lambda=lambda epoch: cosine_with_warmup_lambda(
            current_step=epoch,
            warmup_steps=warmup_epochs,
            total_steps=epochs,
            min_lr_ratio=min_lr_ratio,
        ),
    )


def get_residual(
    truth_npy: np.ndarray ,
    ae_npy: np.ndarray,
    scale_mode: str = "default",         # replace with your actual default
    scale_clamp_quantile: float = 0.99,  # replace with your actual default
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:


    if truth_npy is None or ae_npy is None:
        raise ValueError("Provide both truth_npy and ae_npy")

    residual = compute_residual_target(truth_npy, ae_npy)
    residual_scaled, depth_scale, _ = scale_residual_target(
        residual,
        scale_mode=scale_mode,
        clamp_quantile=scale_clamp_quantile,
    )



    if residual_scaled.shape != truth_npy.shape:
        raise ValueError(f"residual_scaled and truth shapes differ: {residual_scaled.shape} vs {truth_npy.shape}")
    if depth_scale.shape[0] != residual_scaled.shape[1]:
        raise ValueError(
            f"depth_scale length mismatch: {depth_scale.shape[0]} vs channels {residual_scaled.shape[1]}"
        )
    return residual.astype(np.float32), residual_scaled.astype(np.float32), depth_scale.astype(np.float32)


def normalize_condition_mean_std_per_depth(
    cond: np.ndarray,
    train_idx: np.ndarray,
    eps: float = 1e-6,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    mean_c = cond[train_idx].mean(axis=(0, 2, 3), keepdims=True).astype(np.float32)
    std_c = cond[train_idx].std(axis=(0, 2, 3), ddof=0, keepdims=True).astype(np.float32)
    std_c = np.maximum(std_c, np.float32(eps))
    cond_norm = (cond - mean_c) / std_c
    return cond_norm.astype(np.float32), mean_c.squeeze(), std_c.squeeze()


def normalize_condition_mean_std_global(
    cond: np.ndarray,
    train_idx: np.ndarray,
    eps: float = 1e-6,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    mean_c = cond[train_idx].mean().astype(np.float32)
    std_c = cond[train_idx].std(ddof=0).astype(np.float32)
    std_c = np.maximum(std_c, np.float32(eps))
    cond_norm = (cond - mean_c) / std_c
    return cond_norm.astype(np.float32), mean_c, std_c

def normalize_condition_min_max_global(
    cond: np.ndarray,
    train_idx: np.ndarray,
    eps: float = 1e-6,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    min_c = cond[train_idx].min().astype(np.float32)
    max_c = cond[train_idx].max().astype(np.float32)
    range_c = np.maximum(max_c - min_c, np.float32(eps))
    cond_norm = (cond - min_c) / range_c
    return cond_norm.astype(np.float32), min_c, max_c


def normalize_condition_min_max_per_depth(
    cond: np.ndarray,
    train_idx: np.ndarray,
    eps: float = 1e-6,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    min_c = cond[train_idx].min(axis=(0, 2, 3), keepdims=True).astype(np.float32)
    max_c = cond[train_idx].max(axis=(0, 2, 3), keepdims=True).astype(np.float32)
    range_c = np.maximum(max_c - min_c, np.float32(eps))
    cond_norm = (cond - min_c) / range_c
    return cond_norm.astype(np.float32), min_c.squeeze(), max_c.squeeze()


def split_indices(n_samples: int, val_fraction: float, seed: int, shuffle: bool) -> Tuple[np.ndarray, np.ndarray]:
    if not (0.0 < val_fraction < 1.0):
        raise ValueError(f"val_fraction must be in (0,1), got {val_fraction}")
    idx = np.arange(n_samples)
    if shuffle:
        rng = np.random.default_rng(seed)
        rng.shuffle(idx)
    n_val = max(1, int(round(n_samples * val_fraction)))
    val_idx = np.sort(idx[-n_val:])
    train_idx = np.sort(idx[:-n_val])
    if train_idx.size == 0:
        raise ValueError("Train split is empty; reduce val_fraction")
    return train_idx, val_idx


def checkpoint_better(candidate: float, best: Optional[float], mode: str) -> bool:
    if best is None:
        return True
    if mode == "min":
        return candidate < best
    if mode == "max":
        return candidate > best
    raise ValueError(f"Unknown mode {mode}")


def save_checkpoint(
    path: Path,
    model: FlowResidualNet,
    optimizer: torch.optim.Optimizer,
    scheduler: LambdaLR,
    epoch: int,
    metrics: Dict[str, float],
    args: dict,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch": int(epoch),
            "state_dict": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "metrics": metrics,
            "args": args,
        },
        path,
    )


def train_one_epoch(
    model: FlowResidualNet,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    grad_clip: float,
) -> float:
    model.train()
    total_loss = 0.0
    total_count = 0

    for residual_scaled, cond_scaled in loader:
        residual_scaled = residual_scaled.to(device, non_blocking=True)
        cond_scaled = cond_scaled.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        loss = flow_matching_loss(model, residual_scaled, cond_scaled)
        loss.backward()
        if grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()

        batch_size = residual_scaled.shape[0]
        total_loss += float(loss.item()) * batch_size
        total_count += batch_size

    return total_loss / max(1, total_count)


@torch.no_grad()
def evaluate(
    model: FlowResidualNet,
    loader: DataLoader,
    device: torch.device,
    sample_steps: int,
    depth_scale: torch.Tensor,
) -> Dict[str, float]:
    model.eval()
    total_flow_loss = 0.0
    total_rmse_scaled = 0.0
    total_rmse_phys = 0.0
    total_corr = 0.0
    total_count = 0

    for residual_scaled, cond_scaled in loader:
        residual_scaled = residual_scaled.to(device, non_blocking=True)
        cond_scaled = cond_scaled.to(device, non_blocking=True)

        flow_loss = flow_matching_loss(model, residual_scaled, cond_scaled)
        pred_scaled = sample_flow_residual(model, cond_scaled, sample_steps=sample_steps)

        diff_scaled = pred_scaled - residual_scaled
        mse_scaled = (diff_scaled * diff_scaled).mean()
        rmse_scaled = torch.sqrt(mse_scaled)

        depth_scale_view = depth_scale[None, :, None, None].to(device=device, dtype=pred_scaled.dtype)
        pred_phys = pred_scaled * depth_scale_view
        target_phys = residual_scaled * depth_scale_view
        diff_phys = pred_phys - target_phys
        mse_phys = (diff_phys * diff_phys).mean()
        rmse_phys = torch.sqrt(mse_phys)
        corr = compute_corrcoef(pred_scaled, residual_scaled)

        batch_size = residual_scaled.shape[0]
        total_flow_loss += float(flow_loss.item()) * batch_size
        total_rmse_scaled += float(rmse_scaled.item()) * batch_size
        total_rmse_phys += float(rmse_phys.item()) * batch_size
        total_corr += float(corr) * batch_size
        total_count += batch_size

    denom = max(1, total_count)
    return {
        "val/loss": total_flow_loss / denom,
        "val/rmse_scaled": total_rmse_scaled / denom,
        "val/rmse_phys": total_rmse_phys / denom,
        "val/corr_scaled": total_corr / denom,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Phase 3: flow-only residual training on full-depth MLIC arrays")
    parser.add_argument("--truth-npy", type=str, default=None)
    parser.add_argument("--ae-npy", type=str, default=None)
    parser.add_argument("--residual-scaled-npy", type=str, default=None)
    parser.add_argument("--residual-scale-per-depth-npy", type=str, default=None)
    parser.add_argument("--residual-scale-per-depth-out", type=str, default=None)

    parser.add_argument("--scale-mode", type=str, default="std", choices=["std", "iqr", "mad"])
    parser.add_argument("--scale-clamp-quantile", type=float, default=None)

    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--run-name", type=str, default="phase3_flow_only")
    parser.add_argument("--seed", type=int, default=483)

    parser.add_argument("--val-fraction", type=float, default=0.1)
    parser.add_argument("--shuffle-split", action="store_true")

    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=120)

    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--warmup-epochs", type=int, default=10)
    parser.add_argument("--min-lr-ratio", type=float, default=0.05)

    parser.add_argument("--base-channels", type=int, default=64)
    parser.add_argument("--levels", type=int, default=3)
    parser.add_argument("--time-emb-dim", type=int, default=128)
    parser.add_argument("--sample-steps", type=int, default=50)

    parser.add_argument("--cond-normalization", type=str, default="per_depth_zscore", choices=["none", "per_depth_zscore"])
    parser.add_argument("--device", type=str, default="cuda")

    return parser.parse_args()




if __name__ == "__main__":



    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    xp_name = "flow_only_std_scaled" +f"_{timestamp}"



    config = {
        "ckpt_file": "/Odyssey/private/o23gauvr/code/MLIC/experiments/test_mean_std_along_depth/fixed_weight_loss_64_96_1.0_CR_10000.0_enatl_natl__mean_std_along_depth/20260726_101454/checkpoints/best_checkpoint_rmse.pth.tar",
        "dm_path": "/Odyssey/private/o23gauvr/code/FASCINATION/pickle/enatl_natl_dm_157_196_256_norm_per_split_filtered_z_uniform_alternate_days_7_60_10.pkl",
        "seed": 42,
        "device": "cuda",
        "batch_size": 4,
        "num_workers": 20,
        "epochs": 200,
        "lr": 1e-4,
        "weight_decay": 1e-5,
        "grad_clip": 1.0,
        "warmup_epochs": 10,
        "min_lr_ratio": 0.05,
        "output_dir": "/Odyssey/private/o23gauvr/code/FASCINATION/experiments/residual_flow_full_mlic/residual_flow_outputs",
        "xp_name": xp_name,
        "residual_scaled": True,
        "scale_mode": "std", #"std", "iqr", "mad"
        "clamp_quantile": None,
        "cond_normalization": "per_depth_zscore",  # "none", "per_depth_zscore", "global_mean_std", "per_depth_min_max", "global_min_max"
        "val_fraction": 0.1,
        "shuffle_split": True,
        "base_channels": 64,
        "levels": 3,
        "time_emb_dim": 128,
        "sample_steps": 50,
    }

    set_seed(config["seed"])

    truth_np, ae_np = load_truth_and_ae_unorm_cpu(
        checkpoint_file=config["ckpt_file"],
        datamodule_pickle_path=config["dm_path"],
        batch_size=config["batch_size"],
        crop_idx=20,
        device=config["device"],
        verbose=True,
    )



    if config["device"] == "cuda" and torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    output_root = Path(config["output_dir"]) / xp_name
    ckpt_dir = output_root / "checkpoints"
    tb_dir = output_root / "tensorboard"
    output_root.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    residual, residual_scaled, depth_scale_np = get_residual(
        truth_npy=truth_np,
        ae_npy=ae_np,
        scale_mode=config["scale_mode"],
        scale_clamp_quantile=config["clamp_quantile"],
    )

    n_samples, channels, _, _ = residual_scaled.shape

    train_idx, val_idx = split_indices(
        n_samples=n_samples,
        val_fraction=config["val_fraction"],
        seed=config["seed"],
        shuffle=config["shuffle_split"],
    )

    if config["cond_normalization"] == "per_depth_zscore":
        cond_scaled, cond_mean, cond_std = normalize_condition_mean_std_per_depth(ae_np, train_idx=train_idx)
    elif config["cond_normalization"] == "global_mean_std":
        cond_scaled, cond_mean, cond_std = normalize_condition_mean_std_global(ae_np, train_idx=train_idx)
    elif config["cond_normalization"] == "per_depth_min_max":
        cond_scaled, cond_mean, cond_std = normalize_condition_min_max_per_depth(ae_np, train_idx=train_idx)
    elif config["cond_normalization"] == "global_min_max":
        cond_scaled, cond_mean, cond_std = normalize_condition_min_max_global(ae_np, train_idx=train_idx)
    else:
        cond_scaled = ae_np
        cond_mean = np.zeros((channels,), dtype=np.float32)
        cond_std = np.ones((channels,), dtype=np.float32)


    if config["residual_scaled"]:
        train_ds = ResidualArrayDataset(residual_scaled[train_idx], cond_scaled[train_idx])
        val_ds = ResidualArrayDataset(residual_scaled[val_idx], cond_scaled[val_idx])
    else:
        train_ds = ResidualArrayDataset(residual[train_idx], cond_scaled[train_idx])
        val_ds = ResidualArrayDataset(residual[val_idx], cond_scaled[val_idx])

    train_loader = DataLoader(
        train_ds,
        batch_size=config["batch_size"],
        shuffle=True,
        num_workers=config["num_workers"],
        pin_memory=(device.type == "cuda"),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=config["batch_size"],
        shuffle=False,
        num_workers=config["num_workers"],
        pin_memory=(device.type == "cuda"),
    )

    model = FlowResidualNet(
        channels=channels,
        cond_channels=channels,
        base_channels=config["base_channels"],
        levels=config["levels"],
        time_emb_dim=config["time_emb_dim"],
    ).to(device)

    optimizer = AdamW(model.parameters(), lr=config["lr"], weight_decay=config["weight_decay"])
    scheduler = make_scheduler(
        optimizer=optimizer,
        epochs=config["epochs"],
        warmup_epochs=config["warmup_epochs"],
        min_lr_ratio=config["min_lr_ratio"],
    )

    writer = SummaryWriter(log_dir=str(tb_dir))
    depth_scale_t = torch.from_numpy(depth_scale_np.astype(np.float32))

    checkpoint_rules = {
        "val/loss": MetricRule(mode="min"),
        "val/rmse_scaled": MetricRule(mode="min"),
        "val/rmse_phys": MetricRule(mode="min"),
        "val/corr_scaled": MetricRule(mode="max"),
    }
    best_metrics: Dict[str, Optional[float]] = {key: None for key in checkpoint_rules}

    epoch_rows = []
    for epoch in range(1, config["epochs"] + 1):
        train_loss = train_one_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            device=device,
            grad_clip=config["grad_clip"],
        )

        val_metrics = evaluate(
            model=model,
            loader=val_loader,
            device=device,
            sample_steps=config["sample_steps"],
            depth_scale=depth_scale_t,
        )

        scheduler.step()
        current_lr = float(optimizer.param_groups[0]["lr"])

        writer.add_scalar("train/loss", train_loss, epoch)
        writer.add_scalar("train/lr", current_lr, epoch)
        for key, value in val_metrics.items():
            writer.add_scalar(key, value, epoch)

        row = {
            "epoch": epoch,
            "train/loss": train_loss,
            "train/lr": current_lr,
            **val_metrics,
        }
        epoch_rows.append(row)

        save_checkpoint(
            path=ckpt_dir / "last_checkpoint.pth.tar",
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            epoch=epoch,
            metrics=row,
            args=config,
        )

        for metric_name, rule in checkpoint_rules.items():
            metric_value = float(val_metrics[metric_name])
            if checkpoint_better(metric_value, best_metrics[metric_name], rule.mode):
                best_metrics[metric_name] = metric_value
                metric_tag = metric_name.replace("/", "_")
                save_checkpoint(
                    path=ckpt_dir / f"best_checkpoint_{metric_tag}.pth.tar",
                    model=model,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    epoch=epoch,
                    metrics=row,
                    args=config,
                )

        print(
            f"epoch={epoch:03d}/{config['epochs']:03d} "
            f"train_loss={train_loss:.6f} "
            f"val_loss={val_metrics['val/loss']:.6f} "
            f"val_rmse_scaled={val_metrics['val/rmse_scaled']:.6f} "
            f"val_rmse_phys={val_metrics['val/rmse_phys']:.6f} "
            f"val_corr_scaled={val_metrics['val/corr_scaled']:.6f} "
            f"lr={current_lr:.3e}"
        )

    writer.close()

    history_csv = output_root / "history.csv"
    fieldnames = list(epoch_rows[0].keys()) if epoch_rows else []
    with history_csv.open("w", newline="", encoding="utf-8") as handle:
        writer_csv = csv.DictWriter(handle, fieldnames=fieldnames)
        writer_csv.writeheader()
        writer_csv.writerows(epoch_rows)

    summary = {
        "run_name": xp_name,
        "n_samples": int(n_samples),
        "channels": int(channels),
        "train_samples": int(train_idx.size),
        "val_samples": int(val_idx.size),
        "best_metrics": best_metrics,
        "tensorboard_dir": str(tb_dir),
        "checkpoints_dir": str(ckpt_dir),
        "history_csv": str(history_csv),
        "condition_normalization": config["cond_normalization"],
        "scale_mode": config["scale_mode"],
        "scale_clamp_quantile": config["clamp_quantile"],
        "cond_mean_per_depth": cond_mean.astype(np.float32).tolist(),
        "cond_std_per_depth": cond_std.astype(np.float32).tolist(),
    }
    summary_path = output_root / "run_summary.json"
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)

    print("Phase 3 flow-only training complete")
    print(f"Output directory: {output_root}")
    print(f"Summary: {summary_path}")
