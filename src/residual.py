import sys
import os

running_path = "/Odyssey/private/o23gauvr/code/"
os.chdir(running_path)
sys.path.insert(0,running_path)
sys.path.insert(0, "/Odyssey/private/o23gauvr/code/FASCINATION")


import numpy as np
import json
from pathlib import Path
from typing import Any, Dict, Tuple
import matplotlib.pyplot as plt

from FASCINATION.src.bootstrap_codec_inference import load_truth_and_ae_unorm_cpu



def compute_residual_target(truth_np: np.ndarray, ae_np: np.ndarray) -> np.ndarray:
    """Compute residual target r = x - x_tilde for Phase 2 experiments."""
    if truth_np.shape != ae_np.shape:
        raise ValueError(
            f"truth/ae shape mismatch for residual target: {truth_np.shape} vs {ae_np.shape}"
        )
    if truth_np.ndim != 4:
        raise ValueError(f"Expected shape (T, C, H, W), got {truth_np.shape}")

    residual_np = truth_np.astype(np.float32, copy=False) - ae_np.astype(np.float32, copy=False)
    if not np.isfinite(residual_np).all():
        raise RuntimeError("Non-finite values detected in residual target")
    return residual_np


def _compute_depth_scale(residual_np: np.ndarray, scale_mode: str, eps: float) -> np.ndarray:
    """Compute per-depth scale vector for residual normalization."""
    if scale_mode == "std":
        scale = residual_np.std(axis=(0, 2, 3), ddof=0)
    elif scale_mode == "iqr":
        q75 = np.percentile(residual_np, 75.0, axis=(0, 2, 3))
        q25 = np.percentile(residual_np, 25.0, axis=(0, 2, 3))
        scale = q75 - q25
    elif scale_mode == "mad":
        med = np.median(residual_np, axis=(0, 2, 3))
        abs_dev = np.abs(residual_np - med[None, :, None, None])
        mad = np.median(abs_dev, axis=(0, 2, 3))
        scale = 1.4826 * mad
    else:
        raise ValueError(f"Unknown scale_mode={scale_mode}. Expected one of: std, iqr, mad")

    scale = np.asarray(scale, dtype=np.float32)
    scale = np.maximum(scale, np.float32(eps))
    return scale


def scale_residual_target(
    residual_np: np.ndarray,
    scale_mode: str = "std",
    eps: float = 1e-6,
    clamp_quantile: float | None = None,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """Scale residual target with per-depth statistics and optional clipping."""
    if residual_np.ndim != 4:
        raise ValueError(f"Expected residual shape (T, C, H, W), got {residual_np.shape}")

    depth_scale = _compute_depth_scale(residual_np, scale_mode=scale_mode, eps=eps)
    residual_scaled = residual_np / depth_scale[None, :, None, None]

    clamp_info: Dict[str, Any] = {"enabled": False}
    if clamp_quantile is not None:
        q = float(clamp_quantile)
        if not (0.0 < q <= 1.0):
            raise ValueError(f"clamp_quantile must be in (0, 1], got {q}")
        abs_q = np.quantile(np.abs(residual_scaled), q, axis=(0, 2, 3))
        abs_q = np.maximum(abs_q.astype(np.float32), np.float32(eps))
        residual_scaled = np.clip(
            residual_scaled,
            -abs_q[None, :, None, None],
            abs_q[None, :, None, None],
        )
        clamp_info = {
            "enabled": True,
            "quantile": q,
            "abs_bound_per_depth": abs_q.tolist(),
        }

    if not np.isfinite(residual_scaled).all():
        raise RuntimeError("Non-finite values detected in scaled residual target")

    return residual_scaled.astype(np.float32), depth_scale.astype(np.float32), clamp_info


def summarize_residual_stats(
    residual_np: np.ndarray,
    residual_scaled_np: np.ndarray,
    depth_scale: np.ndarray,
    scale_mode: str,
    clamp_info: Dict[str, Any],
) -> Dict[str, Any]:
    """Build Phase 2 residual statistics dictionary (global + per-depth)."""
    per_depth_mean = residual_np.mean(axis=(0, 2, 3))
    per_depth_std = residual_np.std(axis=(0, 2, 3), ddof=0)
    per_depth_q05 = np.percentile(residual_np, 5.0, axis=(0, 2, 3))
    per_depth_q50 = np.percentile(residual_np, 50.0, axis=(0, 2, 3))
    per_depth_q95 = np.percentile(residual_np, 95.0, axis=(0, 2, 3))

    stats = {
        "shape": {
            "time": int(residual_np.shape[0]),
            "depth": int(residual_np.shape[1]),
            "lat": int(residual_np.shape[2]),
            "lon": int(residual_np.shape[3]),
        },
        "residual_global": {
            "mean": float(residual_np.mean()),
            "std": float(residual_np.std(ddof=0)),
            "mae": float(np.mean(np.abs(residual_np))),
            "min": float(residual_np.min()),
            "max": float(residual_np.max()),
            "q01": float(np.percentile(residual_np, 1.0)),
            "q99": float(np.percentile(residual_np, 99.0)),
        },
        "scaling": {
            "mode": scale_mode,
            "depth_scale_mean": float(depth_scale.mean()),
            "depth_scale_min": float(depth_scale.min()),
            "depth_scale_max": float(depth_scale.max()),
            "depth_scale_per_depth": depth_scale.astype(np.float32).tolist(),
            "clamp": clamp_info,
        },
        "residual_scaled_global": {
            "mean": float(residual_scaled_np.mean()),
            "std": float(residual_scaled_np.std(ddof=0)),
            "mae": float(np.mean(np.abs(residual_scaled_np))),
            "min": float(residual_scaled_np.min()),
            "max": float(residual_scaled_np.max()),
            "q01": float(np.percentile(residual_scaled_np, 1.0)),
            "q99": float(np.percentile(residual_scaled_np, 99.0)),
        },
        "per_depth": {
            "mean": per_depth_mean.astype(np.float32).tolist(),
            "std": per_depth_std.astype(np.float32).tolist(),
            "q05": per_depth_q05.astype(np.float32).tolist(),
            "q50": per_depth_q50.astype(np.float32).tolist(),
            "q95": per_depth_q95.astype(np.float32).tolist(),
        },
    }
    return stats


def save_residual_artifacts(
    output_dir: str | Path,
    truth_np: np.ndarray,
    ae_np: np.ndarray,
    scale_mode: str = "std",
    clamp_quantile: float | None = None,
    save_arrays: bool = True,
) -> Dict[str, Any]:
    """Create and save residual target/scaled residual artifacts."""
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    residual_np = compute_residual_target(truth_np, ae_np)
    residual_scaled_np, depth_scale, clamp_info = scale_residual_target(
        residual_np=residual_np,
        scale_mode=scale_mode,
        clamp_quantile=clamp_quantile,
    )
    stats = summarize_residual_stats(
        residual_np=residual_np,
        residual_scaled_np=residual_scaled_np,
        depth_scale=depth_scale,
        scale_mode=scale_mode,
        clamp_info=clamp_info,
    )

    with (out_dir / "residual_stats.json").open("w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)

    if save_arrays:
        np.save(out_dir / "truth_unnorm.npy", truth_np.astype(np.float32))
        np.save(out_dir / "ae_unnorm.npy", ae_np.astype(np.float32))
        np.save(out_dir / "residual_target.npy", residual_np.astype(np.float32))
        np.save(out_dir / "residual_target_scaled.npy", residual_scaled_np.astype(np.float32))
        np.save(out_dir / "residual_scale_per_depth.npy", depth_scale.astype(np.float32))


    depth_idx = np.arange(residual_np.shape[1])
    per_depth_std = np.asarray(stats["per_depth"]["std"], dtype=np.float32)
    per_depth_mean = np.asarray(stats["per_depth"]["mean"], dtype=np.float32)
    per_depth_q05 = np.asarray(stats["per_depth"]["q05"], dtype=np.float32)
    per_depth_q95 = np.asarray(stats["per_depth"]["q95"], dtype=np.float32)

    fig, axs = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    axs[0].plot(depth_idx, per_depth_mean, label="mean")
    axs[0].fill_between(depth_idx, per_depth_q05, per_depth_q95, alpha=0.25, label="q05-q95")
    axs[0].set_title("Residual mean and spread per depth")
    axs[0].set_ylabel("Residual")
    axs[0].legend()
    axs[0].grid(True, alpha=0.25)

    axs[1].plot(depth_idx, per_depth_std, label="std", color="tab:orange")
    axs[1].set_title("Residual std per depth")
    axs[1].set_xlabel("Depth index")
    axs[1].set_ylabel("Std")
    axs[1].legend()
    axs[1].grid(True, alpha=0.25)

    fig.tight_layout()
    fig.savefig(out_dir / "residual_stats_per_depth.png", dpi=200)
    plt.close(fig)

    return stats




if __name__ == "__main__":

    ckpt_file = "/Odyssey/private/o23gauvr/code/MLIC/experiments/test_mean_std_along_depth/fixed_weight_loss_64_96_1.0_CR_10000.0_enatl_natl__mean_std_along_depth/20260726_101454/checkpoints/best_checkpoint_rmse.pth.tar"
    dm_path = "/Odyssey/private/o23gauvr/code/FASCINATION/pickle/enatl_natl_dm_157_196_256_norm_per_split_filtered_z_uniform_alternate_days_7_60_10.pkl"

    truth_np, ae_np = load_truth_and_ae_unorm_cpu(
        checkpoint_file=ckpt_file,
        datamodule_pickle_path=dm_path,
        batch_size=8,
        crop_idx=20,
        device=None,
        verbose=True,
    )

    output_dir = "/Odyssey/private/o23gauvr/code/FASCINATION/residual"

    stats = save_residual_artifacts(
        output_dir=output_dir,
        truth_np=truth_np,
        ae_np=ae_np,
        scale_mode="std",  #["std", "iqr", "mad"]
        clamp_quantile=None,
        save_arrays=False,
    )
    print("Phase 2 residual artifacts complete")
    print(f"phase2 output dir: {output_dir}")
    print(
        "residual global mean/std: "
        f"{stats['residual_global']['mean']:.6f} / {stats['residual_global']['std']:.6f}"
    )
    print(
        "scaled residual global mean/std: "
        f"{stats['residual_scaled_global']['mean']:.6f} / {stats['residual_scaled_global']['std']:.6f}"
    )
