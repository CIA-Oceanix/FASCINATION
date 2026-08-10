#!/usr/bin/env python3

"""Bootstrap codec-only inference for Phase 0/1 full-depth MLIC++.

This module provides a single entrypoint that:
1. Loads one MLIC++ checkpoint.
2. Loads one datamodule pickle.
3. Runs batched codec inference on test samples.
4. Unnormalizes truth and reconstruction with TEST norm stats only.
5. Returns CPU tensors: truth_unnorm, ae_unnorm.
"""

from __future__ import annotations

import argparse
import pickle
import re
import sys
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np
import torch
import torch.nn as nn
import xarray as xr
from scipy.signal import butter, filtfilt
from tqdm import tqdm

# Ensure imports work whether launched from workspace root or another cwd.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from MLIC.MLIC.models import MLICPlusPlus
from MLIC.MLIC.utils.utils import Config
from FASCINATION.src.utils import norm_ssp_arr_3D, unorm_ssp_arr_3D


def parse_mlic_config(ckpt_path: Path) -> Dict[str, Any]:
    """Parse MLIC experiment_config.log into a config dictionary."""
    config_files = sorted((ckpt_path.parent.parent).rglob("experiment_config.log"))
    if not config_files:
        raise FileNotFoundError(
            f"experiment_config.log not found near checkpoint: {ckpt_path}"
        )

    config_file = config_files[0]
    content = config_file.read_text(encoding="utf-8")

    def _extract_section(section_name: str) -> str:
        header_re = re.compile(rf"{re.escape(section_name)}\n-+\n", flags=re.MULTILINE)
        start_match = header_re.search(content)
        if not start_match:
            return ""

        next_header_re = re.compile(r"\n[A-Z][A-Z _]+:\n-+\n", flags=re.MULTILINE)
        next_match = next_header_re.search(content, start_match.end())
        end_idx = next_match.start() if next_match else len(content)
        return content[start_match.end() : end_idx].strip()

    def _parse_key_value_blocks(block: str) -> Dict[str, str]:
        if not block:
            return {}

        entry_re = re.compile(r"^([A-Za-z_][A-Za-z0-9_]*)\s*:\s*", flags=re.MULTILINE)
        matches = list(entry_re.finditer(block))
        parsed: Dict[str, str] = {}

        for i, match in enumerate(matches):
            key = match.group(1)
            value_start = match.end()
            value_end = matches[i + 1].start() if i + 1 < len(matches) else len(block)
            parsed[key] = block[value_start:value_end].strip()

        return parsed

    def _safe_eval_model_value(raw_value: str) -> Any:
        value = raw_value.replace("<class 'torch.nn.modules.activation.", "nn.").replace("'>", "")
        try:
            return eval(
                value,
                {"__builtins__": {}},
                {"nn": nn, "np": np, "True": True, "False": False, "None": None},
            )
        except Exception:
            return raw_value

    model_block = _extract_section("MODEL CONFIGURATION:")
    model_cfg_raw = _parse_key_value_blocks(model_block)
    cfg: Dict[str, Any] = {k: _safe_eval_model_value(v) for k, v in model_cfg_raw.items()}

    if not cfg:
        raise ValueError(f"Could not parse MODEL CONFIGURATION from {config_file}")

    return cfg


def _load_mlic_model(
    checkpoint_file: Path, device: torch.device
) -> Tuple[torch.nn.Module, Dict[str, Any], Dict[str, Any]]:
    """Load a full-depth MLIC++ model from a checkpoint path."""
    if "MLIC" not in str(checkpoint_file):
        raise ValueError(
            "Only MLIC++ checkpoints are supported in this bootstrap utility. "
            f"Got: {checkpoint_file}"
        )

    if not checkpoint_file.exists():
        raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_file}")

    cfg = parse_mlic_config(checkpoint_file)
    model = MLICPlusPlus(config=Config(cfg))

    checkpoint = torch.load(checkpoint_file, map_location=device, weights_only=False)
    if "state_dict" not in checkpoint:
        raise KeyError(f"Checkpoint missing 'state_dict': {checkpoint_file}")

    model.load_state_dict(checkpoint["state_dict"], strict=False)
    model = model.to(device)
    model.eval()

    for param in model.parameters():
        param.requires_grad = False

    return model, checkpoint, cfg


def _resolve_model_test_norm_stats(
    checkpoint_state: Dict[str, Any],
    dm_test_norm: Dict[str, Any],
) -> Dict[str, Any]:
    """Resolve the test-time normalization used by the codec path."""
    if "test_norm_stats" in checkpoint_state:
        return deepcopy(checkpoint_state["test_norm_stats"])

    model_norm_stats = deepcopy(dm_test_norm)
    model_norm_stats["method"] = "mean_std_along_depth"
    return model_norm_stats


def _get_model_input_truth(
    ssp_truth_unnorm_da: xr.DataArray,
    model_norm_stats: Dict[str, Any],
    cfg: Dict[str, Any],
) -> Tuple[np.ndarray, np.ndarray | None]:
    """Build normalized model input exactly as in the metric pipeline."""
    uniform_z_flag = cfg.get("datamodule_info", {}).get("uniform_z", True)
    if uniform_z_flag == "False":
        non_uniform_z_raw = cfg.get("datamodule_info", {}).get("depth_array", None)
        if non_uniform_z_raw is None:
            raise KeyError("Config indicates non-uniform z but datamodule_info.depth_array is missing")

        non_uniform_z = np.fromstring(
            str(non_uniform_z_raw).replace("\n", " ").strip("[]"),
            sep=" ",
        )
        truth_for_model = ssp_truth_unnorm_da.interp(
            z=non_uniform_z,
            method="cubic",
            kwargs={"fill_value": "extrapolate"},
        )
        truth_norm_np = norm_ssp_arr_3D(truth_for_model, model_norm_stats).values.astype(np.float32)
        return truth_norm_np, non_uniform_z

    truth_norm_np = norm_ssp_arr_3D(ssp_truth_unnorm_da, model_norm_stats).values.astype(np.float32)
    return truth_norm_np, None


def _postprocess_reconstruction(
    ssp_ae_norm_np: np.ndarray,
    model_norm_stats: Dict[str, Any],
    truth_unnorm_da: xr.DataArray,
    original_depth_array: np.ndarray,
    cfg: Dict[str, Any],
) -> np.ndarray:
    """Mirror decoder-side post-processing from the metric pipeline."""

    ssp_ae = unorm_ssp_arr_3D(ssp_ae_norm_np, model_norm_stats).astype(np.float32)

    if not cfg.get("output_low_band_filter_use", False):
        b, a = butter(N=2, Wn=0.107, btype="low", analog=False)
        ssp_ae = filtfilt(b, a, ssp_ae, axis=1).astype(np.float32)

    ssp_ae_da = xr.DataArray(
        ssp_ae,
        coords=truth_unnorm_da.coords,
        dims=truth_unnorm_da.dims,
        name="ssp_reconstructed",
    )

    if cfg.get("datamodule_info", {}).get("uniform_z", True) == "False":
        ssp_ae_da = ssp_ae_da.interp(z=original_depth_array)

    return ssp_ae_da.values.astype(np.float32)


def _run_batched_inference(
    model: torch.nn.Module,
    truth_norm_np: np.ndarray,
    batch_size: int,
    device: torch.device,
    verbose: bool = False,
) -> torch.Tensor:
    """Run model inference in batches and return normalized reconstruction on CPU."""
    if batch_size < 1:
        raise ValueError(f"batch_size must be >= 1, got {batch_size}")

    truth_norm_t = torch.from_numpy(truth_norm_np).to(dtype=torch.float32, device="cpu")
    outputs = []

    iterator = range(0, truth_norm_t.shape[0], batch_size)
    if verbose:
        iterator = tqdm(iterator, desc="Codec inference", total=(truth_norm_t.shape[0] + batch_size - 1) // batch_size)

    with torch.inference_mode():
        for start in iterator:
            end = min(start + batch_size, truth_norm_t.shape[0])
            batch = truth_norm_t[start:end].to(device)
            out = model(batch)

            if isinstance(out, dict):
                if "x_hat" not in out:
                    raise KeyError("MLIC output dict does not contain 'x_hat'")
                x_hat = out["x_hat"]
            elif torch.is_tensor(out):
                x_hat = out
            else:
                raise TypeError(f"Unexpected model output type: {type(out)}")

            outputs.append(x_hat.detach().cpu())

    ae_norm_cpu = torch.cat(outputs, dim=0)
    return ae_norm_cpu.detach().cpu().to(dtype=torch.float32).numpy()


def load_truth_and_ae_unorm_cpu(
    checkpoint_file: str,
    datamodule_pickle_path: str,
    batch_size: int,
    crop_idx: int | None = None,
    device: str | None = None,
    verbose: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return unnormalized truth and reconstructed tensors on CPU.

    Parameters
    ----------
    checkpoint_file : str
        Path to one MLIC++ checkpoint (.tar).
    datamodule_pickle_path : str
        Path to a pickled datamodule containing test_ds.input.
    batch_size : int
        Inference batch size along time axis.
    device : str | None
        Device for inference. If None, use cuda when available else cpu.
    verbose : bool
        If True, print progress information.

    Returns
    -------
    Tuple[torch.Tensor, torch.Tensor]
        (truth_unnorm_cpu, ae_unnorm_cpu), both float32 CPU tensors.
    """
    ckpt_path = Path(checkpoint_file)
    dm_path = Path(datamodule_pickle_path)

    crop_slice = slice(crop_idx, -crop_idx) if crop_idx is not None else slice(None)

    if not dm_path.exists():
        raise FileNotFoundError(f"Datamodule pickle not found: {dm_path}")

    runtime_device = torch.device(device if device is not None else ("cuda" if torch.cuda.is_available() else "cpu"))

    with dm_path.open("rb") as f:
        dm = pickle.load(f)

    if not hasattr(dm, "test_ds") or not hasattr(dm.test_ds, "input"):
        raise AttributeError("Datamodule pickle must expose dm.test_ds.input")

    ssp_truth_da = dm.test_ds.input

    if "norm_stats" not in ssp_truth_da.attrs:
        raise KeyError("dm.test_ds.input.attrs does not contain 'norm_stats'")

    # Phase requirement: always use TEST norm stats (never train stats here).
    test_norm_stats = deepcopy(ssp_truth_da.attrs["norm_stats"])

    original_depth_array = ssp_truth_da.z.values
    ssp_truth_unnorm_da = unorm_ssp_arr_3D(ssp_truth_da, test_norm_stats)

    model, checkpoint_state, cfg = _load_mlic_model(ckpt_path, runtime_device)
    model_norm_stats = _resolve_model_test_norm_stats(checkpoint_state, test_norm_stats)
    ssp_truth_norm_np, _ = _get_model_input_truth(
        ssp_truth_unnorm_da=ssp_truth_unnorm_da,
        model_norm_stats=model_norm_stats,
        cfg=cfg,
    )

    if ssp_truth_norm_np.ndim != 4:
        raise ValueError(
            f"Expected test tensor with 4 dims (T, C, H, W), got shape {ssp_truth_norm_np.shape}"
        )

    ssp_ae_norm_np = _run_batched_inference(
        model=model,
        truth_norm_np=ssp_truth_norm_np,
        batch_size=int(batch_size),
        device=runtime_device,
        verbose=verbose,
    )



    ae_unnorm_np = _postprocess_reconstruction(
        ssp_ae_norm_np=ssp_ae_norm_np,
        model_norm_stats=model_norm_stats,
        truth_unnorm_da=ssp_truth_unnorm_da,
        original_depth_array=original_depth_array,
        cfg=cfg,
    )

    if ssp_ae_norm_np.shape != ssp_truth_norm_np.shape:
        raise RuntimeError(
            "Reconstruction shape mismatch: "
            f"truth={ssp_truth_norm_np.shape} vs ae={ssp_ae_norm_np.shape}"
        )

    truth_unnorm_np = ssp_truth_unnorm_da.values.astype(np.float32)

    ae_unnorm_np = ae_unnorm_np[:,:, crop_slice, crop_slice] 
    truth_unnorm_np = truth_unnorm_np[:,:, crop_slice, crop_slice]


    if not np.isfinite(truth_unnorm_np).all():
        raise RuntimeError("Non-finite values detected in unnormalized truth tensor")
    if not np.isfinite(ae_unnorm_np).all():
        raise RuntimeError("Non-finite values detected in unnormalized reconstruction tensor")

    return truth_unnorm_np, ae_unnorm_np


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run codec-only MLIC++ inference and return unnormalized CPU tensors"
    )
    parser.add_argument("--checkpoint_file", type=str,default="/Odyssey/private/o23gauvr/code/MLIC/experiments/test_mean_std_along_depth/fixed_weight_loss_64_96_1.0_CR_10000.0_enatl_natl__mean_std_along_depth/20260726_101454/checkpoints/best_checkpoint_rmse.pth.tar", help="Path to one MLIC++ checkpoint (.tar)")
    parser.add_argument("--datamodule_pickle_path", type=str, default="/Odyssey/private/o23gauvr/code/FASCINATION/pickle/enatl_natl_dm_157_196_256_norm_per_split_filtered_z_uniform_alternate_days_7_60_10.pkl", help="Path to datamodule pickle")
    parser.add_argument("--batch_size", type=int, default=8, help="Inference batch size")
    parser.add_argument("--device", type=str, default=None, help="Inference device (e.g., 'cuda' or 'cpu')")
    parser.add_argument("--verbose", action="store_true", default=False, help="Enable progress output")
    return parser


def main() -> None:
    args = _build_arg_parser().parse_args()
    truth_np, ae_np = load_truth_and_ae_unorm_cpu(
        checkpoint_file=args.checkpoint_file,
        datamodule_pickle_path=args.datamodule_pickle_path,
        batch_size=args.batch_size,
        crop_idx=None,
        device=args.device,
        verbose=args.verbose,
    )

    print("Bootstrap inference complete")
    print(f"truth shape: {tuple(truth_np.shape)}, dtype: {truth_np.dtype}")
    print(f"ae shape: {tuple(ae_np.shape)}, dtype: {ae_np.dtype}")
    print(f"truth min/max: {float(truth_np.min()):.6f} / {float(truth_np.max()):.6f}")
    print(f"ae min/max: {float(ae_np.min()):.6f} / {float(ae_np.max()):.6f}")



if __name__ == "__main__":
    main()
