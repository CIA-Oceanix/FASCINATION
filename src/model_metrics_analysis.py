"""
Utilities to analyze model_metrics pickle files produced by test_metrics.py.

This module provides:
- Correlation matrix computation across selected metrics.
- Top-k appearance counting with per-metric min/max direction.
- 2-metric model scatter plotting.

CLI examples:
  python -m FASCINATION.src.model_metrics_analysis --pickle-path /path/to/model_metrics.pkl --task corr
  python -m FASCINATION.src.model_metrics_analysis --pickle-path /path/to/model_metrics.pkl --task topk
  python -m FASCINATION.src.model_metrics_analysis --pickle-path /path/to/model_metrics.pkl --task plot --metric-x RMSE --metric-y ECS
"""

from __future__ import annotations

import argparse
import pickle
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from numbers import Number


def load_model_metrics_pickle(pickle_path: Union[str, Path]) -> Dict:
    """Load model_metrics pickle from disk."""
    with open(pickle_path, "rb") as f:
        return pickle.load(f)


def _extract_model_iteration(model_name: str) -> Optional[int]:
    """Extract an iteration number from model name if present."""
    matches = re.findall(r"(?<!\d)(\d{4,})(?!\d)", str(model_name))
    if not matches:
        return None
    return int(matches[-1])


def _build_metrics_dataframe(
    model_metrics: Dict,
    season: str = "all",
    split: str = "all",
) -> pd.DataFrame:
    """Flatten nested model_metrics dict to one row per (model, cr, season)."""
    rows: List[Dict] = []
    selected_split = str(split).lower()
    for model_name, cr_dict in model_metrics.items():
        if not isinstance(cr_dict, dict):
            continue

        model_iter = _extract_model_iteration(model_name)

        for cr, season_dict in cr_dict.items():
            if not isinstance(season_dict, dict):
                continue

            for season_name, metrics_by_split in season_dict.items():
                if season != "*" and str(season_name).lower() != str(season).lower():
                    continue
                if not isinstance(metrics_by_split, dict):
                    continue

                split_names = metrics_by_split.keys() if selected_split == "all" else [split]
                for split_name in split_names:
                    split_metrics = metrics_by_split.get(split_name, {})
                    if not isinstance(split_metrics, dict):
                        continue

                    row = {
                        "model": str(model_name),
                        "iteration": model_iter,
                        "cr": float(cr) if isinstance(cr, (int, float, np.floating)) else np.nan,
                        "season": str(season_name),
                        "split": str(split_name),
                    }

                    for k, v in split_metrics.items():
                        if isinstance(v, Number):
                            row[str(k).lower()] = float(v)

                    rows.append(row)

    return pd.DataFrame(rows)


def compute_metrics_correlation_matrix(
    model_metrics: Dict,
    metrics: Union[str, List[str]] = "all",
    season: str = "all",
    split: str = "all",
    method: str = "pearson",
) -> pd.DataFrame:
    """
    Compute a correlation matrix across metrics.

    Parameters
    ----------
    model_metrics : Dict
        Loaded model_metrics dictionary.
    metrics : Union[str, List[str]], optional
        Metrics to include. Use "all" (default) for all available numeric metrics.
    season : str, optional
        Season to select (default: "all"). Use "*" to include all seasons.
    split : str, optional
        Split to read in each season dict ("SSP", "GRAD", or "all").
    method : str, optional
        Correlation method supported by pandas ("pearson", "spearman", "kendall").
    """
    df = _build_metrics_dataframe(model_metrics=model_metrics, season=season, split=split)
    if df.empty:
        raise ValueError("No rows available for the selected season/split.")

    df = df.drop(
        [
            "num_samples",
            "tl_grid_n_profiles",
            "nsr_spatial_depth0_time_median_resolution_1",
            "nsr_spatial_depth0_time_std_resolution_1",
        ],
        axis=1,
        errors="ignore",
    )

    metadata_cols = {"model", "iteration", "cr", "season", "split"}
    numeric_cols = [
        c for c in df.columns if c not in metadata_cols and np.issubdtype(df[c].dtype, np.number)
    ]

    if isinstance(metrics, str) and metrics.lower() == "all":
        selected_cols = numeric_cols
    else:
        wanted = [m.lower() for m in metrics]
        missing = [m for m in wanted if m not in numeric_cols]
        if missing:
            raise ValueError(f"Unknown or non-numeric metrics: {missing}")
        selected_cols = wanted

    if len(selected_cols) < 2:
        raise ValueError("Need at least two metrics to compute a correlation matrix.")

    return df[selected_cols].corr(method=method)


def count_topk_metric_appearances(
    model_metrics: Dict,
    metrics: Union[str, List[str]] = "all",
    season: str = "all",
    split: str = "all",
    top_k: int = 10,
    metric_directions: Optional[Dict[str, str]] = None,
    group_by: str = "iteration",
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Count how many times each model (or iteration) appears in top-k across metrics.

    metric_directions maps metric -> "min" or "max".
    If a metric is not provided in metric_directions, default is "min".
    """
    df = _build_metrics_dataframe(model_metrics=model_metrics, season=season, split=split)
    if df.empty:
        raise ValueError("No rows available for the selected season/split.")

    metadata_cols = {"model", "iteration", "cr", "season", "split"}
    numeric_cols = [
        c for c in df.columns if c not in metadata_cols and np.issubdtype(df[c].dtype, np.number)
    ]

    if isinstance(metrics, str) and metrics.lower() == "all":
        selected_metrics = numeric_cols
    else:
        selected_metrics = [m.lower() for m in metrics]
        missing = [m for m in selected_metrics if m not in numeric_cols]
        if missing:
            raise ValueError(f"Unknown or non-numeric metrics: {missing}")

    if metric_directions is None:
        metric_directions = {
            "rmse": "min",
            "mae": "min",
            "mse": "min",
            "psnr": "max",
            "ecs": "min",
            "extremum_pos": "min",
            "f1_score": "max",
            "pearson": "max",
            "r2_score": "max",
            "dtw": "min",
            "wasserstein": "min",
            "power_wasserstein": "min",
            "lsd": "min",
            "peak_freq_error": "min",
            "ssim": "max",
            "ms_ssim": "max",
            "tl_grid_mae_freq_10": "min",
            "tl_grid_mae_freq_500": "min",
            "tl_grid_mae_freq_1000": "min",
            "tl_grid_ssim_freq_10": "max",
            "tl_grid_ssim_freq_500": "max",
            "tl_grid_ssim_freq_1000": "max",
        }

    metric_directions = {k.lower(): v.lower() for k, v in metric_directions.items()}

    if group_by not in {"iteration", "model"}:
        raise ValueError("group_by must be 'iteration' or 'model'")

    top_rows: List[Dict] = []
    for metric in selected_metrics:
        metric_series = df[["model", "iteration", "cr", metric]].dropna(subset=[metric])
        if metric_series.empty:
            continue

        direction = metric_directions.get(metric, "min")
        if direction not in {"min", "max"}:
            raise ValueError(f"Invalid direction '{direction}' for metric '{metric}'. Use 'min' or 'max'.")

        ascending = direction == "min"
        ranked = metric_series.sort_values(metric, ascending=ascending).head(int(top_k)).reset_index(drop=True)

        for rank_idx, (_, row) in enumerate(ranked.iterrows(), start=1):
            top_rows.append(
                {
                    "metric": metric,
                    "direction": direction,
                    "rank": int(rank_idx),
                    "model": row["model"],
                    "iteration": row["iteration"],
                    "cr": row["cr"],
                    "value": float(row[metric]),
                }
            )

    details_df = pd.DataFrame(top_rows)
    if details_df.empty:
        return pd.DataFrame(columns=[group_by, "n_topk"]), details_df

    counts_df = (
        details_df.groupby(group_by, dropna=False)
        .size()
        .rename("n_topk")
        .reset_index()
        .sort_values("n_topk", ascending=False)
        .reset_index(drop=True)
    )

    return counts_df, details_df


def plot_models_by_two_metrics(
    model_metrics: Dict,
    metric_x: str,
    metric_y: str,
    season: str = "all",
    split: str = "all",
    annotate: bool = False,
    figsize: Tuple[float, float] = (9, 6),
) -> Tuple[plt.Figure, plt.Axes, pd.DataFrame]:
    """Scatter plot of models with x/y metrics."""
    df = _build_metrics_dataframe(model_metrics=model_metrics, season=season, split=split)
    if df.empty:
        raise ValueError("No rows available for the selected season/split.")

    x_col = metric_x.lower()
    y_col = metric_y.lower()
    missing = [m for m in (x_col, y_col) if m not in df.columns]
    if missing:
        raise ValueError(f"Metric(s) not found in dataframe: {missing}")

    plot_df = df[["model", "iteration", "cr", x_col, y_col]].dropna().copy()
    if plot_df.empty:
        raise ValueError("No valid points to plot after dropping NaNs.")

    fig, ax = plt.subplots(figsize=figsize)
    sc = ax.scatter(plot_df[x_col], plot_df[y_col], c=plot_df["cr"], cmap="viridis", alpha=0.85)
    cbar = plt.colorbar(sc, ax=ax)
    cbar.set_label("Compression ratio (cr)")

    if annotate:
        for _, row in plot_df.iterrows():
            label = str(row["iteration"]) if not pd.isna(row["iteration"]) else str(row["model"])
            ax.annotate(label, (row[x_col], row[y_col]), fontsize=8, alpha=0.8)

    ax.set_xlabel(metric_x.upper())
    ax.set_ylabel(metric_y.upper())
    ax.set_title(f"Models by {metric_x.upper()} vs {metric_y.upper()} ({split}, season={season})")
    ax.grid(alpha=0.25)
    fig.tight_layout()

    return fig, ax, plot_df


def plot_correlation_matrix(
    corr_df: pd.DataFrame,
    figsize: Tuple[float, float] = (11, 9),
    cmap: str = "coolwarm",
    annotate: bool = True,
) -> Tuple[plt.Figure, plt.Axes]:
    """Visualize a correlation matrix as a heatmap."""
    if corr_df.empty:
        raise ValueError("Correlation DataFrame is empty.")
    corr_df = corr_df.abs()
    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(corr_df.values, cmap=cmap, vmin=0.0, vmax=1.0)

    ax.set_xticks(np.arange(len(corr_df.columns)))
    ax.set_yticks(np.arange(len(corr_df.index)))
    ax.set_xticklabels(corr_df.columns, rotation=90)
    ax.set_yticklabels(corr_df.index)

    if annotate:
        for i in range(corr_df.shape[0]):   
            for j in range(corr_df.shape[1]):
                val = corr_df.iat[i, j]
                text_color = "white" if val > 0.5 else "black"
                ax.text(j, i, f"{val:.2f}", ha="center", va="center", color=text_color, fontsize=8)

    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label("Correlation")
    ax.set_title("Absolute Correlation Matrix Heatmap")
    fig.tight_layout()
    return fig, ax


def _ensure_output_dir(path_like: Union[str, Path]) -> Path:
    """Create and return output directory path."""
    out_dir = Path(path_like)
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


def _parse_metric_list(text: Union[str, List[str]]) -> Union[str, List[str]]:
    if isinstance(text, list):
        return text
    text =text.strip()
    if text.lower() == "all":
        return "all"
    else:
        raise ValueError("Metrics must be a list of strings or 'all'")


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze model_metrics pickle results.")
    parser.add_argument("--pickle-path", default="/Odyssey/private/o23gauvr/code/FASCINATION/pickle/model_metrics_compare_test_along_depth_test.pkl", help="Path to model_metrics pickle file")
    parser.add_argument("--task", choices=["corr", "topk", "plot", "all"], default="all")
    parser.add_argument("--season", default="all", help="Season to select (default: all, use * for all seasons)")
    parser.add_argument("--split", default="all", choices=["SSP", "GRAD", "all"], help="Metrics split")

    parser.add_argument("--metrics", default=['rmse', 'mae', 'psnr', 'ecs', 'extremum_pos', 'f1_score', 'pearson', 'pearson_pvalue', 'r2_score', 'dtw', 'wasserstein', 'lsd', 'peak_freq_error', 'power_wasserstein', 'ssim', 'ms_ssim', 'rmse_pca_3_components', 'rmse_pca_6_components', 'nsr_depth_resolution_10', 'nsr_spatial_resolution_10', 'nsr_spatial_depth0_time_mean_resolution_10', 'nsr_spatial_depth40_time_mean_resolution_10', 'tl_grid_mae_freq_10', 'tl_grid_mae_freq_500','tl_grid_mae_freq_1000','tl_grid_ssim_freq_10','tl_grid_ssim_freq_500', 'tl_grid_ssim_freq_1000'], help="Comma-separated list of metrics or 'all'") #'nsr_depth_resolution_1', 'nsr_spatial_resolution_1', 'nsr_spatial_depth0_time_mean_resolution_1', 'nsr_spatial_depth40_time_mean_resolution_1', 'nsr_depth_resolution_10', 'nsr_spatial_resolution_10', 'nsr_spatial_depth0_time_mean_resolution_10', 'nsr_spatial_depth40_time_mean_resolution_10',
    parser.add_argument("--method", default="pearson", choices=["pearson", "spearman", "kendall"], help="Correlation method")

    parser.add_argument("--top-k", type=int, default=10, help="Top-k size for topk task")
    parser.add_argument("--group-by", default="model", choices=["iteration", "model"], help="Grouping key for topk counts")

    parser.add_argument("--metric_x", default="RMSE", help="X metric for plot task")
    parser.add_argument("--metric_y", default="tl_grid_ssim_freq_1000", help="Y metric for plot task")
    parser.add_argument("--annotate", action="store_true", help="Annotate points on plot")
    parser.add_argument("--corr-annotate", action="store_true", help="Annotate correlation heatmap values")

    parser.add_argument("--save-csv", default="/Odyssey/private/o23gauvr/code/FASCINATION/results", help="Directory to save CSV outputs")
    parser.add_argument("--save-plot", default="/Odyssey/private/o23gauvr/code/FASCINATION/results", help="Directory to save plot outputs")

    args = parser.parse_args()

    model_metrics = load_model_metrics_pickle(args.pickle_path)
    metrics = _parse_metric_list(args.metrics)
    save_csv_dir = _ensure_output_dir(args.save_csv) if args.save_csv else None
    save_plot_dir = _ensure_output_dir(args.save_plot) if args.save_plot else None

    if args.task in {"corr", "all"}:
        corr_df = compute_metrics_correlation_matrix(
            model_metrics=model_metrics,
            metrics=metrics,
            season=args.season,
            split=args.split,
            method=args.method,
        )
        corr_fig, _ = plot_correlation_matrix(corr_df, annotate=args.corr_annotate)
        print(corr_df)
        if save_csv_dir is not None:
            output_file = save_csv_dir / "results_corr.csv"
            corr_df.to_csv(output_file, index=True)
        if save_plot_dir is not None:
            corr_fig.savefig(save_plot_dir / "results_corr_heatmap.png", dpi=150, bbox_inches="tight")
        elif args.task == "corr":
            plt.show()
        plt.close(corr_fig)

    if args.task in {"topk", "all"}:
        counts_df, details_df = count_topk_metric_appearances(
            model_metrics=model_metrics,
            metrics=metrics,
            season=args.season,
            split=args.split,
            top_k=args.top_k,
            group_by=args.group_by,
        )
        print("\nTop-k appearance counts:\n")
        print(counts_df)
        print("\nTop-k details:\n")
        print(details_df)
        if save_csv_dir is not None:
            counts_df.to_csv(save_csv_dir / "results_counts.csv", index=False)
            details_df.to_csv(save_csv_dir / "results_details.csv", index=False)

    if args.task in {"plot", "all"}:
        fig, ax, plot_df = plot_models_by_two_metrics(
            model_metrics=model_metrics,
            metric_x=args.metric_x,
            metric_y=args.metric_y,
            season=args.season,
            split=args.split,
            annotate=args.annotate,
        )
        print(plot_df)
        if save_plot_dir is not None:
            fig.savefig(
                save_plot_dir / f"results_scatter_{args.metric_x.lower()}_vs_{args.metric_y.lower()}.png",
                dpi=150,
                bbox_inches="tight",
            )
        elif args.task == "plot":
            plt.show()
        plt.close(fig)


if __name__ == "__main__":



    main()
