"""
Compute bootstrap confidence intervals for change in metric (ACC or RMSE) between
two models.
"""

import numpy as np
import xarray as xr
import os
import argparse
from src.utils.util_shared import write_nc_file

def fisher_z(r):
    r = xr.where(np.abs(r) >= 0.999999, np.sign(r) * 0.999999, r) 
    return 0.5 * np.log((1.0 + r) / (1.0 - r))


def bootstrap_metric_significance(
    metric_a: xr.DataArray,
    metric_b: xr.DataArray,
    n_bootstrap: int = 5000,
    alpha: float = 0.05,
    random_seed: int | None = None,
):
    """
    Compare two sets of metrics that verify against the same set of targets,
    generates a block (year) bootstrap distribution of the difference in metric, 
    and computes confidence intervals.

    Args:
    metric_a, metric_b: xr.DataArray
        Dimensions can include 'member_id' (required if present in either), 'lead_time' 
        (required), 'start_prediction_month' (required), and 'nn_member_id' (optional)
    n_bootstrap: int
        Number of bootstrap samples to draw
    alpha: float
        Significance level for confidence intervals

    Returns:
    xr.Dataset with variables
        delta_z   : mean Fisher‑z difference
        p_value   : two‑sided bootstrap p‑value
        ci_low / ci_high : lower / upper (1‑alpha) CI bounds
    All with dims ('month', 'lead_time').
    """
    rng = np.random.default_rng(random_seed)
    metric_a, metric_b = xr.align(metric_a, metric_b, join="inner", copy=False)

    d = metric_a - metric_b

    def _bootstrap_mean(diff_da):
        """
        Return mean diff, p-value, CI for a (year, sample) DataArray.
        """
        years = np.unique(diff_da["year"].values)
        idx_by_year = {yr: np.where(diff_da["year"].values == yr)[0] for yr in years}

        boot_means = np.empty(n_bootstrap)
        for b in range(n_bootstrap):
            samp_years = rng.choice(years, size=len(years), replace=True)
            mask = np.concatenate([idx_by_year[yr] for yr in samp_years])
            boot_means[b] = diff_da.values[mask].mean()

        boot_means = np.sort(boot_means)
        ci_low = np.quantile(boot_means, alpha / 2)
        ci_hi  = np.quantile(boot_means, 1 - alpha / 2)

        # two-sided p-value: proportion of boot_means whose sign is opposite to mean
        p_val = 2 * min(
            (boot_means <= 0).mean(),
            (boot_means >= 0).mean(),
        )
        return diff_da.mean().item(), p_val, ci_low, ci_hi

    d = d.assign_coords(
        year=("start_prediction_month", d["start_prediction_month"].dt.year.data),
        month=("start_prediction_month", d["start_prediction_month"].dt.month.data),
    )

    months = np.arange(1, 13, dtype=int)
    leads = d["lead_time"].values

    out = {
        "delta_z": np.full((12, 6), np.nan),
        "p_value": np.full((12, 6), np.nan),
        "ci_low": np.full((12, 6), np.nan),
        "ci_high": np.full((12, 6), np.nan)
    }

    for mi, m in enumerate(months):
        dm = d.where(d["month"] == m, drop=True)
        if dm.size == 0:
            continue
        for li, lead in enumerate(leads):
            subset = dm.sel(lead_time=lead)
            subset_flat = subset.stack(sample=subset.dims).dropna("sample")
            delta, p, lo, hi = _bootstrap_mean(subset_flat)
            out["delta_z"][mi, li] = delta
            out["p_value"][mi, li] = p
            out["ci_low"][mi, li] = lo
            out["ci_high"][mi, li] = hi

    ds_out = xr.Dataset(
        {k: xr.DataArray(v, coords={"month": months, "lead_time": leads}, dims=("month", "lead_time"))
         for k, v in out.items()}
    )
    return ds_out


def main():
    parser = argparse.ArgumentParser(description="Compute bootstrap CIs for metric differences")
    parser.add_argument("--metric", type=str, choices=["acc", "rmse"], required=True,
                        help="Metric to analyze")
    parser.add_argument("--model_a_metric_fp", type=str, required=True,
                        help="Filepath to model A metrics (xarray Dataset in NetCDF format)")
    parser.add_argument("--model_b_metric_fp", type=str, required=True,
                        help="Filepath to model B metrics (xarray Dataset in NetCDF format)")
    parser.add_argument("--output_fp", type=str, required=True,
                        help="Filepath to save output Dataset (NetCDF format)")
    parser.add_argument("--transform", type=str, choices=["fisher_z", "none"], default="none",)

    parser.add_argument("--n_bootstrap", type=int, default=5000,
                        help="Number of bootstrap samples to draw")
    parser.add_argument("--alpha", type=float, default=0.05,
                        help="Significance level for confidence intervals")
    parser.add_argument("--random_seed", type=int, default=None,
                        help="Random seed for reproducibility")
    parser.add_argument("--overwrite", action="store_true")

    args = parser.parse_args()

    if os.path.exists(args.output_fp) and not args.overwrite:
        print(f"Output file {args.output_fp} already exists. Use --overwrite to overwrite.")
        return

    ds_a = xr.load_dataset(args.model_a_metric_fp)
    ds_b = xr.load_dataset(args.model_b_metric_fp)

    if args.transform == "fisher_z":
        ds_a[args.metric] = fisher_z(ds_a[args.metric])
        ds_b[args.metric] = fisher_z(ds_b[args.metric])

    result_ds = bootstrap_metric_significance(
        ds_a[args.metric],
        ds_b[args.metric],
        n_bootstrap=args.n_bootstrap,
        alpha=args.alpha,
        random_seed=args.random_seed,
    )

    write_nc_file(result_ds, args.output_fp, overwrite=args.overwrite)
    print(f"Bootstrap results saved to {args.output_fp}")