"""Compute hierarchical-bootstrap confidence intervals for differences in two models' ACC or RMSE diagnostics.

Command-line usage:
    --metric {acc,rmse}       Required metric to compare.
    --config_a NAME           Required first prediction-directory/config name.
    --config_b NAME           Required second prediction-directory/config name.
    --transform {fisher_z,none}
                              Optional metric transform before bootstrapping (default: none).
    --n_bootstrap INTEGER     Number of bootstrap resamples (default: 5000).
    --alpha FLOAT             Two-sided significance level for confidence intervals (default: 0.05).
    --random_seed INTEGER     Optional seed for reproducible resampling.
    --overwrite               Replace an existing confidence-interval output.

Examples:
    python -m src.utils.bootstrap --metric acc --config_a exp1_input2 --config_b exp1_input3a
    python -m src.utils.bootstrap --metric rmse --config_a exp1_input3a --config_b exp1_input3e --n_bootstrap 10000 --random_seed 42 --overwrite
"""

import argparse
import os

import numpy as np
import xarray as xr

from src import config_cesm
from src.utils.util_shared import write_nc_file


def roll_metric(metric):
    """Align initialization-month metrics to valid forecast month."""
    return xr.concat(
        [
            metric.roll(
                month=int(lead - 1), roll_coords=False
            ).sel(lead_time=lead)
            for lead in metric.lead_time.values
        ],
        dim="lead_time",
    )


def fisher_z(r):
    """Apply a clipped Fisher z transform to correlation coefficients."""
    r = xr.where(np.abs(r) >= 0.999999, np.sign(r) * 0.999999, r)
    return 0.5 * np.log((1.0 + r) / (1.0 - r))


def benjamini_hochberg(p_values):
    """Return Benjamini-Hochberg adjusted p-values, preserving NaNs."""
    p_values = np.asarray(p_values, dtype=float)
    adjusted = np.full(p_values.shape, np.nan, dtype=float)
    finite = np.isfinite(p_values)
    if not finite.any():
        return adjusted
    if np.any((p_values[finite] < 0) | (p_values[finite] > 1)):
        raise ValueError("p-values must be between zero and one")

    finite_values = p_values[finite]
    order = np.argsort(finite_values)
    ranked = finite_values[order]
    ranks = np.arange(1, len(ranked) + 1)
    ranked_adjusted = ranked * len(ranked) / ranks
    ranked_adjusted = np.minimum.accumulate(ranked_adjusted[::-1])[::-1]
    ranked_adjusted = np.minimum(ranked_adjusted, 1.0)
    inverse_order = np.empty_like(order)
    inverse_order[order] = np.arange(len(order))
    adjusted[finite] = ranked_adjusted[inverse_order]
    return adjusted


def _year_seed_sums_and_counts(values, cell_years, years):
    """Aggregate one metric to year-by-training-seed sufficient statistics."""
    n_seeds = values.shape[-1]
    sums = np.zeros((len(years), n_seeds), dtype=float)
    counts = np.zeros((len(years), n_seeds), dtype=np.int64)
    for year_index, year in enumerate(years):
        block = values[cell_years == year].reshape(-1, n_seeds)
        valid = ~np.isnan(block)
        sums[year_index] = np.where(valid, block, 0).sum(axis=0)
        counts[year_index] = valid.sum(axis=0)
    return sums, counts


def _bootstrap_crossed_mean(
    block_sums,
    block_counts,
    sampled_years,
    sampled_seeds,
):
    """Mean over independently sampled years and training seeds."""
    selected_sums = block_sums[sampled_years]
    selected_counts = block_counts[sampled_years]
    seed_indices = np.broadcast_to(
        sampled_seeds[:, None, :],
        (
            sampled_seeds.shape[0],
            sampled_years.shape[1],
            sampled_seeds.shape[1],
        ),
    )
    total = np.take_along_axis(
        selected_sums, seed_indices, axis=2
    ).sum((1, 2))
    count = np.take_along_axis(
        selected_counts, seed_indices, axis=2
    ).sum((1, 2))
    return np.divide(
        total,
        count,
        out=np.full(total.shape, np.nan, dtype=float),
        where=count > 0,
    )


def bootstrap_metric_significance(
    metric_a: xr.DataArray,
    metric_b: xr.DataArray,
    n_bootstrap: int = 5000,
    alpha: float = 0.05,
    random_seed: int | None = None,
):
    """Bootstrap paired evaluation years and independent training seeds.

    Verification cases remain paired between the two configurations. Training
    seeds are sampled independently because equal nn_member_id labels do not
    identify paired realizations of training uncertainty.
    """
    if n_bootstrap <= 0:
        raise ValueError("n_bootstrap must be positive")
    if not 0 < alpha < 1:
        raise ValueError("alpha must be between zero and one")

    required_dims = {"start_prediction_month", "lead_time"}
    for name, metric in (("metric_a", metric_a), ("metric_b", metric_b)):
        missing = required_dims - set(metric.dims)
        if missing:
            raise ValueError(
                f"{name} is missing required dimensions: {sorted(missing)}"
            )

    seed_dim = "nn_member_id"
    seed_dim_a = f"{seed_dim}_a"
    seed_dim_b = f"{seed_dim}_b"
    if seed_dim in metric_a.dims:
        metric_a = metric_a.rename({seed_dim: seed_dim_a})
    else:
        metric_a = metric_a.expand_dims({seed_dim_a: [0]})
    if seed_dim in metric_b.dims:
        metric_b = metric_b.rename({seed_dim: seed_dim_b})
    else:
        metric_b = metric_b.expand_dims({seed_dim_b: [0]})

    non_seed_dims_a = set(metric_a.dims) - {seed_dim_a}
    non_seed_dims_b = set(metric_b.dims) - {seed_dim_b}
    if non_seed_dims_a != non_seed_dims_b:
        raise ValueError(
            "Metrics must have identical non-seed dimensions; got "
            f"{sorted(non_seed_dims_a)} and {sorted(non_seed_dims_b)}"
        )
    metric_a, metric_b = xr.align(
        metric_a, metric_b, join="exact", copy=False
    )

    metric_a = metric_a.assign_coords(
        year=(
            "start_prediction_month",
            metric_a["start_prediction_month"].dt.year.data,
        ),
        month=(
            "start_prediction_month",
            metric_a["start_prediction_month"].dt.month.data,
        ),
    )
    sample_dims = sorted(
        non_seed_dims_a - {"start_prediction_month", "lead_time"}
    )
    metric_a = metric_a.transpose(
        "start_prediction_month", "lead_time", *sample_dims, seed_dim_a
    )
    metric_b = metric_b.transpose(
        "start_prediction_month", "lead_time", *sample_dims, seed_dim_b
    )
    values_a = np.asarray(metric_a.values, dtype=float)
    values_b = np.asarray(metric_b.values, dtype=float)
    year_values = np.asarray(metric_a["year"].values)
    month_values = np.asarray(metric_a["month"].values)
    months = np.arange(1, 13, dtype=int)
    leads = metric_a["lead_time"].values

    out = {
        name: np.full((len(months), len(leads)), np.nan)
        for name in ("delta", "p_value", "ci_low", "ci_high")
    }
    rng = np.random.default_rng(random_seed)

    for month_index, month in enumerate(months):
        month_positions = np.flatnonzero(month_values == month)
        if len(month_positions) == 0:
            continue
        cell_years = year_values[month_positions]
        years = np.unique(cell_years)

        for lead_index in range(len(leads)):
            cell_values_a = values_a[month_positions, lead_index]
            cell_values_b = values_b[month_positions, lead_index]
            sums_a, counts_a = _year_seed_sums_and_counts(
                cell_values_a, cell_years, years
            )
            sums_b, counts_b = _year_seed_sums_and_counts(
                cell_values_b, cell_years, years
            )
            valid_years = (counts_a.sum(axis=1) > 0) & (
                counts_b.sum(axis=1) > 0
            )
            sums_a, counts_a = sums_a[valid_years], counts_a[valid_years]
            sums_b, counts_b = sums_b[valid_years], counts_b[valid_years]
            if not len(sums_a):
                continue

            sampled_years = rng.integers(
                len(sums_a), size=(n_bootstrap, len(sums_a))
            )
            sampled_seeds_a = rng.integers(
                sums_a.shape[1], size=(n_bootstrap, sums_a.shape[1])
            )
            sampled_seeds_b = rng.integers(
                sums_b.shape[1], size=(n_bootstrap, sums_b.shape[1])
            )
            boot_means = _bootstrap_crossed_mean(
                sums_a, counts_a, sampled_years, sampled_seeds_a
            ) - _bootstrap_crossed_mean(
                sums_b, counts_b, sampled_years, sampled_seeds_b
            )
            boot_means = boot_means[np.isfinite(boot_means)]
            if not len(boot_means):
                continue
            boot_means.sort()

            mean_a = sums_a.sum() / counts_a.sum()
            mean_b = sums_b.sum() / counts_b.sum()
            out["delta"][month_index, lead_index] = mean_a - mean_b
            opposite_tail_count = min(
                np.count_nonzero(boot_means <= 0),
                np.count_nonzero(boot_means >= 0),
            )
            out["p_value"][month_index, lead_index] = min(
                1.0,
                2 * (opposite_tail_count + 1) / (len(boot_means) + 1),
            )
            out["ci_low"][month_index, lead_index] = np.quantile(
                boot_means, alpha / 2
            )
            out["ci_high"][month_index, lead_index] = np.quantile(
                boot_means, 1 - alpha / 2
            )

    result = xr.Dataset(
        {
            name: xr.DataArray(
                data,
                coords={"month": months, "lead_time": leads},
                dims=("month", "lead_time"),
            )
            for name, data in out.items()
        }
    )
    result["q_value"] = xr.DataArray(
        benjamini_hochberg(result["p_value"].values),
        coords=result["p_value"].coords,
        dims=result["p_value"].dims,
    )
    result["q_value"].attrs["fdr_scope"] = (
        "Benjamini-Hochberg across all finite month-by-lead tests in this "
        "configuration comparison"
    )
    return result


def main():
    parser = argparse.ArgumentParser(description="Compute bootstrap CIs for metric differences")
    parser.add_argument("--metric", type=str, choices=["acc", "rmse"], required=True,
                        help="Metric to analyze")
    parser.add_argument("--config_a", type=str, required=True)
    parser.add_argument("--config_b", type=str, required=True)
    parser.add_argument("--transform", type=str, choices=["fisher_z", "none"], default="none",)

    parser.add_argument("--n_bootstrap", type=int, default=5000,
                        help="Number of bootstrap samples to draw")
    parser.add_argument("--alpha", type=float, default=0.05,
                        help="Significance level for confidence intervals")
    parser.add_argument("--random_seed", type=int, default=None,
                        help="Random seed for reproducibility")
    parser.add_argument("--overwrite", action="store_true")

    args = parser.parse_args()

    if args.metric != "acc" and args.transform == "fisher_z":
        parser.error("fisher_z is only defined for ACC")

    output_dir = os.path.join(config_cesm.ANALYSIS_RESULTS_DIRECTORY, "confidence_intervals")
    output_fp = os.path.join(output_dir, f"{args.config_a}_{args.config_b}_{args.metric}.nc")
    os.makedirs(output_dir, exist_ok=True)

    if os.path.exists(output_fp) and not args.overwrite:
        print(f"Output file {output_fp} already exists. Use --overwrite to overwrite.")
        return

    print(f"Computing CIs for {args.metric} between {args.config_a} and {args.config_b}...")
    ds_a = xr.load_dataset(os.path.join(config_cesm.PREDICTIONS_DIRECTORY, args.config_a, "diagnostics", f"{args.metric}.nc"))
    ds_b = xr.load_dataset(os.path.join(config_cesm.PREDICTIONS_DIRECTORY, args.config_b, "diagnostics", f"{args.metric}.nc"))

    # the -1 nn_member_id is used to denote the ensemble mean prediction
    # if the predictions has this member id, drop it. We only want to bootstrap
    # over individual ensemble members, not the ensemble mean
    if "nn_member_id" in ds_a.dims:
        ds_a = ds_a.where(ds_a["nn_member_id"] != -1, drop=True)
    if "nn_member_id" in ds_b.dims:
        ds_b = ds_b.where(ds_b["nn_member_id"] != -1, drop=True)


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
    result_ds.attrs.update(
        metric=args.metric,
        transform=args.transform,
        training_seed_resampling="independent",
        evaluation_year_resampling="paired",
    )

    result_ds = roll_metric(result_ds)

    write_nc_file(result_ds, output_fp, overwrite=args.overwrite)
    print(f"done! Bootstrap results saved to {output_fp}\n")


if __name__ == '__main__':
    main()
