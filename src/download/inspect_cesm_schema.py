"""Inspect coordinate schemas for representative CESM2-LE AWS datasets.

This utility opens monthly atmosphere, ocean, and sea-ice datasets through the
same Intake-ESM catalog used by ``download_cesm_data.py``. It reads metadata and
coordinate arrays only; it does not download the climate variable data itself.

Run from the repository root with::

    python -m src.download.inspect_cesm_schema
"""

import argparse

import intake
import numpy as np


DEFAULT_CATALOG_URL = (
    "https://raw.githubusercontent.com/NCAR/cesm2-le-aws/main/"
    "intake-catalogs/aws-cesm2-le.json"
)

SAMPLE_VARIABLES = {
    "atm": "Z3",
    "ocn": "TEMP",
    "ice": "hi",
}


def format_values(values, max_values=12):
    """Return a compact representation of a coordinate array."""
    values = np.asarray(values)
    flattened = values.ravel()
    if flattened.size <= max_values:
        return np.array2string(flattened, separator=", ")

    half = max_values // 2
    sample = np.concatenate((flattened[:half], flattened[-half:]))
    return f"{np.array2string(sample, separator=', ')} (first/last {half})"


def print_array_schema(name, array, load_values=True):
    """Print dimensions, shape, attributes, and optional value summaries."""
    print(f"  {name}:")
    print(f"    dims: {array.dims}")
    print(f"    shape: {array.shape}")
    print(f"    dtype: {array.dtype}")
    print(f"    attrs: {dict(array.attrs)}")

    if array.size == 0 or not load_values:
        return

    values = array.values
    print(f"    values: {format_values(values)}")
    if np.issubdtype(values.dtype, np.number):
        print(f"    range: [{np.nanmin(values)}, {np.nanmax(values)}]")


def open_catalog_dataset(catalog, **search_kwargs):
    """Open the single dataset matched by a constrained catalog search."""
    subset = catalog.search(**search_kwargs)
    print("  catalog rows:")
    print(subset.df.to_string(index=False))
    datasets = subset.to_dataset_dict(storage_options={"anon": True})

    if len(datasets) != 1:
        keys = sorted(datasets)
        raise RuntimeError(
            f"Expected exactly one dataset for {search_kwargs}, found {keys}"
        )

    key, dataset = next(iter(datasets.items()))
    print(f"  dataset key: {key}")
    return dataset


def inspect_variable(catalog, component, variable):
    """Inspect one representative monthly variable for a CESM component."""
    print(f"\n{'=' * 72}")
    print(f"{component.upper()} MONTHLY VARIABLE: {variable}")
    print(f"{'=' * 72}")
    dataset = open_catalog_dataset(
        catalog,
        component=component,
        variable=variable,
        frequency="monthly",
        experiment="historical",
        forcing_variant="cmip6",
    )

    print(f"  dataset sizes: {dict(dataset.sizes)}")
    print(f"  coordinates: {list(dataset.coords)}")
    print(f"  data variables: {list(dataset.data_vars)}")
    print_array_schema(variable, dataset[variable], load_values=False)

    print("  dimension coordinates:")
    for dimension in dataset[variable].dims:
        if dimension in dataset.coords:
            print_array_schema(dimension, dataset.coords[dimension])
        else:
            print(f"  {dimension}: no coordinate variable")


def inspect_static_grid(catalog, component):
    """Inspect the static grid store for an ocean or sea-ice component."""
    print(f"\n{'=' * 72}")
    print(f"{component.upper()} STATIC GRID")
    print(f"{'=' * 72}")
    dataset = open_catalog_dataset(
        catalog,
        component=component,
        frequency="static",
        experiment="historical",
        forcing_variant="cmip6",
    )

    print(f"  dataset sizes: {dict(dataset.sizes)}")
    print(f"  coordinates: {list(dataset.coords)}")
    print(f"  data variables: {list(dataset.data_vars)}")
    relevant_variables = (
        "TLAT", "TLONG", "z_t", "z_w_top", "z_w_bot", "dz"
    )
    for name in relevant_variables:
        print_array_schema(name, dataset[name])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--catalog-url",
        default=DEFAULT_CATALOG_URL,
        help="Intake-ESM catalog URL to inspect.",
    )
    args = parser.parse_args()

    catalog = intake.open_esm_datastore(args.catalog_url)
    for component, variable in SAMPLE_VARIABLES.items():
        inspect_variable(catalog, component, variable)

    # The catalog lists ice/static/grid.zarr, but that S3 key is absent.
    # The ice variable uses the same 384x320 grid as the ocean component.
    inspect_static_grid(catalog, "ocn")


if __name__ == "__main__":
    main()
