import numpy as np
import pytest
import xarray as xr

from src.download import download_cesm_data as download


def make_ocean_grid():
    return xr.Dataset(
        {
            "TLAT": (("nlat", "nlon"), [[-50.0, -50.0], [-40.0, -40.0], [-20.0, -20.0]]),
            "TLONG": (("nlat", "nlon"), [[0.0, 180.0], [0.0, 180.0], [0.0, 180.0]]),
            "z_t": ("z_t", [500.0, 1500.0, 2500.0]),
            "z_w_top": ("z_w_top", [0.0, 1000.0, 2000.0]),
            "z_w_bot": ("z_w_bot", [1000.0, 2000.0, 3000.0]),
            "dz": ("z_t", [1000.0, 1000.0, 1000.0]),
        }
    )


def test_direct_processor_selects_nearest_physical_pressure():
    raw = xr.Dataset(
        {"Z3": ("lev", [4000.0, 5247.0, 7000.0])},
        coords={"lev": [400.0, 524.6871747, 700.0]},
    )
    raw["Z3"].attrs["units"] = "m"
    settings = {
        "raw_variables": ("Z3",),
        "vertical_selection": {
            "coordinate": "lev",
            "value": 500.0,
            "method": "nearest",
        },
        "long_name": "Geopotential height near 500 hPa",
    }

    result = download.process_direct(raw, "geopotential", settings)

    assert result.name == "geopotential"
    assert result.item() == pytest.approx(5247.0)
    assert result["lev"].item() == pytest.approx(524.6871747)


def test_depth_weighted_mean_uses_partial_boundary_layer():
    raw = xr.Dataset(
        {"TEMP": ("z_t", [1.0, 3.0, 5.0])},
        coords={"z_t": [500.0, 1500.0, 2500.0]},
    )
    raw["TEMP"].attrs["units"] = "degC"
    settings = {
        "raw_variables": ("TEMP",),
        "depth_coordinate": "z_t",
        "depth_bounds": (0.0, 2500.0),
        "long_name": "Test depth mean",
    }

    result = download.process_depth_weighted_mean(
        raw, "ohc200", settings, make_ocean_grid()
    )

    expected = (1.0 * 1000.0 + 3.0 * 1000.0 + 5.0 * 500.0) / 2500.0
    assert result.name == "ohc200"
    assert result.item() == pytest.approx(expected)
    assert result.attrs["units"] == "degC"


def test_depth_weighted_mean_renormalizes_missing_shallow_cells():
    raw = xr.Dataset(
        {"TEMP": ("z_t", [1.0, np.nan, 5.0])},
        coords={"z_t": [500.0, 1500.0, 2500.0]},
    )
    settings = {
        "raw_variables": ("TEMP",),
        "depth_coordinate": "z_t",
        "depth_bounds": (0.0, 2500.0),
        "long_name": "Test depth mean",
    }

    result = download.process_depth_weighted_mean(
        raw, "ohc200", settings, make_ocean_grid()
    )

    expected = (1.0 * 1000.0 + 5.0 * 500.0) / 1500.0
    assert result.item() == pytest.approx(expected)


def test_ocean_latitude_bounds_select_rows_by_tlat_values():
    raw = xr.Dataset(
        {"TEMP": (("nlat", "nlon"), np.arange(6).reshape(3, 2))}
    )

    result = download._select_latitude(
        raw, "ocn", (-90.0, -30.0), make_ocean_grid()
    )

    assert result.sizes == {"nlat": 2, "nlon": 2}
    np.testing.assert_allclose(result["lat"], [[-50.0, -50.0], [-40.0, -40.0]])
    np.testing.assert_allclose(result["lon"], [[0.0, 180.0], [0.0, 180.0]])


def test_ice_grid_renames_ocean_horizontal_dimensions():
    raw = xr.Dataset({"hi": (("nj", "ni"), np.ones((3, 2)))})

    result = download._select_latitude(
        raw, "ice", (-90.0, -30.0), make_ocean_grid()
    )

    assert result["hi"].dims == ("nj", "ni")
    assert result["lat"].dims == ("nj", "ni")
    assert result.sizes == {"nj": 2, "ni": 2}


def test_expanding_all_members_does_not_mutate_original_settings():
    original = {
        "vars": ["icefrac"],
        "member_id": {"icefrac": "all"},
        "chunk": "default",
    }

    expanded = download._expanded_download_settings(original)

    assert original["member_id"]["icefrac"] == "all"
    assert expanded["member_id"]["icefrac"] == download.config.AVAILABLE_CESM_MEMBERS
    assert expanded["member_id"]["icefrac"] is not download.config.AVAILABLE_CESM_MEMBERS
