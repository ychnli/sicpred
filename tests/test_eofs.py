"""Tests for the xeofs wrapper."""

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from src import config_cesm
from src.experiment_configs import ExperimentConfig, ensemble_member_split
from src.utils import eofs


def _config():
    return ExperimentConfig(
        experiment_name="synthetic",
        notes="",
        data_name="synthetic",
        data_split=ensemble_member_split(
            "synthetic",
            train=["member1"],
            val=["member2"],
            test=["member3"],
            time_range=pd.date_range("2000-01", "2000-04", freq="MS"),
        ),
        input_config={"var1": {"include": True, "auxiliary": False}},
        target_config={"predict_anom": True, "predict_classes": False},
    )


@pytest.fixture
def normalized_inputs(monkeypatch, tmp_path):
    normalized_dir = tmp_path / "normalized_inputs" / "synthetic"
    normalized_dir.mkdir(parents=True)
    members = ["member1", "member2", "member3"]
    times = pd.date_range("2000-01", "2000-04", freq="MS")
    rng = np.random.default_rng(4)
    for variable in ("var1", "var2"):
        xr.DataArray(
            rng.normal(size=(3, 4, 2, 2)),
            dims=("member_id", "time", "y", "x"),
            coords={
                "member_id": members,
                "time": times,
                "y": [-1.0, 1.0],
                "x": [-1.0, 1.0],
            },
            name=variable,
        ).to_dataset().to_netcdf(normalized_dir / f"{variable}_norm.nc")

    grid = xr.Dataset(
        {
            "lat": (("y", "x"), [[-80.0, -70.0], [-60.0, -50.0]]),
            "lon": (("y", "x"), [[-45.0, 45.0], [-135.0, 135.0]]),
        },
        coords={"y": [-1.0, 1.0], "x": [-1.0, 1.0]},
    )
    monkeypatch.setattr(config_cesm, "PROCESSED_DATA_DIRECTORY", str(tmp_path))
    monkeypatch.setattr(eofs, "generate_sps_grid", lambda grid_size: grid)
    return _config()


def test_compute_univariate_eofs_and_pcs(normalized_inputs):
    model = eofs.compute_eofs(
        normalized_inputs,
        ["var1"],
        member_ids=["member1", "member2"],
        n_modes=2,
    )

    components = model.components()
    scores = model.scores()

    assert components.dims == ("mode", "y", "x")
    assert set(scores.dims) == {"mode", "time", "member_id"}
    assert list(scores.member_id.values) == ["member1", "member2"]
    np.testing.assert_allclose(components.lat, [[-80, -70], [-60, -50]])


def test_compute_multivariate_eofs_uses_all_members(normalized_inputs):
    model = eofs.compute_eofs(
        normalized_inputs, ["var1", "var2"], n_modes=2
    )

    components = model.components()

    assert isinstance(components, list)
    assert len(components) == 2
    assert model.scores().sizes["member_id"] == 3


def test_compute_eofs_rejects_unavailable_member(normalized_inputs):
    with pytest.raises(ValueError, match="not available"):
        eofs.compute_eofs(
            normalized_inputs, ["var1"], member_ids=["missing"], n_modes=1
        )
