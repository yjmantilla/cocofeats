"""Tests for xarray operation nodes."""

from __future__ import annotations

import numpy as np
import pytest
import xarray as xr
from structlog.testing import capture_logs

from neurodags.definitions import Artifact, NodeResult
from neurodags.nodes.operations import (
    aggregate_across_dimension,
    binarize_with_median,
    extract_data_var,
    mean_across_dimension,
    slice_xarray,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def simple_da():
    return xr.DataArray(
        np.arange(24, dtype=float).reshape(4, 3, 2),
        dims=("times", "channel", "frequency"),
        coords={
            "times": np.arange(4, dtype=float),
            "channel": ["Cz", "Pz", "Fz"],
            "frequency": [10.0, 20.0],
        },
    )


@pytest.fixture
def simple_ds(simple_da):
    return xr.Dataset({"power": simple_da})


# ---------------------------------------------------------------------------
# binarize_with_median
# ---------------------------------------------------------------------------


def test_binarize_returns_noderesulet(simple_da):
    result = binarize_with_median(simple_da, dim="times")
    assert isinstance(result, NodeResult)
    assert ".nc" in result.artifacts


def test_binarize_values_binary(simple_da):
    result = binarize_with_median(simple_da, dim="times")
    arr = result.artifacts[".nc"].item
    unique = set(arr.values.flatten().tolist())
    assert unique <= {0, 1}


def test_binarize_from_path(simple_da, tmp_path):
    nc_path = tmp_path / "data.nc"
    simple_da.to_netcdf(nc_path)
    result = binarize_with_median(nc_path, dim="times")
    assert isinstance(result, NodeResult)


def test_binarize_invalid_type_raises():
    with pytest.raises(ValueError, match="xarray DataArray"):
        binarize_with_median(42, dim="times")


def test_binarize_invalid_path_raises(tmp_path):
    with pytest.raises(ValueError, match="Failed to load"):
        binarize_with_median(tmp_path / "nonexistent.nc", dim="times")


# ---------------------------------------------------------------------------
# mean_across_dimension
# ---------------------------------------------------------------------------


def test_mean_returns_noderesulet(simple_da):
    result = mean_across_dimension(simple_da, dim="times")
    assert isinstance(result, NodeResult)
    assert ".nc" in result.artifacts


def test_mean_reduces_dimension(simple_da):
    result = mean_across_dimension(simple_da, dim="times")
    arr = result.artifacts[".nc"].item
    assert "times" not in arr.dims


def test_mean_from_path(simple_da, tmp_path):
    nc_path = tmp_path / "data.nc"
    simple_da.to_netcdf(nc_path)
    result = mean_across_dimension(nc_path, dim="times")
    assert isinstance(result, NodeResult)


def test_mean_invalid_type_raises():
    with pytest.raises(ValueError, match="xarray DataArray"):
        mean_across_dimension({"not": "xarray"}, dim="times")


# ---------------------------------------------------------------------------
# extract_data_var
# ---------------------------------------------------------------------------


def test_extract_from_dataset(simple_ds):
    result = extract_data_var(simple_ds, data_var="power")
    assert isinstance(result, NodeResult)
    arr = result.artifacts[".nc"].item
    assert isinstance(arr, xr.DataArray)


def test_extract_from_dataarray(simple_da):
    da = simple_da.copy()
    da.name = "power"
    result = extract_data_var(da, data_var="power")
    assert isinstance(result, NodeResult)


def test_extract_from_dataarray_no_name(simple_da):
    da = simple_da.copy()
    da.name = None
    result = extract_data_var(da, data_var="anything")
    arr = result.artifacts[".nc"].item
    assert arr.name == "anything"


def test_extract_from_noderesulet(simple_ds):
    nr = NodeResult(artifacts={".nc": Artifact(item=simple_ds, writer=lambda p: None)})
    result = extract_data_var(nr, data_var="power")
    assert isinstance(result, NodeResult)


def test_extract_noderesulet_missing_nc_raises(simple_da):
    nr = NodeResult(artifacts={".fif": Artifact(item=simple_da, writer=lambda p: None)})
    with pytest.raises(ValueError, match=r"\.nc"):
        extract_data_var(nr, data_var="power")


def test_extract_missing_var_raises(simple_ds):
    with pytest.raises(KeyError, match="nonexistent"):
        extract_data_var(simple_ds, data_var="nonexistent")


def test_extract_dataarray_wrong_name_raises(simple_da):
    da = simple_da.copy()
    da.name = "other"
    with pytest.raises(ValueError, match="does not match requested variable name"):
        extract_data_var(da, data_var="power")


def test_extract_from_path(simple_ds, tmp_path):
    nc_path = tmp_path / "ds.nc"
    simple_ds.to_netcdf(nc_path)
    result = extract_data_var(nc_path, data_var="power")
    assert isinstance(result, NodeResult)


def test_extract_from_path_missing_var_raises(simple_ds, tmp_path):
    nc_path = tmp_path / "ds.nc"
    simple_ds.to_netcdf(nc_path)
    with pytest.raises(KeyError, match="nonexistent"):
        extract_data_var(nc_path, data_var="nonexistent")


def test_extract_invalid_type_raises():
    with pytest.raises(ValueError, match="must be a NodeResult"):
        extract_data_var(42, data_var="power")


# ---------------------------------------------------------------------------
# slice_xarray
# ---------------------------------------------------------------------------


def test_slice_by_index(simple_da):
    result = slice_xarray(simple_da, dim="times", start=1, end=3)
    arr = result.artifacts[".nc"].item
    assert arr.sizes["times"] == 2


def test_slice_by_coord(simple_da):
    result = slice_xarray(simple_da, dim="times", start=1.0, end=2.0)
    arr = result.artifacts[".nc"].item
    assert "times" in arr.dims or arr.ndim < simple_da.ndim


def test_slice_full_range(simple_da):
    result = slice_xarray(simple_da, dim="times")
    arr = result.artifacts[".nc"].item
    assert arr.sizes["times"] == 4


def test_slice_from_path(simple_da, tmp_path):
    nc_path = tmp_path / "data.nc"
    simple_da.to_netcdf(nc_path)
    result = slice_xarray(nc_path, dim="times", start=0, end=2)
    assert isinstance(result, NodeResult)


def test_slice_from_noderesulet(simple_da):
    nr = NodeResult(artifacts={".nc": Artifact(item=simple_da, writer=lambda p: None)})
    result = slice_xarray(nr, dim="times", start=0, end=2)
    assert isinstance(result, NodeResult)


def test_slice_noderesulet_missing_nc_raises(simple_da):
    nr = NodeResult(artifacts={".fif": Artifact(item=simple_da, writer=lambda p: None)})
    with pytest.raises(ValueError, match=r"\.nc"):
        slice_xarray(nr, dim="times", start=0, end=2)


def test_slice_invalid_dim_raises(simple_da):
    with pytest.raises(ValueError, match="Dimension"):
        slice_xarray(simple_da, dim="nonexistent", start=0, end=2)


def test_slice_invalid_type_raises():
    with pytest.raises(ValueError, match="xarray DataArray"):
        slice_xarray(42, dim="times")


def test_slice_single_index_squeezes(simple_da):
    result = slice_xarray(simple_da, dim="times", start=1, end=2)
    arr = result.artifacts[".nc"].item
    assert "times" not in arr.dims


# ---------------------------------------------------------------------------
# aggregate_across_dimension
# ---------------------------------------------------------------------------


def test_aggregate_mean(simple_da):
    result = aggregate_across_dimension(simple_da, dim="times", operation="mean")
    arr = result.artifacts[".nc"].item
    assert "times" not in arr.dims


def test_aggregate_sum(simple_da):
    result = aggregate_across_dimension(simple_da, dim="times", operation="sum")
    arr = result.artifacts[".nc"].item
    assert float(arr.values.sum()) > 0


def test_aggregate_max(simple_da):
    result = aggregate_across_dimension(simple_da, dim="times", operation="max")
    arr = result.artifacts[".nc"].item
    assert "times" not in arr.dims


def test_aggregate_from_path(simple_da, tmp_path):
    nc_path = tmp_path / "data.nc"
    simple_da.to_netcdf(nc_path)
    result = aggregate_across_dimension(nc_path, dim="times", operation="mean")
    assert isinstance(result, NodeResult)


def test_aggregate_from_noderesulet(simple_da):
    nr = NodeResult(artifacts={".nc": Artifact(item=simple_da, writer=lambda p: None)})
    result = aggregate_across_dimension(nr, dim="times", operation="mean")
    assert isinstance(result, NodeResult)


def test_aggregate_noderesulet_missing_nc_raises(simple_da):
    nr = NodeResult(artifacts={".fif": Artifact(item=simple_da, writer=lambda p: None)})
    with pytest.raises(ValueError, match=r"\.nc"):
        aggregate_across_dimension(nr, dim="times", operation="mean")


def test_aggregate_invalid_operation_raises(simple_da):
    with pytest.raises(ValueError, match="not valid"):
        aggregate_across_dimension(simple_da, dim="times", operation="nonexistent_op")


def test_aggregate_invalid_type_raises():
    with pytest.raises(ValueError, match="xarray DataArray"):
        aggregate_across_dimension(42, dim="times", operation="mean")


def test_aggregate_with_args(simple_da):
    result = aggregate_across_dimension(
        simple_da, dim="times", operation="mean", args={"keepdims": False}
    )
    arr = result.artifacts[".nc"].item
    assert isinstance(arr, xr.DataArray)


# --- non-finite / skipna visibility (issue #19) ----------------------------


@pytest.fixture
def da_with_nan():
    # NaN at (times=1, channel=a) and (times=2, channel=b) -> 1 dropped per channel
    return xr.DataArray(
        np.array([[1.0, 2.0], [np.nan, 4.0], [5.0, np.nan], [7.0, 8.0]]),
        dims=("times", "channel"),
        coords={"times": [0, 1, 2, 3], "channel": ["a", "b"]},
    )


def test_aggregate_raise_on_dropped_nan(da_with_nan):
    with pytest.raises(ValueError, match="non-finite"):
        aggregate_across_dimension(da_with_nan, dim="times", operation="mean", on_dropped="raise")


def test_aggregate_warn_on_dropped_nan(da_with_nan):
    with capture_logs() as logs:
        aggregate_across_dimension(da_with_nan, dim="times", operation="mean")  # default warn
    warnings = [
        e for e in logs if e.get("log_level") == "warning" and "dropped non-finite" in e["event"]
    ]
    assert len(warnings) == 1
    assert warnings[0]["n_dropped"] == 2


def test_aggregate_ignore_on_dropped_nan(da_with_nan):
    with capture_logs() as logs:
        result = aggregate_across_dimension(
            da_with_nan, dim="times", operation="mean", on_dropped="ignore"
        )
    assert not [e for e in logs if "dropped non-finite" in e.get("event", "")]
    arr = result.artifacts[".nc"].item
    # skipna mean is still applied: channel a -> (1+5+7)/3
    assert float(arr.sel(channel="a")) == pytest.approx(13.0 / 3.0)


def test_aggregate_emit_counts(da_with_nan):
    result = aggregate_across_dimension(
        da_with_nan, dim="times", operation="mean", emit_counts=True
    )
    arr = result.artifacts[".nc"].item
    assert "n_used" in arr.coords
    assert "n_dropped" in arr.coords
    assert arr["n_used"].sel(channel="a").item() == 3
    assert arr["n_dropped"].sel(channel="a").item() == 1
    assert arr["n_dropped"].sel(channel="b").item() == 1


def test_aggregate_no_nan_is_unchanged(simple_da):
    with capture_logs() as logs:
        result = aggregate_across_dimension(simple_da, dim="times", operation="mean")
    assert not [e for e in logs if "dropped non-finite" in e.get("event", "")]
    arr = result.artifacts[".nc"].item
    assert "n_used" not in arr.coords


def test_aggregate_skipna_false_no_drop_policy(da_with_nan):
    # skipna=False -> NaN is not dropped (it propagates), so on_dropped must not fire
    result = aggregate_across_dimension(
        da_with_nan,
        dim="times",
        operation="mean",
        on_dropped="raise",
        args={"skipna": False},
    )
    arr = result.artifacts[".nc"].item
    assert bool(np.isnan(arr.sel(channel="a")))


def test_aggregate_invalid_on_dropped_raises(simple_da):
    with pytest.raises(ValueError, match="on_dropped must be one of"):
        aggregate_across_dimension(simple_da, dim="times", operation="mean", on_dropped="bogus")
