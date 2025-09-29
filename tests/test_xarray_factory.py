from __future__ import annotations

import json

import numpy as np
import pytest
import xarray as xr
from numpy.testing import assert_allclose

from cocofeats.definitions import Artifact, NodeResult
from cocofeats.nodes import get_node
from cocofeats.nodes.factories import apply_1d

try:  # pragma: no cover - optional dependency during tests
    import mne
except Exception:  # pragma: no cover - surface import-time optionality
    mne = None  # type: ignore[assignment]


def _make_dataarray() -> xr.DataArray:
    data = np.arange(12, dtype=float).reshape(3, 4)
    return xr.DataArray(
        data,
        dims=("channels", "time"),
        coords={
            "channels": ["C3", "Cz", "C4"],
            "time": np.linspace(0.0, 0.3, 4),
        },
    )


def test_apply_1d_scalar_output() -> None:
    arr = _make_dataarray()
    result = apply_1d(arr, dim="time", pure_function=np.mean)
    out = result.artifacts[".nc"].item

    assert out.dims == ("channels",)
    assert_allclose(out.values, arr.mean(dim="time").values)


def test_apply_1d_sequence_output_with_coords() -> None:
    arr = _make_dataarray()

    def stats(vector: np.ndarray) -> tuple[float, float]:
        return float(vector.mean()), float(vector.std())

    result = apply_1d(
        arr,
        dim="time",
        pure_function=stats,
        result_dim="stat",
        result_coords=("mean", "std"),
    )

    out = result.artifacts[".nc"].item

    assert out.dims == ("channels", "stat")
    assert list(out.coords["stat"].values) == ["mean", "std"]
    assert_allclose(out.sel(stat="mean").values, arr.mean(dim="time").values)


def test_iterative_mode_matches_vectorized_and_reports_timings() -> None:
    arr = _make_dataarray()

    def stats(vector: np.ndarray) -> np.ndarray:
        return np.array([vector.sum(), vector.mean()])

    vector_result = apply_1d(
        arr,
        dim="time",
        pure_function=stats,
        result_dim="stat",
        result_coords=("sum", "mean"),
        mode="vectorized",
    )
    iterative_result = apply_1d(
        arr,
        dim="time",
        pure_function=stats,
        result_dim="stat",
        result_coords=("sum", "mean"),
        mode="iterative",
    )

    iterative_da = iterative_result.artifacts[".nc"].item
    vector_da = vector_result.artifacts[".nc"].item

    assert_allclose(iterative_da.values, vector_da.values)

    metadata = json.loads(iterative_da.attrs["metadata"])
    assert metadata["mode"] == "iterative"
    assert metadata["per_slice_duration_unit"] == "seconds"

    timing_da = xr.DataArray.from_dict(metadata["per_slice_duration"])
    assert timing_da.dims == iterative_da.dims
    assert timing_da.shape == iterative_da.shape
    assert np.all(timing_da.values >= 0.0)


def test_apply_1d_accepts_noderesult_input() -> None:
    arr = _make_dataarray()
    node_input = NodeResult({".nc": Artifact(item=arr, writer=lambda path: arr.to_netcdf(path))})

    result = apply_1d(node_input, dim="time", pure_function=np.mean)
    out = result.artifacts[".nc"].item

    assert out.shape == (3,)
    assert_allclose(out.values, arr.mean(dim="time").values)


@pytest.mark.skipif(mne is None, reason="mne not available")
def test_apply_1d_handles_mne_raw() -> None:
    pytest.importorskip("scipy", reason="SciPy required for MNE test")

    sfreq = 100.0
    data = np.vstack([
        np.linspace(0.0, 1.0, 200),
        np.linspace(1.0, 2.0, 200),
    ])
    info = mne.create_info(ch_names=["C3", "C4"], sfreq=sfreq, ch_types="eeg")
    raw = mne.io.RawArray(data, info, verbose="error")

    result = apply_1d(raw, dim="time", pure_function=np.mean)
    out = result.artifacts[".nc"].item

    assert out.dims == ("channels",)
    expected = raw.get_data().mean(axis=1)
    assert_allclose(out.values, expected)


def test_xarray_factory_registered_node_supports_dotted_path() -> None:
    node = get_node("xarray_factory")
    arr = _make_dataarray()

    result = node(arr, dim="time", pure_function="numpy.mean")
    out = result.artifacts[".nc"].item

    assert out.dims == ("channels",)
    assert_allclose(out.values, arr.mean(dim="time").values)


def test_apply_1d_raises_on_missing_dimension() -> None:
    arr = _make_dataarray()

    with pytest.raises(ValueError):
        apply_1d(arr, dim="frequency", pure_function=np.mean)


def test_apply_1d_invalid_mode() -> None:
    arr = _make_dataarray()

    with pytest.raises(ValueError):
        apply_1d(arr, dim="time", pure_function=np.mean, mode="invalid")


def test_apply_1d_raises_on_bad_result_coords_length() -> None:
    arr = _make_dataarray()

    def stats(vector: np.ndarray) -> tuple[float, float]:
        return float(vector.mean()), float(vector.std())

    with pytest.raises(ValueError):
        apply_1d(
            arr,
            dim="time",
            pure_function=stats,
            result_dim="stat",
            result_coords=("mean",),
        )

if __name__ == "__main__":
#    pytest.main([__file__])
    pytest.main(["-v", "-s", "-q", "--no-cov", "--pdb", __file__])
