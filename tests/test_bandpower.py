from __future__ import annotations

import numpy as np
import xarray as xr

from numpy.testing import assert_allclose

from cocofeats.definitions import Artifact, FeatureResult
from cocofeats.features.spectral import bandpower


def _make_psd() -> xr.DataArray:
    freqs = np.arange(0.0, 11.0, 1.0)
    data = np.ones((2, 3, freqs.size))
    return xr.DataArray(
        data,
        dims=("epochs", "spaces", "frequencies"),
        coords={
            "epochs": [0, 1],
            "spaces": ["A", "B", "C"],
            "frequencies": freqs,
        },
    )


def test_bandpower_replaces_frequency_dimension() -> None:
    psd = _make_psd()
    feature = bandpower(psd, bands={"broad": (0.0, 10.0)})
    band_da = feature.artifacts[".nc"].item

    assert isinstance(band_da, xr.DataArray)
    assert band_da.dims == ("epochs", "spaces", "freqbands")
    assert list(band_da.coords["freqbands"].values) == ["broad"]
    assert_allclose(band_da.sel(freqbands="broad").values, 10.0)


def test_bandpower_relative_normalisation() -> None:
    psd = _make_psd()
    bands = {"low": (0.0, 5.0), "high": (5.0, 10.0)}
    feature = bandpower(psd, bands=bands, relative=True)
    band_da = feature.artifacts[".nc"].item

    assert list(band_da.coords["freqbands"].values) == ["low", "high"]
    assert_allclose(band_da.sel(freqbands="low").values, 0.5)
    assert_allclose(band_da.sel(freqbands="high").values, 0.5)
    assert_allclose(band_da.coords["freqband_low"].values, [0.0, 5.0])
    assert_allclose(band_da.coords["freqband_high"].values, [5.0, 10.0])


def test_bandpower_accepts_featureresult_input() -> None:
    psd = _make_psd()
    feature_input = FeatureResult(
        artifacts={
            ".nc": Artifact(item=psd, writer=lambda path: psd.to_netcdf(path)),
        }
    )
    feature = bandpower(feature_input, bands={"broad": (0.0, 10.0)})
    band_da = feature.artifacts[".nc"].item

    assert band_da.shape == (2, 3, 1)
    assert "metadata" in band_da.attrs
