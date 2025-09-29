from __future__ import annotations

import numpy as np
import xarray as xr

from numpy.testing import assert_allclose

from cocofeats.definitions import Artifact, NodeResult
from cocofeats.nodes.spectral import bandpower, band_ratios


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
    feature_input = NodeResult(
        artifacts={
            ".nc": Artifact(item=psd, writer=lambda path: psd.to_netcdf(path)),
        }
    )
    feature = bandpower(feature_input, bands={"broad": (0.0, 10.0)})
    band_da = feature.artifacts[".nc"].item

    assert band_da.shape == (2, 3, 1)
    assert "metadata" in band_da.attrs


def test_band_ratios_default_permutations() -> None:
    psd = _make_psd()
    bands = {"low": (0.0, 5.0), "high": (5.0, 10.0), "mid": (2.0, 7.0)}
    band_feature = bandpower(psd, bands=bands)
    ratio_feature = band_ratios(band_feature)
    ratio_da = ratio_feature.artifacts[".nc"].item

    assert isinstance(ratio_da, xr.DataArray)
    assert ratio_da.dims == ("epochs", "spaces", "freqbandPairs")
    assert len(ratio_da.coords["freqbandPairs"].values) == 6
    assert "low/high" in ratio_da.coords["freqbandPairs"].values
    assert_allclose(ratio_da.sel(freqbandPairs="low/high").values, 1.0)
    assert_allclose(ratio_da.sel(freqbandPairs="mid/low").values, 1.0)


def test_band_ratios_handles_small_denominator() -> None:
    band_da = xr.DataArray(
        data=np.array([[[1.0, 0.0]]]),
        dims=("epochs", "spaces", "freqbands"),
        coords={
            "epochs": [0],
            "spaces": ["A"],
            "freqbands": ["top", "zero"],
        },
    )
    feature_input = NodeResult(
        artifacts={
            ".nc": Artifact(item=band_da, writer=lambda path: band_da.to_netcdf(path)),
        }
    )
    ratio_feature = band_ratios(feature_input, eps=1e-6)
    ratio_da = ratio_feature.artifacts[".nc"].item

    assert np.isnan(ratio_da.sel(freqbandPairs="top/zero").values).all()
    assert_allclose(ratio_da.sel(freqbandPairs="zero/top").values, 0.0)
