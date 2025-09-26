import json
import os
from collections.abc import Mapping
from typing import Any

import mne
import numpy as np
import xarray as xr

from cocofeats.definitions import Artifact, FeatureResult
from cocofeats.loaders import load_meeg
from cocofeats.loggers import get_logger
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas

from . import register_feature
from cocofeats.writers import _json_safe

log = get_logger(__name__)


DEFAULT_BANDS: dict[str, tuple[float, float]] = {
    "delta": (1.0, 4.0),
    "theta": (4.0, 8.0),
    "alpha": (8.0, 13.0),
    "beta": (13.0, 30.0),
    "pre_alpha": (5.5, 8.0),
    "slow_theta": (4.0, 5.5),
}


def _resolve_psd_dataarray(
    psd_like: FeatureResult | xr.DataArray | str | os.PathLike[str]
) -> xr.DataArray:
    if isinstance(psd_like, xr.DataArray):
        return psd_like

    if isinstance(psd_like, FeatureResult):
        if ".nc" not in psd_like.artifacts:
            raise ValueError("FeatureResult does not contain a .nc artifact to process.")
        candidate = psd_like.artifacts[".nc"].item
        if isinstance(candidate, xr.DataArray):
            return candidate
        if isinstance(candidate, (str, os.PathLike)):
            return xr.open_dataarray(candidate)
        raise ValueError("Unsupported artifact payload for .nc in FeatureResult.")

    if isinstance(psd_like, (str, os.PathLike)):
        return xr.open_dataarray(psd_like)

    raise ValueError("Input must be a FeatureResult, xarray.DataArray, or path to netCDF artifact.")


@register_feature
def spectrum(
    meeg: mne.io.BaseRaw | mne.BaseEpochs,
    compute_psd_kwargs: dict[str, Any] | None = None,
    extra_artifacts: bool = True,
) -> FeatureResult:
    """
    Compute the power spectral density of M/EEG data.

    Parameters
    ----------
    meeg : mne.io.BaseRaw or mne.BaseEpochs
        The M/EEG data to analyze. Can be raw data or epochs.
    compute_psd_kwargs : dict, optional
        Additional keyword arguments to pass to `mne.compute_psd`.
    extra_artifacts : bool, optional
        Whether to generate extra artifacts (MNE Report). Default is True.
    Returns
    -------
    dict
        A dictionary containing the power spectral density results, metadata, and artifacts (MNE Report).
    """

    if isinstance(meeg, FeatureResult):
        if ".fif" in meeg.artifacts:
            meeg = meeg.artifacts[".fif"].item
        else:
            raise ValueError("FeatureResult does not contain a .fif artifact to process.")

    if isinstance(meeg, str | os.PathLike):
        meeg = load_meeg(meeg)
        log.debug("MNEReport: loaded MNE object from file", input=meeg)

    if compute_psd_kwargs is None:
        compute_psd_kwargs = {}

    if "fmax" in compute_psd_kwargs:
        if (
            compute_psd_kwargs["fmax"] is not None
            and compute_psd_kwargs["fmax"] > meeg.info["sfreq"] / 2
        ):
            log.warning("fmax is greater than Nyquist frequency, adjusting to Nyquist")
            compute_psd_kwargs["fmax"] = meeg.info["sfreq"] / 2

    spectra = meeg.compute_psd(**compute_psd_kwargs)
    log.debug("MNEReport: computed spectra", spectra=spectra)

    if extra_artifacts:
        report = mne.Report(title="Spectrum", verbose="error")
        report.add_figure(spectra.plot(show=False), title="Spectrum")
        log.debug("MNEReport: computed report")

    extra_artifact = Artifact(
        item=report, writer=lambda path: report.save(path, overwrite=True, open_browser=False)
    )
    if isinstance(meeg, mne.io.BaseRaw):
        this_xarray = xr.DataArray(
            data=spectra.get_data(),
            dims=["spaces", "frequencies"],
            coords={"spaces": spectra.ch_names, "frequencies": spectra.freqs},
        )
    elif isinstance(meeg, mne.BaseEpochs):
        this_xarray = xr.DataArray(
            data=spectra.get_data(),
            dims=["epochs", "spaces", "frequencies"],
            coords={
                "epochs": list(range(len(spectra))),
                "spaces": spectra.ch_names,
                "frequencies": spectra.freqs,
            },
        )

    this_metadata = {
        "compute_psd_kwargs": compute_psd_kwargs,
    }

    this_xarray.attrs["metadata"] = json.dumps(this_metadata, indent=2)

    # Also add metadata to the report
    if extra_artifacts:
        extra_artifact.item.add_html(
            f"<pre>{json.dumps(this_metadata, indent=2)}</pre>",
            title="Metadata",
            section="Metadata",
        )

    artifacts = {".nc": Artifact(item=this_xarray, writer=lambda path: this_xarray.to_netcdf(path))}

    if extra_artifacts:
        artifacts[".report.html"] = extra_artifact

    out = FeatureResult(artifacts=artifacts)
    return out


import imageio
import matplotlib.pyplot as plt


def _make_gifs(psd_xarray: xr.DataArray) -> dict[str, Artifact]:
    """
    Generate GIFs depending on PSD dimensionality.
    - (spaces, frequencies): one GIF across channels
    - (epochs, spaces, frequencies): one GIF per channel across epochs

    Each frame has the channel name (and epoch number if present).
    """
    artifacts: dict[str, Artifact] = {}

    # Case 1: Only channels × freqs
    if list(psd_xarray.dims) == ["spaces", "frequencies"]:
        freqs = psd_xarray.coords["frequencies"].values
        spaces = psd_xarray.coords["spaces"].values

        frames = []
        for i, ch in enumerate(spaces):
            fig, ax = plt.subplots()
            ax.plot(freqs, psd_xarray[i, :].values)
            ax.set_title(f"Channel: {ch}")
            ax.set_xlabel("Frequency (Hz)")
            ax.set_ylabel("PSD")

            canvas = FigureCanvas(fig)
            canvas.draw()
            frame = np.frombuffer(canvas.buffer_rgba(), dtype=np.uint8)
            frame = frame.reshape(canvas.get_width_height()[::-1] + (4,))
            frame = frame[:, :, :3]  # keep RGB only
            frames.append(frame)
            plt.close(fig)

        artifacts[".channels.gif"] = Artifact(
            item=frames,
            writer=lambda path, frames=frames: imageio.mimsave(path, frames, fps=2)
        )

    # Case 2: Epochs × Channels × Freqs
    elif list(psd_xarray.dims) == ["epochs", "spaces", "frequencies"]:
        freqs = psd_xarray.coords["frequencies"].values
        spaces = psd_xarray.coords["spaces"].values
        epochs = psd_xarray.coords["epochs"].values

        for j, ch in enumerate(spaces):
            frames = []
            for i, ep in enumerate(epochs):
                fig, ax = plt.subplots()
                ax.plot(freqs, psd_xarray[i, j, :].values)
                ax.set_title(f"Channel: {ch}, Epoch: {ep}")
                ax.set_xlabel("Frequency (Hz)")
                ax.set_ylabel("PSD")

                canvas = FigureCanvas(fig)
                canvas.draw()
                frame = np.frombuffer(canvas.buffer_rgba(), dtype=np.uint8)
                frame = frame.reshape(canvas.get_width_height()[::-1] + (4,))
                frame = frame[:, :, :3]  # keep RGB only
                frames.append(frame)
                plt.close(fig)

            artifacts[f".{ch}.gif"] = Artifact(
                item=frames,
                writer=lambda path, frames=frames: imageio.mimsave(path, frames, fps=2)
            )

    return artifacts


@register_feature
def spectrum_array(
    meeg: mne.io.BaseRaw | mne.BaseEpochs,
    method: str = "welch",
    method_kwargs: dict[str, Any] | None = None,
    extra_artifacts: bool = True,
) -> FeatureResult:
    """Compute PSD from array data using Welch or multitaper algorithms.

    Parameters
    ----------
    meeg : mne.io.BaseRaw or mne.BaseEpochs
        The M/EEG data to analyze. Can be raw data or epochs.
    method : {"welch", "multitaper"}, optional
        PSD estimation routine to call. Defaults to "welch".
    method_kwargs : dict, optional
        Extra keyword arguments forwarded to the selected MNE function.

    Returns
    -------
    FeatureResult
        An object containing the PSD as a ``.nc`` artifact (``xarray.DataArray``)
        plus metadata describing the output dimensions. When multitaper is used
        with ``output='complex'``, taper weights are stored as an additional
        ``.weights.nc`` artifact.
    """

    method = method.lower()
    if method not in {"welch", "multitaper"}:
        raise ValueError("method must be either 'welch' or 'multitaper'")

    method_kwargs = dict(method_kwargs or {})

    if isinstance(meeg, FeatureResult):
        if ".fif" in meeg.artifacts:
            meeg = meeg.artifacts[".fif"].item
        else:
            raise ValueError("FeatureResult does not contain a .fif artifact to process.")

    if isinstance(meeg, str | os.PathLike):
        meeg = load_meeg(meeg)
        log.debug("MNEReport: loaded MNE object from file", input=meeg)

    if isinstance(meeg, mne.io.BaseRaw):
        data_values = meeg.get_data(return_times=True)
        times = meeg.times
        sfreq = meeg.info["sfreq"]
        time_dim = "times"
        base_dims = ["spaces"]
        base_coords = {"spaces": meeg.ch_names}
    elif isinstance(meeg, mne.BaseEpochs):
        data_values = meeg.get_data()
        times = meeg.times
        sfreq = meeg.info["sfreq"]
        time_dim = "times"
        base_dims = ["epochs", "spaces"]
        base_coords = {
            "epochs": list(range(len(meeg))),
            "spaces": meeg.ch_names,
        }

    # times is the last dimension
    # data_values shape is (..., time)
    if data_values.shape[-1] != len(times):
        raise ValueError("Data last dimension must be time")


    psd_func = {
        "welch": mne.time_frequency.psd_array_welch,
        "multitaper": mne.time_frequency.psd_array_multitaper,
    }[method]

    psd_result = psd_func(data_values, sfreq=sfreq, **method_kwargs)
    weights = None
    if method == "multitaper" and isinstance(psd_result, tuple) and len(psd_result) == 3:
        psds, freqs, weights = psd_result
    else:
        psds, freqs = psd_result  # type: ignore[misc]

    sample_dims = list(base_dims)
    psd_dims = list(sample_dims)
    psd_coords = dict(base_coords)
    dimension_origins = {dim: "input" for dim in sample_dims}

    dimension_details: list[dict[str, Any]] = []
    for idx, dim in enumerate(sample_dims):
        dimension_details.append(
            {
                "name": dim,
                "origin": "input",
                "size": int(psds.shape[idx]),
            }
        )

    additional_axes: list[str] = []
    average = None
    output_mode = None

    if method == "welch":
        average = method_kwargs.get("average", "mean")
        freq_axis = len(sample_dims)
        psd_dims.append("frequencies")
        psd_coords["frequencies"] = np.asarray(freqs)
        dimension_origins["frequencies"] = "frequency"
        dimension_details.append(
            {
                "name": "frequencies",
                "origin": "welch_frequency",
                "size": int(psds.shape[freq_axis]),
            }
        )
        additional_axes.append("frequencies")
        if average is None:
            seg_axis = freq_axis + 1
            psd_dims.append("segments")
            psd_coords["segments"] = np.arange(psds.shape[seg_axis])
            dimension_origins["segments"] = "welch_segments"
            dimension_details.append(
                {
                    "name": "segments",
                    "origin": "welch_segments",
                    "size": int(psds.shape[seg_axis]),
                }
            )
            additional_axes.append("segments")
    elif method == "multitaper":
        output_mode = method_kwargs.get("output", "power")
        if output_mode == "complex":
            taper_axis = len(sample_dims)
            psd_dims.append("tapers")
            psd_coords["tapers"] = np.arange(psds.shape[taper_axis])
            dimension_origins["tapers"] = "multitaper_tapers"
            dimension_details.append(
                {
                    "name": "tapers",
                    "origin": "multitaper_tapers",
                    "size": int(psds.shape[taper_axis]),
                }
            )
            additional_axes.append("tapers")
            freq_axis = taper_axis + 1
        else:
            freq_axis = len(sample_dims)
        psd_dims.append("frequencies")
        psd_coords["frequencies"] = np.asarray(freqs)
        dimension_origins["frequencies"] = "frequency"
        dimension_details.append(
            {
                "name": "frequencies",
                "origin": "multitaper_frequency",
                "size": int(psds.shape[freq_axis]),
            }
        )
        additional_axes.append("frequencies")

    psd_xarray = xr.DataArray(data=psds, dims=psd_dims, coords=psd_coords)

    dimension_notes = {
        "time_dim": time_dim,
        "replaced_with": "frequencies",
        "additional_axes": additional_axes,
    }

    metadata: dict[str, Any] = {
        "method": method,
        "method_kwargs": _json_safe(method_kwargs),
        "sampling_frequency": sfreq,
        "input": {
            "shape": [int(v) for v in data_values.shape],
            "dims": sample_dims + [time_dim],
        },
        "output": {
            "shape": [int(v) for v in psds.shape],
            "dims": psd_dims,
            "dimension_details": dimension_details,
        },
        "dimension_notes": dimension_notes,
    }

    if method == "welch":
        metadata["output"]["average"] = _json_safe(average)
    if method == "multitaper":
        metadata["output"]["output_parameter"] = output_mode or "power"

    weights_xarray: xr.DataArray | None = None
    if weights is not None:
        weights_array = np.asarray(weights)
        weights_dims = list(sample_dims)
        weights_coords = dict(base_coords)
        weights_dims.append("tapers")
        weights_coords["tapers"] = np.arange(weights_array.shape[-1])
        weights_xarray = xr.DataArray(weights_array, dims=weights_dims, coords=weights_coords)
        weights_metadata = {
            "method": method,
            "description": "DPSS weights returned by psd_array_multitaper",
            "shape": [int(v) for v in weights_array.shape],
            "dims": weights_dims,
        }
        weights_xarray.attrs["metadata"] = json.dumps(
            weights_metadata,
            indent=2,
            default=_json_safe,
        )
        metadata["weights_shape"] = [int(v) for v in weights_array.shape]

    psd_xarray.attrs["metadata"] = json.dumps(metadata, indent=2, default=_json_safe)

    artifacts: dict[str, Artifact] = {
        ".nc": Artifact(item=psd_xarray, writer=lambda path: psd_xarray.to_netcdf(path,engine='netcdf4',format='NETCDF4')),
        #".zarr": Artifact(item=psd_xarray, writer=lambda path: psd_xarray.to_zarr(path)),
    }
    if weights_xarray is not None:
        artifacts[".weights.nc"] = Artifact(
            item=weights_xarray, writer=lambda path: weights_xarray.to_netcdf(path)
        )

    # Support visualization if dimension are 2 (spaces, frequencies) or 3 (epochs, spaces, frequencies)

    if extra_artifacts:
        #gif_artifacts = _make_gifs(psd_xarray)
        #artifacts.update(gif_artifacts)

        report = mne.Report(title="Spectrum", verbose="error")
        if len(psd_dims) == 2 and psd_dims == ["spaces", "frequencies"]:
            fig = psd_xarray.plot(
                x="frequencies",
                y="spaces",
                yincrease=False,
                figsize=(8, 12),
                aspect="auto",
            ).figure
            report.add_figure(fig, title="Spectrum")

        elif len(psd_dims) == 3 and psd_dims == ["epochs", "spaces", "frequencies"]:
            mean_over_epochs = psd_xarray.mean(dim="epochs")
            fig = mean_over_epochs.plot(
                x="frequencies",
                y="spaces",
                yincrease=False,
                figsize=(8, 12),
                aspect="auto",
            ).figure
            report.add_figure(fig, title="Mean Spectrum over epochs")
        report.add_html(
            f"<pre>{json.dumps(metadata, indent=2)}</pre>",
            title="Metadata",
            section="Metadata",
        )
        artifacts[".report.html"] = Artifact(
            item=report, writer=lambda path: report.save(path, overwrite=True, open_browser=False)
        )

    return FeatureResult(artifacts=artifacts)


@register_feature
def bandpower(
    psd_like: FeatureResult | xr.DataArray | str | os.PathLike[str],
    *,
    bands: Mapping[str, tuple[float, float]] | None = None,
    freq_dim: str = "frequencies",
    relative: bool = False,
) -> FeatureResult:
    """Compute absolute or relative band power from a PSD ``xarray.DataArray``.

    Parameters
    ----------
    psd_like : FeatureResult | xarray.DataArray | path-like
        Output from ``spectrum``/``spectrum_array`` or a compatible ``xarray`` artifact.
    bands : mapping, optional
        Frequency bands as ``{"label": (low, high)}``. If omitted ``DEFAULT_BANDS`` is used.
    freq_dim : str, optional
        Name of the frequency dimension in the PSD. Defaults to ``"frequencies"``.
    relative : bool, optional
        If ``True`` each band power is normalised by the total power across ``freq_dim``.

    Returns
    -------
    FeatureResult
        ``.nc`` artifact whose ``freq_dim`` is replaced by ``freqbands`` containing band powers.
    """

    psd_xr = _resolve_psd_dataarray(psd_like)

    if freq_dim not in psd_xr.dims:
        raise ValueError(f"Frequency dimension '{freq_dim}' not present in input dims: {psd_xr.dims}")

    bands_dict = dict(bands or DEFAULT_BANDS)
    if not bands_dict:
        raise ValueError("At least one frequency band must be provided.")

    freqs = psd_xr.coords.get(freq_dim)
    if freqs is None:
        raise ValueError(f"Frequency dimension '{freq_dim}' must have coordinate values.")

    if freqs.ndim != 1:
        raise ValueError("Frequency coordinate must be one-dimensional.")

    freq_axis = psd_xr.get_axis_num(freq_dim)

    total_power = psd_xr.integrate(freq_dim)

    band_arrays: list[xr.DataArray] = []
    band_edges: list[tuple[float, float]] = []

    for label, band_range in bands_dict.items():
        if len(band_range) != 2:
            raise ValueError(f"Band '{label}' must be a (low, high) pair.")

        low, high = map(float, band_range)
        if not np.isfinite(low) or not np.isfinite(high): #TODO: maybe we should allow inf? (as get everything below or above a threshold)
            raise ValueError(f"Band '{label}' has non-finite boundaries: {band_range}.")
        if high <= low:
            raise ValueError(f"Band '{label}' must have high > low (got {band_range}).")

        band_slice = psd_xr.sel({freq_dim: slice(low, high)})
        if band_slice.sizes.get(freq_dim, 0) == 0:
            band_power = xr.full_like(total_power, np.nan)
        else:
            band_power = band_slice.integrate(freq_dim)

        # Insert the new freqbands dimension where the original frequencies lived
        band_power = band_power.expand_dims({"freqbands": [label]}, axis=freq_axis)
        band_arrays.append(band_power)
        band_edges.append((low, high))

    band_power_xr = xr.concat(band_arrays, dim="freqbands")

    # Restore dimension order so freqbands replaces the frequency dimension position
    original_dims = list(psd_xr.dims)
    target_dims = ["freqbands" if dim == freq_dim else dim for dim in original_dims]
    band_power_xr = band_power_xr.transpose(*target_dims)

    band_power_xr = band_power_xr.assign_coords(
        freqbands=list(bands_dict),
    )
    band_power_xr.coords["freqband_low"] = ("freqbands", [edge[0] for edge in band_edges])
    band_power_xr.coords["freqband_high"] = ("freqbands", [edge[1] for edge in band_edges])

    if relative:
        denom = total_power
        with np.errstate(divide="ignore", invalid="ignore"):
            normalised = band_power_xr / denom
        band_power_xr = xr.where(denom == 0, np.nan, normalised)

    metadata: dict[str, Any] = {
        "bands": {
            label: {"low": float(low), "high": float(high)} for label, (low, high) in bands_dict.items()
        },
        "relative": relative,
        "freq_dim": freq_dim,
        "input_dims": list(psd_xr.dims),
        "input_shape": [int(psd_xr.sizes[dim]) for dim in psd_xr.dims],
        "output_dims": list(band_power_xr.dims),
        "output_shape": [int(band_power_xr.sizes[dim]) for dim in band_power_xr.dims],
        "integration": "xarray.DataArray.integrate (trapezoidal)",
    }

    source_metadata = psd_xr.attrs.get("metadata")
    if source_metadata is not None:
        metadata["source_metadata"] = source_metadata

    band_power_xr.attrs["metadata"] = json.dumps(metadata, indent=2, default=_json_safe)

    artifacts = {
        ".nc": Artifact(item=band_power_xr, writer=lambda path: band_power_xr.to_netcdf(path)),
    }

    return FeatureResult(artifacts=artifacts)
