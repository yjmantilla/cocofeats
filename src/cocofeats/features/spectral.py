import copy
import io
import json
import math
import os
import time
from collections.abc import Mapping, Sequence
from itertools import permutations
from typing import Any, Literal

import mne
import numpy as np
import xarray as xr
from fooof import FOOOF

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
    "preAlpha": (5.5, 8.0),
    "slowTheta": (4.0, 5.5),
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


def _resolve_eval_strings(value: Any) -> Any:
    """Recursively interpret ``eval%`` prefixed expressions within nested structures."""

    if isinstance(value, dict):
        return {key: _resolve_eval_strings(sub_value) for key, sub_value in value.items()}
    if isinstance(value, list):
        return [_resolve_eval_strings(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_resolve_eval_strings(item) for item in value)
    if isinstance(value, str) and value.startswith("eval%"):
        expression = value.removeprefix("eval%")
        namespace = {"np": np, "math": math}
        return eval(expression, namespace)  # noqa: S307 - explicit request for dynamic evaluation
    return value


@register_feature
def spectrum(
    meeg: mne.io.BaseRaw | mne.BaseEpochs,
    compute_psd_kwargs: dict[str, Any] | None = None,
    extra_artifacts: bool = False,
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


@register_feature
def spectrum_array(
    meeg: mne.io.BaseRaw | mne.BaseEpochs,
    method: str = "welch",
    method_kwargs: dict[str, Any] | None = None,
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
        An object containing the PSD as a ``.nc`` artifact (``xarray.Dataset``)
        plus metadata describing the output dimensions. When multitaper is used
        with ``output='complex'``, taper weights are included in the same
        dataset under the ``weights`` variable.
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

    psd_xarray = xr.DataArray(data=psds, dims=psd_dims, coords=psd_coords)
    metadata_json = json.dumps(metadata, indent=2, default=_json_safe)
    psd_xarray.attrs["metadata"] = metadata_json

    dataset_vars: dict[str, xr.DataArray] = {"spectrum": psd_xarray}
    if weights_xarray is not None:
        dataset_vars["weights"] = weights_xarray

    psd_dataset = xr.Dataset(data_vars=dataset_vars)
    psd_dataset.attrs["metadata"] = metadata_json

    artifacts: dict[str, Artifact] = {
        ".nc": Artifact(
            item=psd_dataset,
            writer=lambda path: psd_dataset.to_netcdf(path, engine='netcdf4', format='NETCDF4'),
        ),
    }

    return FeatureResult(artifacts=artifacts)


@register_feature
def fooof(
    psd_like: FeatureResult | xr.DataArray | str | os.PathLike[str],
    *,
    freq_dim: str = "frequencies",
    freqs: Sequence[float] | np.ndarray | None = None,
    fooof_options: Mapping[str, Any] | None = None,
    allow_eval_strings: bool = True,
    failure_value: str | None = "{}",
    include_timings: bool = True,
) -> FeatureResult:
    """Fit FOOOF models for every non-frequency slice in a PSD ``xarray`` artifact.

    Parameters
    ----------
    psd_like : FeatureResult | xarray.DataArray | path-like
        Output from ``spectrum``/``spectrum_array`` or a compatible ``xarray`` artifact.
    freq_dim : str, optional
        Name of the frequency dimension. Defaults to ``"frequencies"``.
    freqs : sequence of float, optional
        Explicit frequency values. If omitted they are read from the coordinate
        of ``freq_dim``.
    fooof_options : mapping, optional
        Nested configuration dictionary using the legacy layout
        ``{"FOOOF": {...}, "fit": {...}, "save": {...}, "freq_res": ...}``.
    allow_eval_strings : bool, optional
        Interpret values that start with ``"eval%"`` using ``eval`` with
        ``numpy``/``math`` in scope. Mirrors the historic behaviour.
    failure_value : str | None, optional
        Fallback string stored when a FOOOF fit fails. Defaults to ``"{}"``.
    include_timings : bool, optional
        Whether to output a timings artifact (seconds per fit).

    Returns
    -------
    FeatureResult
        ``.fooof.nc`` artifact with JSON serialisations of each fitted model and
        optionally a ``.fooof_timings.nc`` artifact with execution times.
    """

    psd_xr = _resolve_psd_dataarray(psd_like)

    if freq_dim not in psd_xr.dims:
        raise ValueError(
            f"Frequency dimension '{freq_dim}' not present in input dims: {psd_xr.dims}"
        )

    if freqs is None:
        coord = psd_xr.coords.get(freq_dim)
        if coord is None:
            raise ValueError(
                "Frequency dimension must provide coordinates when 'freqs' is not supplied."
            )
        freq_values = np.asarray(coord.values, dtype=float)
    else:
        freq_values = np.asarray(freqs, dtype=float)

    if freq_values.ndim != 1:
        raise ValueError("Frequency information must be one-dimensional.")

    n_freqs = freq_values.size
    if n_freqs == 0:
        raise ValueError("Frequency array must contain at least one value.")

    options = copy.deepcopy(dict(fooof_options or {}))
    if allow_eval_strings:
        options = _resolve_eval_strings(options)

    fooof_init_kwargs = dict(options.pop("FOOOF", {}))
    fit_kwargs = dict(options.pop("fit", {}))
    save_kwargs = dict(options.pop("save", {}))
    freq_res = options.pop("freq_res", None)
    unused_options = options  # Whatever remains is captured for metadata purposes.

    save_defaults = {"save_results": True, "save_settings": True, "save_data": False}
    for key, value in save_defaults.items():
        save_kwargs.setdefault(key, value)

    if freq_res is not None:
        freq_res = float(freq_res)
        if freq_res <= 0:
            raise ValueError("freq_res must be a positive float if provided.")

    if n_freqs > 1:
        diffs = np.diff(freq_values)
        valid_diffs = diffs[np.nonzero(diffs)]
        current_res = float(np.median(np.abs(valid_diffs))) if valid_diffs.size else float("nan")
    else:
        current_res = float("nan")

    downsample_step = 1
    if freq_res is not None and n_freqs > 1 and np.isfinite(current_res) and current_res > 0:
        if freq_res < current_res:
            log.warning(
                "Requested freq_res is finer than available resolution; skipping downsampling",
                requested=freq_res,
                available=current_res,
            )
        else:
            downsample_step = max(1, int(math.ceil(freq_res / current_res)))

    freq_values_downsampled = freq_values[::downsample_step]
    if freq_values_downsampled.size == 0:
        raise ValueError("Downsampling removed all frequency points; check freq_res setting.")

    other_dims = [dim for dim in psd_xr.dims if dim != freq_dim]
    transposed = psd_xr.transpose(*(other_dims + [freq_dim]))
    psd_values = np.asarray(transposed.values)
    if psd_values.ndim == 1:
        psd_values = psd_values[np.newaxis, :]

    other_shape = [int(transposed.sizes[dim]) for dim in other_dims]
    flattened = psd_values.reshape(-1, n_freqs)

    fooof_payloads = np.empty(flattened.shape[0], dtype=object)
    timings = np.full(flattened.shape[0], np.nan, dtype=float)
    failure_records: list[dict[str, Any]] = []

    coords_cache: dict[str, np.ndarray] = {}
    coords_for_output: dict[str, np.ndarray] = {}
    for dim in other_dims:
        if dim in transposed.coords:
            values = np.asarray(transposed.coords[dim].values)
        else:
            values = np.arange(transposed.sizes[dim])
        coords_cache[dim] = values
        coords_for_output[dim] = values

    fallback_value = "" if failure_value is None else str(failure_value)

    for flat_idx in range(flattened.shape[0]):
        if other_dims:
            unravel = np.unravel_index(flat_idx, tuple(other_shape))
            coord_mapping = {
                dim: _json_safe(coords_cache[dim][unravel[idx]])
                for idx, dim in enumerate(other_dims)
            }
        else:
            unravel = ()
            coord_mapping = {}

        start = time.perf_counter()
        try:
            signal = np.asarray(flattened[flat_idx])
            if signal.size != n_freqs:
                raise ValueError(
                    "Each PSD slice must have the same number of frequencies as the coordinate array."
                )

            signal_to_fit = signal[::downsample_step]
            if signal_to_fit.size != freq_values_downsampled.size:
                raise ValueError("Downsampled signal and frequency vectors must be the same length.")

            fm = FOOOF(verbose=False, **fooof_init_kwargs)
            fm.fit(freq_values_downsampled, signal_to_fit, **fit_kwargs)

            buffer = io.StringIO()
            fm.save(
                buffer,
                file_path=None,
                append=False,
                save_results=save_kwargs.get("save_results", True),
                save_settings=save_kwargs.get("save_settings", True),
                save_data=save_kwargs.get("save_data", False),
            )
            payload = buffer.getvalue() or json.dumps({}, indent=2)
            fooof_payloads[flat_idx] = payload
        except Exception as exc:  # noqa: BLE001 - domain specific fallbacks are required
            duration = time.perf_counter() - start
            timings[flat_idx] = duration
            fooof_payloads[flat_idx] = fallback_value
            failure_records.append({
                "index": int(flat_idx),
                "coords": coord_mapping,
                "error": repr(exc),
            })
            log.warning("FOOOF fit failed", coords=coord_mapping, error=str(exc))
            continue

        timings[flat_idx] = time.perf_counter() - start

    if other_dims:
        result_shape = tuple(other_shape)
        coords = coords_for_output
    else:
        result_shape = ()
        coords = {}

    result_array = fooof_payloads.reshape(result_shape)
    fooof_xr = xr.DataArray(result_array, dims=other_dims, coords=coords, name="fooof_json")

    metadata: dict[str, Any] = {
        "freq_dim": freq_dim,
        "frequencies": _json_safe(freq_values),
        "frequencies_downsampled": _json_safe(freq_values_downsampled),
        "downsample_step": int(downsample_step),
        "fooof_kwargs": _json_safe(fooof_init_kwargs),
        "fit_kwargs": _json_safe(fit_kwargs),
        "save_kwargs": _json_safe(save_kwargs),
        "unused_options": _json_safe(unused_options) if unused_options else None,
        "allow_eval_strings": allow_eval_strings,
        "failure_value": fallback_value,
        "input_dims": list(psd_xr.dims),
        "input_shape": [int(psd_xr.sizes[dim]) for dim in psd_xr.dims],
        "output_dims": list(fooof_xr.dims),
        "output_shape": [int(fooof_xr.sizes[dim]) for dim in fooof_xr.dims],
        "failures": failure_records,
        "frequency_resolution": {
            "requested": freq_res,
            "available": current_res,
        },
    }

    source_metadata = psd_xr.attrs.get("metadata")
    if source_metadata is not None:
        metadata["source_metadata"] = source_metadata

    fooof_xr.attrs["metadata"] = json.dumps(metadata, indent=2, default=_json_safe)

    artifacts: dict[str, Artifact] = {
        ".fooof.nc": Artifact(
            item=fooof_xr,
            writer=lambda path: fooof_xr.to_netcdf(path, engine="netcdf4", format="NETCDF4"),
        )
    }

    if include_timings:
        timings_array = timings.reshape(result_shape)
        fooof_timings = xr.DataArray(
            timings_array,
            dims=other_dims,
            coords=coords,
            name="fooof_fit_seconds",
        )
        timing_metadata = {
            "description": "Wall-clock duration per FOOOF fit",
            "unit": "seconds",
        }
        fooof_timings.attrs["metadata"] = json.dumps(timing_metadata, indent=2, default=_json_safe)
        artifacts[".fooof_timings.nc"] = Artifact(
            item=fooof_timings,
            writer=lambda path: fooof_timings.to_netcdf(path, engine="netcdf4", format="NETCDF4"),
        )

    return FeatureResult(artifacts=artifacts)




@register_feature
def fooof_scalars(
    fooof_like: FeatureResult | xr.DataArray | str | os.PathLike[str],
    *,
    component: Literal["aperiodic_params", "r_squared", "error", "all"] = "aperiodic_params",
    freq_dim: str = "frequencies",
) -> FeatureResult:
    """Extract scalar outputs from serialized FOOOF models.

    Parameters
    ----------
    fooof_like : FeatureResult | xarray.DataArray | path-like
        Artifact generated by :func:`fooof`, containing JSON strings per slice.
    component : {"aperiodic_params", "r_squared", "error", "all"}, optional
        Which FOOOF scalar to extract. ``aperiodic_params`` returns offset/(knee)/exponent
        per slice, ``r_squared`` and ``error`` provide the fit metrics, and ``all`` outputs
        the full table as a single artifact.
    freq_dim : str, optional
        Name of the frequency dimension in the original PSD. Used to preserve dimension order
        when aligning with other outputs. Defaults to ``"frequencies"``.

    Returns
    -------
    FeatureResult
        ``.nc`` artifact(s) containing the requested scalar(s).
    """

    component = component.lower()
    valid_components = {"aperiodic_params", "r_squared", "error", "all"}
    if component not in valid_components:
        raise ValueError("component must be one of {'aperiodic_params', 'r_squared', 'error', 'all'}")

    fooof_xr = _resolve_psd_dataarray(fooof_like)
    other_dims = list(fooof_xr.dims)
    other_shape = [int(fooof_xr.sizes[dim]) for dim in other_dims]
    flat_count = int(np.prod(other_shape)) if other_shape else 1

    fooof_flat = np.asarray(fooof_xr.values, dtype=object).reshape(flat_count)
    fooof_meta_raw = fooof_xr.attrs.get("metadata")
    try:
        fooof_meta = json.loads(fooof_meta_raw) if fooof_meta_raw else {}
    except json.JSONDecodeError:
        fooof_meta = {}

    loaded_models: list[FOOOF | None] = [None] * flat_count
    invalid_indices: list[int] = []

    for idx, payload in enumerate(fooof_flat):
        if not isinstance(payload, str) or not payload.strip():
            invalid_indices.append(idx)
            continue

        try:
            fm = FOOOF()
            fm.load(io.StringIO(payload))
            if not fm.has_model:
                raise ValueError("FOOOF object missing model")
        except Exception as exc:  # noqa: BLE001
            log.warning("Failed to load FOOOF payload for scalars", index=idx, error=str(exc))
            invalid_indices.append(idx)
            continue

        loaded_models[idx] = fm

    scalar_arrays: dict[str, np.ndarray] = {
        "aperiodic_params": np.empty((flat_count,), dtype=object),
        "r_squared": np.full((flat_count,), np.nan, dtype=float),
        "error": np.full((flat_count,), np.nan, dtype=float),
        "aperiodic_offset": np.empty((flat_count,), dtype=float),
        "aperiodic_knee": np.empty((flat_count,), dtype=float),
        "aperiodic_exponent": np.empty((flat_count,), dtype=float),
    }

    for idx, fm in enumerate(loaded_models):
        if fm is None:
            scalar_arrays["aperiodic_params"][idx] = None
            continue

        try:
            ap_params = getattr(fm, "aperiodic_params_", getattr(fm, "_aperiodic_params", None))
            if ap_params is None:
                raise ValueError("Missing aperiodic parameters")
            ap_params = np.asarray(ap_params, dtype=float)
            ap_mode = getattr(fm, "aperiodic_mode", getattr(fm, "aperiodic_mode_", "fixed"))
            if ap_mode == "knee" and ap_params.size < 3:
                raise ValueError("Knee mode expects three parameters")
            if ap_mode != "knee" and ap_params.size < 2:
                raise ValueError("Fixed mode expects two parameters")
            scalar_arrays["aperiodic_params"][idx] = ap_params.tolist()

            if ap_params.size == 2:
                scalar_arrays['aperiodic_offset'][idx] = float(ap_params[0])
                scalar_arrays['aperiodic_knee'][idx] = np.nan
                scalar_arrays['aperiodic_exponent'][idx] = float(ap_params[1])
            else:
                scalar_arrays['aperiodic_offset'][idx] = float(ap_params[0])
                scalar_arrays['aperiodic_knee'][idx] = float(ap_params[1])
                scalar_arrays['aperiodic_exponent'][idx] = float(ap_params[2])

            scalar_arrays["r_squared"][idx] = float(
                getattr(fm, "r_squared_", getattr(fm, "r_squared", np.nan))
            )
            scalar_arrays["error"][idx] = float(getattr(fm, "error_", getattr(fm, "error", np.nan)))
        except Exception as exc:  # noqa: BLE001
            log.warning("Failed to compute FOOOF scalar", index=idx, error=str(exc))
            invalid_indices.append(idx)
            scalar_arrays["aperiodic_params"][idx] = None
            scalar_arrays["r_squared"][idx] = np.nan
            scalar_arrays["error"][idx] = np.nan

    coords = {
        dim: (
            fooof_xr.coords[dim].values if dim in fooof_xr.coords else np.arange(fooof_xr.sizes[dim])
        )
        for dim in other_dims
    }

    name_map = {
        "aperiodic_offset": "fooof_aperiodic_offset",
        "aperiodic_knee": "fooof_aperiodic_knee",
        "aperiodic_exponent": "fooof_aperiodic_exponent",
        "r_squared": "fooof_r_squared",
        "error": "fooof_error",
    }

    metadata_base: dict[str, Any] = {
        "components": list(name_map),
        "invalid_count": len(set(invalid_indices)),
        "total_count": flat_count,
        "invalid_indices": sorted(set(invalid_indices)),
    }
    if fooof_meta:
        metadata_base["fooof_metadata"] = fooof_meta

    def make_array(key: str) -> xr.DataArray:
        data = scalar_arrays[key].reshape(other_shape)
        xarr = xr.DataArray(data, dims=other_dims, coords=coords, name=name_map[key])
        this_meta = dict(metadata_base)
        this_meta["component"] = key
        xarr.attrs["metadata"] = json.dumps(this_meta, indent=2, default=_json_safe)
        return xarr

    artifacts: dict[str, Artifact]
    if component == "all":
        artifacts = {}
        for key, suffix in (
            ("aperiodic_offset", ".apOffset.nc"),
            ("aperiodic_knee", ".apKnee.nc"),
            ("aperiodic_exponent", ".apExponent.nc"),
            ("r_squared", ".rSquared.nc"),
            ("error", ".error.nc"),
        ):
            xarr = make_array(key)
            artifacts[suffix] = Artifact(
                item=xarr,
                writer=lambda path, arr=xarr: arr.to_netcdf(path, engine="netcdf4", format="NETCDF4"),
            )
    else:
        mapped = {
            "aperiodic_params": ["aperiodic_offset", "aperiodic_knee", "aperiodic_exponent"],
            "r_squared": ["rSquared"],
            "error": ["error"],
        }
        keys = mapped.get(component, [component])
        if component == "aperiodic_params":
            artifacts = {}
            for key, suffix in zip(
                keys,
                (".apOffset.nc", ".apKnee.nc", ".apExponent.nc"),
            ):
                xarr = make_array(key)
                artifacts[suffix] = Artifact(
                    item=xarr,
                    writer=lambda path, arr=xarr: arr.to_netcdf(path, engine="netcdf4", format="NETCDF4"),
                )
        else:
            key = keys[0]
            xarr = make_array(key)
            artifacts = {
                ".nc": Artifact(
                    item=xarr,
                    writer=lambda path: xarr.to_netcdf(path, engine="netcdf4", format="NETCDF4"),
                )
            }

    return FeatureResult(artifacts=artifacts)

@register_feature
def fooof_component(
    fooof_like: FeatureResult | xr.DataArray | str | os.PathLike[str],
    *,
    component: Literal["aperiodic", "periodic", "residual", "all"] = "aperiodic",
    freq_dim: str = "frequencies",
) -> FeatureResult:
    """Derive linear-space components directly from serialized FOOOF models.

    Parameters
    ----------
    fooof_like : FeatureResult | xarray.DataArray | path-like
        Artifact generated by :func:`fooof`, containing JSON strings per slice.
    component : {"aperiodic", "periodic", "residual", "all"}, optional
        ``aperiodic`` returns the FOOOF background spectrum, ``periodic`` returns
        the modelled oscillatory power (Gaussians minus background), ``residual``
        subtracts the background from the original power spectrum, and ``all``
        emits all three components as separate artifacts.
    freq_dim : str, optional
        Name of the frequency dimension for the output. Defaults to ``"frequencies"``.

    Returns
    -------
    FeatureResult
        ``.nc`` artifact with the requested component in linear power units.
    """

    component = component.lower()
    valid_components = {"aperiodic", "periodic", "residual", "all"}
    if component not in valid_components:
        raise ValueError("component must be one of {'aperiodic', 'periodic', 'residual', 'all'}")

    fooof_xr = _resolve_psd_dataarray(fooof_like)
    other_dims = list(fooof_xr.dims)
    other_shape = [int(fooof_xr.sizes[dim]) for dim in other_dims]
    flat_count = int(np.prod(other_shape)) if other_shape else 1

    fooof_flat = np.asarray(fooof_xr.values, dtype=object).reshape(flat_count)

    fooof_meta_raw = fooof_xr.attrs.get("metadata")
    try:
        fooof_meta = json.loads(fooof_meta_raw) if fooof_meta_raw else {}
    except json.JSONDecodeError:
        fooof_meta = {}

    loaded_models: list[FOOOF | None] = [None] * flat_count
    invalid_indices: list[int] = []
    freq_values_model: np.ndarray | None = None

    for idx, payload in enumerate(fooof_flat):
        if not isinstance(payload, str) or not payload.strip():
            invalid_indices.append(idx)
            continue

        try:
            fm = FOOOF()
            fm.load(io.StringIO(payload))
            if not fm.has_model:
                raise ValueError("FOOOF object missing model")
        except Exception as exc:  # noqa: BLE001 - domain specific fallbacks are required
            log.warning("Failed to load FOOOF payload", index=idx, error=str(exc))
            invalid_indices.append(idx)
            continue

        freqs = np.asarray(getattr(fm, "freqs", None))
        if freqs.size == 0:
            invalid_indices.append(idx)
            continue

        if freq_values_model is None:
            freq_values_model = freqs.astype(float, copy=True)
        elif freqs.shape != freq_values_model.shape or not np.allclose(freqs, freq_values_model):
            log.warning("FOOOF frequencies mismatch; marking slice invalid", index=idx)
            invalid_indices.append(idx)
            continue

        loaded_models[idx] = fm

    if freq_values_model is None or freq_values_model.size == 0:
        raise ValueError("No valid FOOOF models with frequency information were found.")

    freq_len = freq_values_model.size
    component_arrays: dict[str, np.ndarray] = {
        "aperiodic": np.full((flat_count, freq_len), np.nan, dtype=float),
        "periodic": np.full((flat_count, freq_len), np.nan, dtype=float),
        "residual": np.full((flat_count, freq_len), np.nan, dtype=float),
    }

    for idx, fm in enumerate(loaded_models):
        if fm is None:
            continue

        try:
            ap_fit_log = np.asarray(fm._ap_fit, dtype=float)
            if ap_fit_log.size != freq_len:
                raise ValueError("Aperiodic fit length mismatch")
            ap_linear = np.power(10.0, ap_fit_log)

            model_log = np.asarray(getattr(fm, "fooofed_spectrum_", None), dtype=float)
            power_log = np.asarray(getattr(fm, "power_spectrum", None), dtype=float)
            if model_log.size != freq_len or power_log.size != freq_len:
                raise ValueError("FOOOF spectra length mismatch")

            model_linear = np.power(10.0, model_log)
            periodic_linear = np.clip(model_linear - ap_linear, a_min=0.0, a_max=None)
            residual_linear = np.power(10.0, power_log) - model_linear

            component_arrays["aperiodic"][idx] = ap_linear
            component_arrays["periodic"][idx] = periodic_linear
            component_arrays["residual"][idx] = residual_linear
        except Exception as exc:  # noqa: BLE001
            log.warning("Failed to compute FOOOF component", index=idx, error=str(exc))
            invalid_indices.append(idx)
            for key in ("aperiodic", "periodic", "residual"):
                component_arrays[key][idx] = np.nan

    output_shape = tuple(other_shape + [freq_len]) if other_dims else (freq_len,)
    output_dims = other_dims + [freq_dim]

    coords: dict[str, Any] = {}
    for dim in other_dims:
        coord = fooof_xr.coords.get(dim)
        coords[dim] = coord.values if coord is not None else np.arange(fooof_xr.sizes[dim])
    coords[freq_dim] = freq_values_model

    name_map = {
        "aperiodic": "fooof_aperiodic_linear",
        "periodic": "fooof_periodic_linear",
        "residual": "fooof_residual_linear",
    }

    def build_component(key: str) -> xr.DataArray:
        data = component_arrays[key].reshape(output_shape)
        return xr.DataArray(data, dims=output_dims, coords=coords, name=name_map[key])

    metadata_base: dict[str, Any] = {
        "frequency_dimension": freq_dim,
        "frequencies": _json_safe(freq_values_model.tolist()),
        "invalid_count": len(set(invalid_indices)),
        "total_count": flat_count,
        "invalid_indices": sorted(set(invalid_indices)),
    }
    if fooof_meta:
        metadata_base["fooof_metadata"] = fooof_meta

    def attach_metadata(xarr: xr.DataArray, key: str) -> xr.DataArray:
        this_meta = dict(metadata_base)
        this_meta["component"] = key
        xarr.attrs["metadata"] = json.dumps(this_meta, indent=2, default=_json_safe)
        return xarr

    artifacts: dict[str, Artifact]
    if component == "all":
        artifacts = {}
        for key, suffix in (
            ("aperiodic", ".aperiodic.nc"),
            ("periodic", ".periodic.nc"),
            ("residual", ".residual.nc"),
        ):
            xr_component = attach_metadata(build_component(key), key)
            artifacts[suffix] = Artifact(
                item=xr_component,
                writer=lambda path, arr=xr_component: arr.to_netcdf(
                    path, engine="netcdf4", format="NETCDF4"
                ),
            )
    else:
        xr_component = attach_metadata(build_component(component), component)
        artifacts = {
            ".nc": Artifact(
                item=xr_component,
                writer=lambda path: xr_component.to_netcdf(
                    path, engine="netcdf4", format="NETCDF4"
                ),
            )
        }

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


@register_feature
def band_ratios(
    bandpower_like: FeatureResult | xr.DataArray | str | os.PathLike[str],
    *,
    freqband_dim: str = "freqbands",
    combinations: Sequence[tuple[str, str]] | None = None,
    eps: float | None = None,
) -> FeatureResult:
    """Compute ordered band power ratios from an ``xarray`` bandpower artifact.

    Parameters
    ----------
    bandpower_like : FeatureResult | xarray.DataArray | path-like
        Output from :func:`bandpower` or a compatible ``xarray`` artifact that
        exposes a ``freqbands`` dimension.
    freqband_dim : str, optional
        Name of the dimension containing band labels. Defaults to ``"freqbands"``.
    combinations : sequence of tuple[str, str], optional
        Explicit ordered band pairs ``(numerator, denominator)``. If omitted,
        all permutations of length 2 across the available band labels are used.
    eps : float, optional
        Minimum absolute denominator value. Values with ``|denominator| <= eps``
        yield ``NaN`` to avoid unstable ratios. Defaults to machine epsilon for
        the bandpower dtype.

    Returns
    -------
    FeatureResult
        ``.nc`` artifact with ``freqband_dim`` replaced by ``freqbandPairs``.
    """

    band_da = _resolve_psd_dataarray(bandpower_like)

    if freqband_dim not in band_da.dims:
        raise ValueError(
            f"Frequency band dimension '{freqband_dim}' not present in input dims: {band_da.dims}"
        )

    labels = band_da.coords.get(freqband_dim)
    if labels is None:
        raise ValueError(f"Band dimension '{freqband_dim}' must have coordinate labels.")

    band_labels = [str(label) for label in labels.values.tolist()]
    if combinations is None:
        band_pairs = list(permutations(band_labels, 2))
    else:
        band_pairs = [(str(top), str(bottom)) for top, bottom in combinations]

    if not band_pairs:
        raise ValueError("At least one band pair must be provided.")

    freqband_axis = band_da.get_axis_num(freqband_dim)

    if eps is None:
        try:
            eps = float(np.finfo(np.asarray(0.0, dtype=band_da.dtype).dtype).eps)
        except (TypeError, ValueError):
            eps = float(np.finfo(np.float64).eps)
    else:
        eps = float(eps)

    ratio_arrays: list[xr.DataArray] = []
    tops: list[str] = []
    bottoms: list[str] = []
    pair_labels: list[str] = []

    for top, bottom in band_pairs:
        numerator = band_da.sel({freqband_dim: top})
        denominator = band_da.sel({freqband_dim: bottom})

        with np.errstate(divide="ignore", invalid="ignore"):
            ratio = numerator / denominator

        small_denom = np.abs(denominator) <= eps
        ratio = xr.where(small_denom, np.nan, ratio)

        ratio = ratio.expand_dims({"freqbandPairs": [f"{top}/{bottom}"]}, axis=freqband_axis)
        ratio_arrays.append(ratio)
        tops.append(top)
        bottoms.append(bottom)
        pair_labels.append(f"{top}/{bottom}")

    ratio_xr = xr.concat(ratio_arrays, dim="freqbandPairs")

    original_dims = list(band_da.dims)
    target_dims = ["freqbandPairs" if dim == freqband_dim else dim for dim in original_dims]
    ratio_xr = ratio_xr.transpose(*target_dims)

    ratio_xr = ratio_xr.assign_coords(freqbandPairs=pair_labels)
    ratio_xr.coords["freqband_top"] = ("freqbandPairs", tops)
    ratio_xr.coords["freqband_bottom"] = ("freqbandPairs", bottoms)

    metadata: dict[str, Any] = {
        "pairs": [
            {"label": pair_label, "top": top, "bottom": bottom}
            for pair_label, top, bottom in zip(pair_labels, tops, bottoms, strict=True)
        ],
        "freqband_dim": freqband_dim,
        "eps": eps,
        "input_dims": list(band_da.dims),
        "input_shape": [int(band_da.sizes[dim]) for dim in band_da.dims],
        "output_dims": list(ratio_xr.dims),
        "output_shape": [int(ratio_xr.sizes[dim]) for dim in ratio_xr.dims],
    }

    source_metadata = band_da.attrs.get("metadata")
    if source_metadata is not None:
        metadata["source_metadata"] = source_metadata

    ratio_xr.attrs["metadata"] = json.dumps(metadata, indent=2, default=_json_safe)

    artifacts = {
        ".nc": Artifact(item=ratio_xr, writer=lambda path: ratio_xr.to_netcdf(path)),
    }

    return FeatureResult(artifacts=artifacts)
