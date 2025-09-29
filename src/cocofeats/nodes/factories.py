"""Factory utilities for building xarray-based node operations.

This module provides a flexible backbone for node implementations that apply
one-dimensional numerical routines across higher dimensional ``xarray``
containers.  The core helper accepts a pure function that consumes a 1D
``numpy`` array (optionally with extra ``args``/``kwargs``) and produces either a
scalar or a 1D sequence.  The helper takes care of:

* coercing various inputs (``NodeResult``, ``xarray`` objects, ``mne`` Raw/Epochs)
  into a consistent ``xarray.DataArray`` representation,
* iterating/vectorising the pure function along a chosen dimension,
* rebuilding an output ``DataArray`` with well-defined coordinates, and
* returning a ``NodeResult`` suitable for the node registry.

The exported ``xarray_factory`` node can be used directly from feature pipeline
definitions, and ``apply_1d`` is available for composing bespoke nodes in
Python code.
"""

from __future__ import annotations

import json
import os
from importlib import import_module
from typing import TYPE_CHECKING, Any, Callable, Mapping, Sequence

try:  # pragma: no cover - optional dependency
    import mne
except ImportError:  # pragma: no cover - optional dependency
    mne = None  # type: ignore[assignment]

import numpy as np
import xarray as xr

from cocofeats.definitions import Artifact, NodeResult
from cocofeats.loggers import get_logger
from cocofeats.nodes import register_node
from cocofeats.utils import _resolve_eval_strings
from cocofeats.writers import _json_safe

log = get_logger(__name__)

CallableLike = Callable[[np.ndarray], Any]

if TYPE_CHECKING:  # pragma: no cover - typing helpers only
    from mne import BaseEpochs as MNEEpochs  # type: ignore[import-not-found]
    from mne.io import BaseRaw as MNERaw  # type: ignore[import-not-found]
else:  # pragma: no cover - runtime fallback when MNE is absent
    MNEEpochs = object
    MNERaw = object

DataLike = (
    NodeResult
    | xr.DataArray
    | xr.Dataset
    | np.ndarray
    | MNERaw
    | MNEEpochs
    | str
    | os.PathLike[str]
)


class _FactoryError(ValueError):
    """Internal helper error for consistent exception types."""


def _resolve_callable(candidate: CallableLike | str) -> CallableLike:
    """Return a callable from a candidate value.

    The ``candidate`` can already be callable, an ``eval%`` expression (handled
    via :func:`_resolve_eval_strings`), or a dotted import path such as
    ``"numpy.mean"``.  Any other input raises ``TypeError``.
    """

    if callable(candidate):
        return candidate

    if isinstance(candidate, str):
        resolved = _resolve_eval_strings(candidate)
        if callable(resolved):
            return resolved

        if isinstance(resolved, str):
            module_path, sep, attr = resolved.rpartition(".")
            if sep == "":
                raise TypeError(
                    "String callables must be a dotted import path (e.g. 'numpy.mean')."
                )
            module = import_module(module_path)
            target = getattr(module, attr)
            if callable(target):
                return target
            raise TypeError(f"Attribute '{attr}' in module '{module_path}' is not callable.")

    raise TypeError("pure_function must be callable or a resolvable string reference.")


def _ensure_dataarray(data_like: DataLike, *, context: str) -> xr.DataArray:
    """Normalise supported inputs to an ``xarray.DataArray``.

    Parameters
    ----------
    data_like
        Supported inputs include ``NodeResult`` instances containing ``.nc``
        artifacts, ``xarray`` datasets/arrays, NumPy arrays, ``mne`` Raw/Epochs
        objects, or filesystem paths pointing to NetCDF files.
    context
        Short description used in error messages.
    """

    if isinstance(data_like, xr.DataArray):
        return data_like

    if isinstance(data_like, NodeResult):
        if ".nc" not in data_like.artifacts:
            raise _FactoryError(
                f"{context}: NodeResult inputs must contain a '.nc' artifact with an xarray payload."
            )
        candidate = data_like.artifacts[".nc"].item
        return _ensure_dataarray(candidate, context=context)

    if isinstance(data_like, xr.Dataset):
        if len(data_like.data_vars) != 1:
            raise _FactoryError(
                f"{context}: xarray.Dataset inputs must expose exactly one data variable."
            )
        return next(iter(data_like.data_vars.values()))

    if isinstance(data_like, np.ndarray):
        # Expose anonymous dimensions (dim_0, dim_1, ...) for bare numpy arrays.
        dims = tuple(f"dim_{idx}" for idx in range(data_like.ndim))
        return xr.DataArray(data_like, dims=dims)

    if isinstance(data_like, (str, os.PathLike)):
        path = os.fspath(data_like)
        log.debug("Loading DataArray from path", path=path)
        return xr.load_dataarray(path)

    if mne is not None and isinstance(data_like, mne.io.BaseRaw):
        data = data_like.get_data()
        coords = {
            "channels": ("channels", list(data_like.ch_names)),
            "time": ("time", data_like.times.copy()),
        }
        arr = xr.DataArray(data, dims=("channels", "time"), coords=coords)
        arr.attrs.setdefault(
            "metadata",
            json.dumps(
                {
                    "source": "mne.Raw",  # lightweight provenance
                    "sfreq": float(data_like.info.get("sfreq", 0.0)),
                    "n_times": int(data.shape[1]),
                    "n_channels": int(data.shape[0]),
                }
            ),
        )
        return arr

    if mne is not None and isinstance(data_like, mne.BaseEpochs):
        data = data_like.get_data()
        coords = {
            "epochs": ("epochs", np.arange(data.shape[0])),
            "channels": ("channels", list(data_like.ch_names)),
            "time": ("time", data_like.times.copy()),
        }
        arr = xr.DataArray(data, dims=("epochs", "channels", "time"), coords=coords)
        arr.attrs.setdefault(
            "metadata",
            json.dumps(
                {
                    "source": "mne.Epochs",
                    "sfreq": float(data_like.info.get("sfreq", 0.0)),
                    "n_epochs": int(data.shape[0]),
                    "n_times": int(data.shape[2]),
                    "n_channels": int(data.shape[1]),
                }
            ),
        )
        return arr

    raise _FactoryError(f"{context}: unsupported input type '{type(data_like).__name__}'.")


def _normalise_result(value: Any, *, dtype: np.dtype | None) -> tuple[np.ndarray, bool]:
    """Convert function outputs into numpy arrays and flag scalars.

    Returns a tuple ``(array, is_scalar)`` where ``array`` is either a 0-D numpy
    array (scalar case) or a 1-D numpy array.
    """

    if isinstance(value, xr.DataArray):
        value = value.values

    if isinstance(value, np.ndarray):
        arr = value
    else:
        arr = np.array(value)

    if arr.ndim > 1:
        raise _FactoryError("Pure functions must return scalars or 1-D sequences.")

    if dtype is not None and arr.dtype != dtype:
        arr = arr.astype(dtype, copy=False)

    return arr, arr.ndim == 0


def _vectorised_apply(
    data_xr: xr.DataArray,
    *,
    dim: str,
    func: CallableLike,
    args: Sequence[Any],
    kwargs: Mapping[str, Any],
    result_dim: str | None,
    result_coords: Sequence[Any] | None,
    output_dtype: np.dtype | None,
) -> tuple[xr.DataArray, dict[str, Any]]:
    """Apply a 1-D function along ``dim`` and return output + metadata."""

    if dim not in data_xr.dims:
        raise _FactoryError(
            f"Requested dimension '{dim}' not found in input dims {tuple(data_xr.dims)}."
        )

    other_dims = [d for d in data_xr.dims if d != dim]
    if data_xr.sizes[dim] == 0:
        raise _FactoryError(f"Input dimension '{dim}' has size zero; cannot build 1-D slices.")
    for axis in other_dims:
        if data_xr.sizes[axis] == 0:
            raise _FactoryError(
                f"Input dimension '{axis}' has size zero; cannot evaluate pure function across slices."
            )

    # Use the first slice to infer output structure and dtype.
    selector = {d: 0 for d in other_dims}
    sample_vector = np.asarray(data_xr.isel(selector).data)
    if sample_vector.ndim != 1:
        sample_vector = sample_vector.reshape(-1)

    first_output = func(sample_vector, *args, **kwargs)
    inferred_dtype = np.dtype(output_dtype) if output_dtype is not None else None
    first_array, is_scalar = _normalise_result(first_output, dtype=inferred_dtype)

    if inferred_dtype is None:
        inferred_dtype = first_array.dtype

    if not is_scalar:
        if first_array.ndim != 1:
            raise _FactoryError("Non-scalar outputs must be one-dimensional sequences.")
        result_dim_name = result_dim or "outputs"
        result_length = int(first_array.shape[0])
        if result_length == 0:
            raise _FactoryError("Pure function returned an empty sequence; cannot build outputs.")
        if result_coords is not None and len(tuple(result_coords)) != result_length:
            raise _FactoryError(
                "Length of result_coords does not match the function output length."
            )
        output_core_dims = [[result_dim_name]]
        output_sizes = {result_dim_name: result_length}
        first_shape = first_array.shape
    else:
        result_dim_name = None
        result_length = None
        output_core_dims = [[]]
        output_sizes = None
        first_shape = ()

    def _wrapped(vector: np.ndarray) -> np.ndarray:
        # ``vector`` arrives as numpy thanks to xarray.apply_ufunc vectorisation.
        flat = np.asarray(vector)
        if flat.ndim != 1:
            flat = flat.reshape(-1)
        output = func(flat, *args, **kwargs)
        result_array, scalar_flag = _normalise_result(output, dtype=inferred_dtype)
        if scalar_flag != is_scalar:
            raise _FactoryError("Pure function returned inconsistent scalar/non-scalar outputs.")
        if not scalar_flag and result_array.shape != first_shape:
            raise _FactoryError("Pure function returned sequences of varying length.")
        return result_array

    apply_kwargs: dict[str, Any] = {
        "input_core_dims": [[dim]],
        "output_core_dims": output_core_dims,
        "output_dtypes": [inferred_dtype],
        "vectorize": True,
        "keep_attrs": True,
        "dask": "parallelized",
    }
    if output_sizes:
        apply_kwargs["output_sizes"] = output_sizes

    result_da = xr.apply_ufunc(_wrapped, data_xr, **apply_kwargs)

    # Reorder dimensions so the replacement dim occupies the original slot.
    input_dims = list(data_xr.dims)
    if result_dim_name is None:
        target_dims = [d for d in input_dims if d != dim]
    else:
        idx = input_dims.index(dim)
        target_dims = input_dims.copy()
        target_dims[idx : idx + 1] = [result_dim_name]
    result_da = result_da.transpose(*target_dims)

    if result_dim_name is not None:
        if result_coords is not None:
            result_da = result_da.assign_coords({result_dim_name: list(result_coords)})
        elif result_dim_name not in result_da.coords:
            result_da = result_da.assign_coords({result_dim_name: np.arange(first_shape[0])})

    metadata = {
        "factory": "xarray_factory",
        "dimension": dim,
        "is_scalar": is_scalar,
        "result_dimension": result_dim_name,
        "result_length": result_length,
        "input_dims": list(data_xr.dims),
        "input_shape": [int(data_xr.sizes[d]) for d in data_xr.dims],
        "output_dims": list(result_da.dims),
        "output_shape": [int(result_da.sizes[d]) for d in result_da.dims],
        "function": getattr(func, "__name__", repr(func)),
        "function_module": getattr(func, "__module__", None),
    }

    return result_da, metadata


def apply_1d(
    data_like: DataLike,
    *,
    dim: str,
    pure_function: CallableLike | str,
    args: Sequence[Any] | None = None,
    kwargs: Mapping[str, Any] | None = None,
    result_dim: str | None = None,
    result_coords: Sequence[Any] | None = None,
    output_dtype: Any | None = None,
    metadata: Mapping[str, Any] | None = None,
    keep_input_metadata: bool = True,
) -> NodeResult:
    """Apply a 1-D pure function across a chosen xarray dimension.

    Parameters
    ----------
    data_like
        Input data. See :func:`_ensure_dataarray` for supported types.
    dim
        Dimension name whose slices are fed to the pure function.
    pure_function
        Callable (or resolvable string) that accepts a one-dimensional numpy
        array and returns either a scalar or a one-dimensional sequence.
    args, kwargs
        Optional positional and keyword arguments forwarded to
        ``pure_function`` for each slice.
    result_dim
        Name of the new dimension when the pure function returns a sequence. If
        omitted, ``"outputs"`` is used.
    result_coords
        Coordinate labels for ``result_dim``. Length must match the sequence
        length returned by the pure function.
    output_dtype
        Optional numpy dtype override for the output array.
    metadata
        Extra metadata merged into the automatically generated metadata block.
    keep_input_metadata
        When ``True`` (default) and the input carries a ``metadata`` attribute,
        it is nested under ``source_metadata`` in the output metadata.
    """

    func = _resolve_callable(pure_function)
    args = tuple(args or ())
    kwargs = dict(kwargs or {})

    data_xr = _ensure_dataarray(data_like, context="apply_1d")
    result_da, auto_metadata = _vectorised_apply(
        data_xr,
        dim=dim,
        func=func,
        args=args,
        kwargs=kwargs,
        result_dim=result_dim,
        result_coords=result_coords,
        output_dtype=np.dtype(output_dtype) if output_dtype is not None else None,
    )

    combined_metadata = dict(auto_metadata)
    if args:
        combined_metadata["function_args"] = list(args)
    if kwargs:
        combined_metadata["function_kwargs"] = {str(k): _json_safe(v) for k, v in kwargs.items()}
    if metadata:
        combined_metadata.update({str(k): _json_safe(v) for k, v in metadata.items()})
    if keep_input_metadata and "metadata" in data_xr.attrs:
        combined_metadata["source_metadata"] = data_xr.attrs["metadata"]

    result_da.attrs["metadata"] = json.dumps(combined_metadata, indent=2, default=_json_safe)

    artifacts = {
        ".nc": Artifact(
            item=result_da,
            writer=lambda path: result_da.to_netcdf(path, engine="netcdf4", format="NETCDF4"),
        )
    }
    return NodeResult(artifacts=artifacts)


@register_node(name="xarray_factory", override=True)
def xarray_factory(
    data_like: DataLike,
    *,
    dim: str,
    pure_function: CallableLike | str,
    args: Sequence[Any] | None = None,
    kwargs: Mapping[str, Any] | None = None,
    result_dim: str | None = None,
    result_coords: Sequence[Any] | None = None,
    output_dtype: Any | None = None,
    metadata: Mapping[str, Any] | None = None,
    keep_input_metadata: bool = True,
) -> NodeResult:
    """Node entry-point delegating to :func:`apply_1d`.

    This thin wrapper enables declarative use from pipeline YAML definitions by
    exposing the factory through the standard node registry.  All parameters are
    forwarded verbatim to :func:`apply_1d`.
    """

    return apply_1d(
        data_like,
        dim=dim,
        pure_function=pure_function,
        args=args,
        kwargs=kwargs,
        result_dim=result_dim,
        result_coords=result_coords,
        output_dtype=output_dtype,
        metadata=metadata,
        keep_input_metadata=keep_input_metadata,
    )


__all__ = ["apply_1d", "xarray_factory"]
