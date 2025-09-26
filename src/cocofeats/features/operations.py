import json
import os

import mne

from cocofeats.definitions import Artifact, FeatureResult, PathLike
from cocofeats.loggers import get_logger
from cocofeats.writers import save_dict_to_json

from cocofeats.features import register_feature
import xarray as xr
import numpy as np


log = get_logger(__name__)


@register_feature(name="mean_across_dimension", override=True)
def mean_across_dimension(xarray_data, dim):
    """
    Compute the mean across a specified dimension of an xarray DataArray.

    Parameters
    ----------
    xarray_data : xarray.DataArray
        The input xarray DataArray.
    dim : str
        The dimension name to compute the mean over.

    Returns
    -------
    FeatureResult
        A feature result containing the mean as a netcdf4 artifact.
    """
    import xarray as xr
    import numpy as np

    if not isinstance(xarray_data, xr.DataArray):
        raise ValueError("Input must be an xarray DataArray.")

    mean_data = xarray_data.mean(dim=dim)

    # return the new xarray in ncdf4 format
    artifacts = {".nc": Artifact(item=mean_data, writer=lambda path: mean_data.to_netcdf(path))}
    return FeatureResult(artifacts=artifacts)

def slice_xarray(xarray_data, dim, start=None, end=None):
    """
    Slice an xarray DataArray along a specified dimension.

    Parameters
    ----------
    xarray_data : xarray.DataArray
        The input xarray DataArray.
    dim : str
        The dimension name to slice.
    start : int or None
        The starting index for the slice (inclusive). If None, starts from the beginning.
    end : int or None
        The ending index for the slice (exclusive). If None, goes to the end.

    Returns
    -------
    FeatureResult
        A feature result containing the sliced data as a netcdf4 artifact.
    """

    if isinstance(xarray_data, (str, os.PathLike)):
        xarray_data = xr.open_dataarray(xarray_data)
        log.debug("Loaded xarray DataArray from file", input=xarray_data)

    if isinstance(xarray_data, FeatureResult):
        if ".nc" in xarray_data.artifacts:
            xarray_data = xarray_data.artifacts[".nc"].item
        else:
            raise ValueError("FeatureResult does not contain a .nc artifact to process.")

    if not isinstance(xarray_data, xr.DataArray):
        raise ValueError("Input must be an xarray DataArray.")

    slicer = {dim: slice(start, end)}
    sliced_data = xarray_data.isel(**slicer)

    # return the new xarray in ncdf4 format
    artifacts = {".nc": Artifact(item=sliced_data, writer=lambda path: sliced_data.to_netcdf(path))}
    return FeatureResult(artifacts=artifacts)

@register_feature(name="aggregate_across_dimension", override=True)
def aggregate_across_dimension(xarray_data, dim, operation='mean', args=None):
    """
    Aggregate data across a specified dimension of an xarray DataArray using a given operation.

    Parameters
    ----------
    xarray_data : xarray.DataArray
        The input xarray DataArray.
    dim : str
        The dimension name to aggregate over.
    operation : str
        The aggregation operation to perform ('mean', 'sum', 'max', 'min', etc.).
    args : dict, optional
        Additional arguments to pass to the aggregation function.

    Returns
    -------
    FeatureResult
        A feature result containing the aggregated data as a netcdf4 artifact.
    """

    if isinstance(xarray_data, (str, os.PathLike)):
        xarray_data = xr.open_dataarray(xarray_data)
        log.debug("Loaded xarray DataArray from file", input=xarray_data)

    if isinstance(xarray_data, FeatureResult):
        if ".nc" in xarray_data.artifacts:
            xarray_data = xarray_data.artifacts[".nc"].item
        else:
            raise ValueError("FeatureResult does not contain a .nc artifact to process.")

    if not isinstance(xarray_data, xr.DataArray):
        raise ValueError("Input must be an xarray DataArray.")

    if args is None:
        args = {}

    if not hasattr(xarray_data, operation):
        raise ValueError(f"Operation '{operation}' is not valid for xarray DataArray.")

    agg_func = getattr(xarray_data, operation)
    aggregated_data = agg_func(dim=dim, **args)

    # return the new xarray in ncdf4 format
    artifacts = {".nc": Artifact(item=aggregated_data, writer=lambda path: aggregated_data.to_netcdf(path))}
    return FeatureResult(artifacts=artifacts)

if __name__ == "__main__":
    import numpy as np
    # Example usage
    # Example usage
    data = xr.DataArray(np.random.rand(4, 3, 2), dims=("time", "channel", "frequency"), coords={
        "time": np.arange(4),
        "channel": ["Cz", "Pz", "Fz"],
        "frequency": [10, 20]
    })
    result = mean_across_dimension(data, dim="time")
    print(result)

    # test aggregate
    result_agg = aggregate_across_dimension(data, dim="channel", operation='sum')
    print(result_agg)

    # test slice
    result_slice = slice_xarray(data, dim="time", start=1, end=3)
    print(result_slice)
    print(result_slice.artifacts[".nc"].item)  # Access the sliced xarray
