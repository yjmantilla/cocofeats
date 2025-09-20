import json
import os
from typing import Any

import mne
import xarray as xr

from cocofeats.definitions import Artifact, FeatureResult
from cocofeats.loaders import load_meeg
from cocofeats.loggers import get_logger

from . import register_feature

log = get_logger(__name__)


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
            data=spectra.get_data(picks="eeg"),
            dims=["spaces", "frequencies"],
            coords={"spaces": spectra.ch_names, "frequencies": spectra.freqs},
        )
    elif isinstance(meeg, mne.BaseEpochs):
        this_xarray = xr.DataArray(
            data=spectra.get_data(picks="eeg"),
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
