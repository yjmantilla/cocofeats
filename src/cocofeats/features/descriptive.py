from matplotlib.pylab import fmax
from cocofeats.loggers import get_logger
from cocofeats.utils import get_path,replace_bids_suffix
from cocofeats.loaders import load_meeg
from cocofeats.definitions import DatasetConfig
import glob
import mne
import os
from cocofeats.definitions import PathLike
from cocofeats.features.base import FeatureBase
from cocofeats.definitions import Artifact, FeatureResult
from typing import Any
import xarray as xr

log = get_logger(__name__)

def spectrum(
    meeg: mne.io.BaseRaw | mne.BaseEpochs,
    kwargs: dict[str, Any] | None = None) -> FeatureResult:
    """
    Compute the power spectral density of M/EEG data.

    Parameters
    ----------
    meeg : mne.io.BaseRaw or mne.BaseEpochs
        The M/EEG data to analyze. Can be raw data or epochs.
    Returns
    -------
    dict
        A dictionary containing the power spectral density results, metadata, and artifacts (MNE Report).
    """


    if isinstance(meeg, (str, os.PathLike)):
        meeg = load_meeg(meeg)
        log.debug("MNEReport: loaded MNE object from file", input=meeg)

    if kwargs is None:
        kwargs = {}

    spectra = meeg.compute_psd(**kwargs)
    log.debug("MNEReport: computed spectra", spectra=spectra)

    report = mne.Report(title=f'Spectrum', verbose='error')
    report.add_figure(spectra.plot(show=False), title=f'Spectrum')
    log.debug("MNEReport: computed report")

    this_artifact = Artifact(item=report, writer=lambda path: report.save(path, overwrite=True, open_browser=False))
    if isinstance(meeg, mne.io.BaseRaw):
        this_xarray = xr.DataArray(
            data=spectra.get_data(picks='eeg'),
            dims=['spaces', 'frequencies'],
            coords={
                'spaces': spectra.ch_names,
                'frequencies': spectra.freqs
            },
        )
    elif isinstance(meeg, mne.BaseEpochs):
        this_xarray = xr.DataArray(
            data=spectra.get_data(picks='eeg'),
            dims=['epochs', 'spaces', 'frequencies'],
            coords={
                'epochs': list(range(len(spectra))),
                'spaces': spectra.ch_names,
                'frequencies': spectra.freqs,
            },
        )

    this_metadata = {'feature': 'MNEReport', 'fmax': fmax}
    out = FeatureResult(
        data=this_xarray,
        artifacts={'report.html': this_artifact}, # has to include the extension for saving
        metadata=this_metadata)

    return out


class MNEReport(FeatureBase):
    """
    Class to generate MNE reports for M/EEG data files.
    """

    @classmethod
    def compute(cls, 
        input: PathLike | mne.io.BaseRaw | mne.BaseEpochs,
        reference_path: PathLike | None = None,
        suffix: str = None,
        save: bool = True,
        overwrite: bool = False,
        inspection_artifact: bool = True,
        context: dict[str, Any] | None = None, # to pass additional context if needed
        args: dict[str, Any] | None = None, # to pass additional arguments to internal functions if needed
        ):
        if suffix is None:
            suffix = cls.__name__
        output_path = None
        if reference_path is not None:
            # Replace _suffix.ext with _suffix.classname.ext
            output_path = replace_bids_suffix(reference_path, suffix, f'.{cls.__name__}.html')

        if os.path.exists(output_path) and not overwrite:
            log.info("FeatureBase: output file already exists and overwrite is False, skipping computation", path=output_path)
            return output_path

        report = mne.Report(title=f'Inspect {reference_path}', verbose='error')

        if isinstance(input, (str, os.PathLike)):
            input = load_meeg(input)
            log.debug("MNEReport: loaded MNE object from file", input=input)
        if isinstance(input, mne.io.BaseRaw):
            report.add_raw(input, title=f'Raw Data - {reference_path}')
        elif isinstance(input, mne.BaseEpochs):
            report.add_epochs(input, title=f'Epochs - {reference_path}')

        report.add_figure(input.plot_psd(show=False), title=f'{reference_path} Spectrum')
        fmax = input.info['sfreq'] / 2
        if fmax > 200:
            fmax = 200
            report.add_figure(input.plot_psd(show=False, fmax=fmax), title=f'{reference_path} Spectrum Below 200Hz')

        log.debug("MNEReport: computed report")

        if inspection_artifact or save:
            # As the report is both the inspection and the feature artifact, we save it if either is requested
            if reference_path is None:
                raise ValueError("reference_path must be provided if inspection_artifact or save is True")
            report.save(output_path, overwrite=True, open_browser=False)
            log.info("MNEReport: saved inspection report", path=output_path)
        return report

    @staticmethod
    def load(path: PathLike):
        # read the html file and return as string
        with open(path, 'r', encoding='utf-8') as f:
            html_content = f.read()
        log.info("MNEReport: loaded report as HTML content", path=path)
        return html_content