"""Custom nodes for this pipeline (referenced via `new_definitions:` in pipeline.yml)."""

import json
import os

import matplotlib

matplotlib.use("Agg", force=False)  # headless-safe figure rendering (clusters)

import mne

from neurodags.definitions import Artifact, NodeResult
from neurodags.loaders import load_meeg
from neurodags.nodes import register_node


@register_node
def preprocess_bandpass_report(
    mne_object,
    l_freq=0.1,
    h_freq=75.0,
    resample=256,
    epoch_duration=2.0,
    epoch_overlap=0.0,
):
    """Band-pass -> fixed-length epochs -> resample, plus an MNE Report with the
    band-pass spectrum before vs after, averaged across channels.

    Order matches ``basic_preprocessing``: the high-pass runs on the continuous
    recording (a short epoch can't support 0.1 Hz), then epoch, then resample.
    The before/after PSDs are computed on the continuous data (pre- and post-
    filter) so the band-pass roll-off is clearly visible.

    Returns two artifacts: ``.fif`` (the epochs) and ``.report.html``.
    """
    # --- resolve the input to a Raw ---------------------------------------
    if isinstance(mne_object, NodeResult):
        if ".fif" not in mne_object.artifacts:
            raise ValueError("NodeResult does not contain a .fif artifact to process.")
        mne_object = mne_object.artifacts[".fif"].item
    if isinstance(mne_object, str | os.PathLike):
        mne_object = load_meeg(mne_object)
    raw = mne_object.copy().load_data()

    report = mne.Report(title="Band-pass Preprocessing", verbose="error")

    # --- PSD before band-pass (averaged across channels) ------------------
    report.add_figure(
        raw.compute_psd().plot(average=True, show=False),
        title="PSD before band-pass (avg across channels)",
    )

    # --- band-pass filter -------------------------------------------------
    raw.filter(l_freq=l_freq, h_freq=h_freq, verbose=False)

    # --- PSD after band-pass (averaged across channels) -------------------
    report.add_figure(
        raw.compute_psd().plot(average=True, show=False),
        title=f"PSD after band-pass {l_freq}-{h_freq} Hz (avg across channels)",
    )

    # --- fixed-length epochs, then resample -------------------------------
    epochs = mne.make_fixed_length_epochs(
        raw, duration=epoch_duration, overlap=epoch_overlap, preload=True
    )
    if resample:
        epochs.resample(resample, verbose=False)

    params = {
        "l_freq": l_freq,
        "h_freq": h_freq,
        "resample": resample,
        "epoch_duration": epoch_duration,
        "epoch_overlap": epoch_overlap,
    }
    report.add_html(f"<pre>{json.dumps(params, indent=2)}</pre>", title="Parameters")

    return NodeResult(
        artifacts={
            ".fif": Artifact(item=epochs, writer=lambda p: epochs.save(p, overwrite=True)),
            ".report.html": Artifact(
                item=report, writer=lambda p: report.save(p, overwrite=True, open_browser=False)
            ),
        }
    )
