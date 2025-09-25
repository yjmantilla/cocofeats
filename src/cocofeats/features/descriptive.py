import json
import os
import mne

from cocofeats.definitions import Artifact, FeatureResult
from cocofeats.loaders import load_meeg
from cocofeats.loggers import get_logger
from . import register_feature

log = get_logger(__name__)


import json
import numpy as np

def _json_safe(obj):
    """Recursively convert numpy types to native Python types."""
    if isinstance(obj, (np.integer,)):
        return int(obj)
    elif isinstance(obj, (np.floating,)):
        return float(obj)
    elif isinstance(obj, (np.ndarray,)):
        return obj.tolist()
    elif isinstance(obj, (list, tuple)):
        return [_json_safe(x) for x in obj]
    elif isinstance(obj, dict):
        return {k: _json_safe(v) for k, v in obj.items()}
    else:
        return obj

@register_feature
def extract_meeg_metadata(mne_object) -> FeatureResult:
    """
    Extract metadata from an MNE object (Raw or Epochs) and save as JSON.

    Parameters
    ----------
    mne_object : str | os.PathLike | mne.io.Raw | mne.Epochs
        Path to a MEEG file or an already loaded MNE object.

    Returns
    -------
    FeatureResult
        A feature result containing a JSON artifact with metadata.
    """

    if isinstance(mne_object, FeatureResult):
        if ".fif" in mne_object.artifacts:
            mne_object = mne_object.artifacts[".fif"].item
        else:
            raise ValueError("FeatureResult does not contain a .fif artifact to process.")

    if isinstance(mne_object, (str, os.PathLike)):
        mne_object = load_meeg(mne_object)
        log.debug("Loaded MNE object from file", input=mne_object)

    info_dict = {}

    # Shared info
    info_dict["sfreq"] = float(mne_object.info.get("sfreq", None))
    info_dict["n_channels"] = mne_object.info.get("nchan", None)
    info_dict["ch_names"] = list(mne_object.info.get("ch_names", []))

    # Channel types
    try:
        info_dict["ch_types"] = mne_object.get_channel_types()
    except Exception:
        info_dict["ch_types"] = None

    # Dimensions like xarray
    dims = []
    coords = {}
    shape = ()

    if isinstance(mne_object, mne.io.BaseRaw):
        dims = ["time", "channels"]
        n_times = mne_object.n_times
        shape = (n_times, len(info_dict["ch_names"]))
        coords["time"] = {'start': float(mne_object.times[0]), 'stop': float(mne_object.times[-1]), 'n_times': n_times, 'delta': float(mne_object.times[1] - mne_object.times[0])}
        coords["channels"] = info_dict["ch_names"]

    elif isinstance(mne_object, mne.Epochs):
        dims = ["epochs", "time", "channels"]
        n_epochs, n_channels, n_times = mne_object.get_data().shape
        shape = (n_epochs, n_times, n_channels)
        coords["epochs"] = list(range(n_epochs))
        coords["time"] = {'start': float(mne_object.times[0]), 'stop': float(mne_object.times[-1]), 'n_times': n_times, 'delta': float(mne_object.times[1] - mne_object.times[0])}
        coords["channels"] = info_dict["ch_names"]

        # Add event info if available
        if hasattr(mne_object, "events"):
            info_dict["events_shape"] = (
                None if mne_object.events is None else mne_object.events.shape
            )

    info_dict["shape"] = shape
    info_dict["dims"] = dims
    info_dict["coords"] = coords

    # Optional: add annotations if present
    if hasattr(mne_object, "annotations") and len(mne_object.annotations) > 0:
        info_dict["annotations"] = [
            {"onset": float(ann["onset"]), "duration": float(ann["duration"]), "description": ann["description"]}
            for ann in mne_object.annotations
        ]

    # Prepare artifact: human-readable JSON
    def _writer(path):
        with open(path, "w", encoding="utf-8") as f:
            json.dump(_json_safe(info_dict), f, indent=2, ensure_ascii=False)

    artifacts = {".json": Artifact(item=info_dict, writer=_writer)}

    # also present a mne.report with the info?

    return FeatureResult(artifacts=artifacts)
