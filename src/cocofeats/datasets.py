# Inspired from:
# https://github.com/yjmantilla/sovabids/blob/main/tests/test_bids.py
# https://github.com/yjmantilla/sovabids/blob/main/sovabids/datasets.py


from __future__ import annotations

import os
import re
import shutil
import tempfile
from pathlib import Path

import mne
import numpy as np
from mne_bids.write import _write_raw_brainvision

from cocofeats.utils import get_num_digits

PathLike = str | os.PathLike


def replace_brainvision_filename(fpath: PathLike, newname: str) -> None:
    """
    Replace the BrainVision file references (``DataFile`` and ``MarkerFile``) in a ``.vhdr`` header.

    This updates the entries in the ``[Common Infos]`` section so they point to
    ``<newname>.eeg`` and ``<newname>.vmrk``. Any directory components in
    ``newname`` are ignored and extensions (``.eeg`` or ``.vmrk``) are stripped.

    Parameters
    ----------
    fpath : path-like
        Path to the BrainVision header file (``.vhdr``).
    newname : str
        Base name to set for the data and marker files. If it includes
        an extension (``.eeg`` or ``.vmrk``), it will be removed.

    Returns
    -------
    None

    Notes
    -----
    - The function edits only the ``[Common Infos]`` section if present; if those
      keys aren't found there, it falls back to replacing any top-level
      ``DataFile=...`` / ``MarkerFile=...`` lines it finds.
    - Writing is done atomically via a temporary file in the same directory.
    - The function tries to decode with UTF-8 first, then falls back to Latin-1,
      which is commonly used by BrainVision headers.

    Examples
    --------
    >>> replace_brainvision_filename("recording.vhdr", "session01")
    >>> replace_brainvision_filename("recording.vhdr", "session01.eeg")  # extension stripped
    """
    path = Path(fpath)
    if not path.exists():
        raise FileNotFoundError(f"No such file: {path}")

    # Normalize newname: drop directories and strip .eeg/.vmrk (case-insensitive)
    base = os.path.basename(newname)
    base = re.sub(r"\.(eeg|vmrk)$", "", base, flags=re.IGNORECASE)

    # Read bytes; decode with UTF-8, fallback to Latin-1
    raw = path.read_bytes()
    for enc in ("utf-8", "latin-1"):
        try:
            text = raw.decode(enc)
            encoding = enc
            break
        except UnicodeDecodeError:
            continue
    else:
        # Last-resort: replace errors (keeps file usable)
        text = raw.decode("utf-8", errors="replace")
        encoding = "utf-8"

    # Keep original line endings by splitting with keepends=True
    lines = text.splitlines(keepends=True)

    # Regex helpers
    section_re = re.compile(r"^\s*\[(?P<name>.+?)\]\s*$")
    datafile_re = re.compile(r"^\s*DataFile\s*=.*$", flags=re.IGNORECASE)
    marker_re = re.compile(r"^\s*MarkerFile\s*=.*$", flags=re.IGNORECASE)
    lineend_re = re.compile(r"(\r\n|\r|\n)$")

    def _ending(s: str) -> str:
        m = lineend_re.search(s)
        return m.group(1) if m else ""

    def _set_datafile(end: str) -> str:
        return f"DataFile={base}.eeg{end}"

    def _set_markerfile(end: str) -> str:
        return f"MarkerFile={base}.vmrk{end}"

    # First pass: replace within [Common Infos] if present
    inside_common = False
    changed = False
    for i, line in enumerate(lines):
        m = section_re.match(line)
        if m:
            inside_common = m.group("name").strip().lower() == "common infos"
            continue

        if inside_common and datafile_re.match(line):
            lines[i] = _set_datafile(_ending(line))
            changed = True
            continue
        if inside_common and marker_re.match(line):
            lines[i] = _set_markerfile(_ending(line))
            changed = True
            continue

    # Fallback: if nothing changed, replace any top-level occurrences
    if not changed:
        for i, line in enumerate(lines):
            if datafile_re.match(line):
                lines[i] = _set_datafile(_ending(line))
                changed = True
            elif marker_re.match(line):
                lines[i] = _set_markerfile(_ending(line))
                changed = True

    # Write back atomically
    tmp_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", delete=False, dir=str(path.parent), encoding=encoding, newline=""
        ) as tf:
            tmp_path = Path(tf.name)
            tf.writelines(lines)
        tmp_path.replace(path)
    finally:
        # Clean up if something went wrong before replace
        if tmp_path is not None and tmp_path.exists():
            try:
                tmp_path.unlink()
            except Exception:
                pass


def make_dummy_dataset(
    EXAMPLE,
    PATTERN="T%task%/S%session%/sub%subject%_%acquisition%_%run%",
    DATASET="DUMMY",
    NSUBS=2,
    NSESSIONS=2,
    NTASKS=2,
    NACQS=2,
    NRUNS=2,
    PREFIXES=None,
    ROOT=None,
):
    """Create a dummy dataset given some parameters.

    Parameters
    ----------
    EXAMPLE : str,PathLike|list , required
        Path of the file to replicate as each file in the dummy dataset.
        If a list, it is assumed each item is a file. All of these items are replicated.
    PATTERN : str, optional
        The pattern in placeholder notation using the following fields:
        %dataset%, %task%, %session%, %subject%, %run%, %acquisition%
    DATASET : str, optional
        Name of the dataset.
    NSUBS : int, optional
        Number of subjects.
    NSESSIONS : int, optional
        Number of sessions.
    NTASKS : int, optional
        Number of tasks.
    NACQS : int, optional
        Number of acquisitions.
    NRUNS : int, optional
        Number of runs.
    PREFIXES : dict, optional
        Dictionary with the following keys:'subject', 'session', 'task' and 'acquisition'.
        The values are the corresponding prefix. RUN is not present because it has to be a number.
    ROOT : str, optional
        Path where the files will be generated.
        If None, the _data subdir will be used.

    """

    if PREFIXES is None:
        PREFIXES = {
            "subject": "SU",
            "session": "SE",
            "task": "TA",
            "acquisition": "AC",
            "run": "RU",
        }
    if ROOT is None:
        this_dir = os.path.dirname(__file__)
        data_dir = os.path.abspath(os.path.join(this_dir, "..", "_data"))
    else:
        data_dir = ROOT
    os.makedirs(data_dir, exist_ok=True)

    sub_zeros = get_num_digits(NSUBS)
    subs = [PREFIXES["subject"] + str(x).zfill(sub_zeros) for x in range(NSUBS)]

    task_zeros = get_num_digits(NTASKS)
    tasks = [PREFIXES["task"] + str(x).zfill(task_zeros) for x in range(NTASKS)]

    run_zeros = get_num_digits(NRUNS)
    runs = [str(x).zfill(run_zeros) for x in range(NRUNS)]

    ses_zeros = get_num_digits(NSESSIONS)
    sessions = [PREFIXES["session"] + str(x).zfill(ses_zeros) for x in range(NSESSIONS)]

    acq_zeros = get_num_digits(NACQS)
    acquisitions = [PREFIXES["acquisition"] + str(x).zfill(acq_zeros) for x in range(NACQS)]

    for task in tasks:
        for session in sessions:
            for run in runs:
                for sub in subs:
                    for acq in acquisitions:
                        dummy = PATTERN.replace("%dataset%", DATASET)
                        dummy = dummy.replace("%task%", task)
                        dummy = dummy.replace("%session%", session)
                        dummy = dummy.replace("%subject%", sub)
                        dummy = dummy.replace("%run%", run)
                        dummy = dummy.replace("%acquisition%", acq)
                        path = [data_dir, *dummy.split("/")]
                        fpath = os.path.join(*path)
                        dirpath = os.path.join(*path[:-1])
                        os.makedirs(dirpath, exist_ok=True)
                        if isinstance(EXAMPLE, list):
                            for ff in EXAMPLE:
                                fname, ext = os.path.splitext(ff)
                                shutil.copyfile(ff, fpath + ext)
                                if "vmrk" in ext or "vhdr" in ext:
                                    replace_brainvision_filename(fpath + ext, path[-1])
                        else:
                            fname, ext = os.path.splitext(EXAMPLE)
                            shutil.copyfile(EXAMPLE, fpath + ext)


def generate_1_over_f_noise(n_channels, n_times, exponent=1.0, random_state=None):
    """Generate 1/f noise (pink noise) for MNE Raw data.
    Parameters
    ----------
    n_channels : int
        Number of channels to generate noise for.
    n_times : int
        Number of time points for each channel.
    exponent : float, optional
        Exponent for the 1/f noise. Default is 1.0.
    random_state : int, optional
        Random seed for reproducibility. Default is None.
    Returns
    -------
    np.ndarray
        Generated pink noise with shape (n_channels, n_times).
    """
    rng = np.random.default_rng(random_state)
    noise = np.zeros((n_channels, n_times))

    freqs = np.fft.rfftfreq(n_times, d=1.0)  # d=1.0 assumes unit sampling rate
    freqs[0] = freqs[1]  # avoid division by zero at DC

    scale = 1.0 / np.power(freqs, exponent)

    for ch in range(n_channels):
        # Generate white noise in time domain
        white = rng.standard_normal(n_times)
        # Transform to frequency domain
        white_fft = np.fft.rfft(white)
        # Apply 1/f scaling
        pink_fft = white_fft * scale
        # Transform back to time domain
        pink = np.fft.irfft(pink_fft, n=n_times)
        # Normalize to zero mean, unit variance
        pink = (pink - pink.mean()) / pink.std()
        noise[ch, :] = pink

    return noise


def get_dummy_raw(
    NCHANNELS=5,
    SFREQ=200,
    STOP=10,
    NUMEVENTS=10,
):
    """
    Create a dummy MNE Raw file given some parameters.

    Parameters
    ----------
    NCHANNELS : int, optional
        Number of channels.
    SFREQ : float, optional
        Sampling frequency of the data.
    STOP : float, optional
        Time duration of the data in seconds.
    NUMEVENTS : int, optional
        Number of events along the duration.
    Returns
    -------
    raw : mne.io.Raw
        The generated MNE Raw object.
    new_events : mne.events
        The generated MNE events.
    """
    # Create some dummy metadata
    n_channels = NCHANNELS
    sampling_freq = SFREQ  # in Hertz
    info = mne.create_info(n_channels, sfreq=sampling_freq)

    times = np.linspace(0, STOP, STOP * sampling_freq, endpoint=False)
    data = generate_1_over_f_noise(NCHANNELS, times.shape[0], exponent=1.0)
    # np.zeros((NCHANNELS,times.shape[0]))

    raw = mne.io.RawArray(data, info)
    raw.set_channel_types(dict.fromkeys(raw.ch_names, "eeg"))
    new_events = mne.make_fixed_length_events(raw, duration=STOP // NUMEVENTS)

    return raw, new_events


def save_dummy_vhdr(fpath, dummy_args=None):
    """
    Save a dummy vhdr file.

    Parameters
    ----------
    fpath : str, required
        Path where to save the file.
    kwargs : dict, optional
        Dictionary with the arguments of the get_dummy_raw function.

    Returns
    -------
    List with the Paths of the desired vhdr file, if those were succesfully created,
    None otherwise.
    """
    if dummy_args is None:
        dummy_args = {}

    raw, new_events = get_dummy_raw(**dummy_args)
    _write_raw_brainvision(raw, fpath, new_events, overwrite=True)
    eegpath = fpath.replace(".vhdr", ".eeg")
    vmrkpath = fpath.replace(".vhdr", ".vmrk")
    if all(os.path.isfile(x) for x in [fpath, eegpath, vmrkpath]):
        return [fpath, eegpath, vmrkpath]
    else:
        return None


def generate_dummy_dataset(data_params=None):
    """
    Generate a dummy dataset.

    Parameters
    ----------
    data_params : dict, optional
        Parameters for dataset generation (e.g., number of subjects, sessions, tasks,
        acquisitions, runs, prefixes, and root). See
        :func:`sovabids.datasets.make_dummy_dataset` for the full argument set.

    Returns
    -------
    Any
        Implementation-specific return value (e.g., list of created paths), or ``None``
        if nothing was created.
    """

    if data_params is None:
        DEF_DATASET_PARAMS = {
            "PATTERN": "T%task%/S%session%/sub%subject%_%acquisition%_%run%",
            "DATASET": "DUMMY",
            "NSUBS": 2,
            "NTASKS": 2,
            "NRUNS": 1,
            "NSESSIONS": 1,
            "NACQS": 1,
        }
        data_params = DEF_DATASET_PARAMS
    # Getting current file path and then going to _data directory
    this_dir = os.path.dirname(__file__)
    data_dir = os.path.join(this_dir, "..", "..", "_data")
    data_dir = os.path.abspath(data_dir)

    # Defining relevant conversion paths
    dataset_name = data_params.get("DATASET", "DUMMY")
    test_root = os.path.join(data_dir, dataset_name)
    input_root = os.path.join(test_root, dataset_name + "_SOURCE")
    bids_path = os.path.join(test_root, dataset_name + "_BIDS")

    # Make example File
    example_fpath = save_dummy_vhdr(os.path.join(data_dir, "dummy.vhdr"))

    # PARAMS for making the dummy dataset
    DATA_PARAMS = {"EXAMPLE": example_fpath, "ROOT": input_root}
    DATA_PARAMS.update(data_params)

    # Preparing directories
    dirs = [input_root, bids_path]
    for dir in dirs:
        if os.path.isdir(dir):
            shutil.rmtree(dir)

    [os.makedirs(dir, exist_ok=True) for dir in dirs]

    # Generating the dummy dataset
    make_dummy_dataset(**DATA_PARAMS)
