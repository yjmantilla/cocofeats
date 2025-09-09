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
    EXAMPLE: PathLike | list[PathLike],
    PATTERN: str = "T%task%/S%session%/sub%subject%_%acquisition%_%run%",
    DATASET: str = "DUMMY",
    NSUBS: int = 2,
    NSESSIONS: int = 2,
    NTASKS: int = 2,
    NACQS: int = 2,
    NRUNS: int = 2,
    PREFIXES: dict[str, str] | None = None,
    ROOT: PathLike | None = None,
) -> None:
    """
    Create a dummy dataset by replicating an example file (or set of files) into a
    directory layout defined by a pattern of placeholders.

    Parameters
    ----------
    EXAMPLE : path-like or list of path-like
        Path of a file to replicate as each file in the dummy dataset. If a list,
        each item is treated as a file path and all are replicated for every
        combination generated by the pattern.
    PATTERN : str, optional
        Directory and base-filename pattern using placeholders:
        ``%dataset%``, ``%task%``, ``%session%``, ``%subject%``, ``%run%``, ``%acquisition%``.
        Forward slashes (``/``) are used as separators inside the pattern.
    DATASET : str, optional
        Name of the dataset (used to replace ``%dataset%``).
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
        Mapping for prefixes with keys ``"subject"``, ``"session"``, ``"task"``,
        ``"acquisition"``, and ``"run"``. ``"run"`` is numeric in the filename, but the
        prefix is used in directory/file naming within the pattern if present.
        Defaults to ``{"subject": "SU", "session": "SE", "task": "TA", "acquisition": "AC", "run": "RU"}``.
    ROOT : path-like, optional
        Directory where files will be generated. If ``None``, uses a ``_data`` subdirectory
        relative to this module.

    Returns
    -------
    None

    Notes
    -----
    - For BrainVision files, if the example set includes ``.vhdr``/``.vmrk``, the function
      updates their internal ``DataFile``/``MarkerFile`` entries to point to the generated base name.
    - The zero-padding width for subject/session/task/acquisition/run is inferred from the
      corresponding count (e.g., ``NSUBS``) via :func:`get_num_digits`.
    - Indices start at 0 to match the original implementation.

    Examples
    --------
    Create a layout with one example file replicated across combinations:

    >>> make_dummy_dataset("example.dat", NSUBS=2, NRUNS=3)

    Replicate a BrainVision trio (.vhdr/.vmrk/.eeg) for every combination:

    >>> brains = ["template.vhdr", "template.vmrk", "template.eeg"]
    >>> make_dummy_dataset(brains, DATASET="Demo", NSUBS=1, NTASKS=1, NRUNS=2)
    """
    # Defaults
    if PREFIXES is None:
        PREFIXES = {
            "subject": "SU",
            "session": "SE",
            "task": "TA",
            "acquisition": "AC",
            "run": "RU",
        }

    # Resolve output root
    if ROOT is None:
        this_dir = Path(__file__).parent
        data_dir = (this_dir / ".." / "_data").resolve()
    else:
        data_dir = Path(ROOT)
    data_dir.mkdir(parents=True, exist_ok=True)

    # Normalize EXAMPLE -> list[Path]
    examples: list[Path] = (
        [Path(EXAMPLE)] if not isinstance(EXAMPLE, list) else [Path(p) for p in EXAMPLE]
    )
    for ex in examples:
        if not ex.exists():
            raise FileNotFoundError(f"Example file does not exist: {ex}")

    # Build label lists (0-based indices; zero-padded lengths inferred from counts)
    sub_zeros = get_num_digits(NSUBS)
    subs = [f"{PREFIXES['subject']}{str(x).zfill(sub_zeros)}" for x in range(NSUBS)]

    task_zeros = get_num_digits(NTASKS)
    tasks = [f"{PREFIXES['task']}{str(x).zfill(task_zeros)}" for x in range(NTASKS)]

    run_zeros = get_num_digits(NRUNS)
    runs = [str(x).zfill(run_zeros) for x in range(NRUNS)]

    ses_zeros = get_num_digits(NSESSIONS)
    sessions = [f"{PREFIXES['session']}{str(x).zfill(ses_zeros)}" for x in range(NSESSIONS)]

    acq_zeros = get_num_digits(NACQS)
    acquisitions = [f"{PREFIXES['acquisition']}{str(x).zfill(acq_zeros)}" for x in range(NACQS)]

    # Generate files per combination
    for task in tasks:
        for session in sessions:
            for run in runs:
                for sub in subs:
                    for acq in acquisitions:
                        # Fill placeholders
                        dummy = (
                            PATTERN.replace("%dataset%", DATASET)
                            .replace("%task%", task)
                            .replace("%session%", session)
                            .replace("%subject%", sub)
                            .replace("%run%", run)
                            .replace("%acquisition%", acq)
                        )

                        # Resolve output path: pattern may include subdirs; last element is base name
                        parts = dummy.split("/")
                        dirpath = data_dir.joinpath(*parts[:-1])
                        dirpath.mkdir(parents=True, exist_ok=True)
                        base_out = data_dir.joinpath(*parts)  # no extension yet

                        # Copy each example, preserving extension; adjust BrainVision headers if present
                        for ex in examples:
                            ext = ex.suffix  # includes leading dot, keeps original case
                            out_fpath = Path(f"{base_out}{ext}")
                            shutil.copy2(ex, out_fpath)

                            # If copying BrainVision header/marker, update references to new base name
                            lower_ext = ext.lower()
                            if lower_ext in {".vhdr", ".vmrk"}:
                                # pass the base filename (without extension) that the header should reference
                                replace_brainvision_filename(out_fpath, parts[-1])


def generate_1_over_f_noise(
    n_channels: int,
    n_times: int,
    exponent: float = 1.0,
    *,
    sfreq: float = 1.0,
    random_state: int | np.random.Generator | None = None,
) -> np.ndarray:
    """
    Generate 1/f^alpha (pink-like) noise, suitable for synthetic EEG/MEG channels.

    The noise is produced by scaling the real FFT of white noise by ``1 / f**exponent``
    (with the DC bin set to 0), then transforming back to the time domain. Each channel
    is z-scored (zero mean, unit variance).

    Parameters
    ----------
    n_channels : int
        Number of channels to generate.
    n_times : int
        Number of time samples per channel.
    exponent : float, optional
        Spectral exponent :math:`\\alpha` in :math:`1/f^{\\alpha}`. Use 1.0 for
        “pink” noise, 0.0 for white, 2.0 for Brownian-like, etc. Default is 1.0.
    sfreq : float, optional
        Sampling frequency in Hz (used only to compute the frequency axis).
        Default is 1.0 (unit sampling).
    random_state : int | numpy.random.Generator, optional
        Seed or generator for reproducibility. If ``None``, a new Generator is used.

    Returns
    -------
    np.ndarray
        Array of shape ``(n_channels, n_times)`` containing 1/f^alpha noise,
        z-scored independently per channel.

    Notes
    -----
    - The DC component (0 Hz) is set to 0 before inverse FFT to avoid a large
      offset when ``exponent > 0``.
    - Each channel is standardized: ``(x - mean) / std``. If a channel has zero
      variance (rare with random inputs), its standard deviation is clamped with
      a tiny epsilon to avoid division-by-zero.
    - The exact amplitude distribution is Gaussian per time point after z-scoring,
      but the spectrum follows the targeted 1/f^alpha profile in expectation.

    Examples
    --------
    >>> x = generate_1_over_f_noise(3, 10000, exponent=1.0, sfreq=250, random_state=0)
    >>> x.shape
    (3, 10000)
    >>> np.allclose(x.mean(axis=1), 0.0, atol=1e-2)
    True
    >>> np.allclose(x.std(axis=1), 1.0, atol=1e-2)
    True
    """
    if n_channels <= 0 or n_times <= 0:
        raise ValueError("n_channels and n_times must be positive integers.")

    # RNG
    rng = (
        random_state
        if isinstance(random_state, np.random.Generator)
        else np.random.default_rng(random_state)
    )

    # White noise: (n_channels, n_times)
    white = rng.standard_normal((n_channels, n_times))

    # Frequency axis for rFFT (spacing d = 1/sfreq)
    # rfftfreq length is n_times//2 + 1
    freqs = np.fft.rfftfreq(n_times, d=1.0 / float(sfreq))

    # 1 / f^exponent scaling; set DC scale to 0 to avoid blow-up
    scale = np.empty_like(freqs, dtype=float)
    scale[0] = 0.0
    if exponent == 0.0:
        # No scaling (white); DC already zeroed.
        scale[1:] = 1.0
    else:
        # Avoid division by zero at DC (already set); scale others.
        scale[1:] = 1.0 / np.power(
            freqs[1:], exponent / 2.0
        )  # divide by 2 because power is squared amplitude

    # FFT along time axis, apply scale, and invert
    white_fft = np.fft.rfft(white, axis=-1)
    pink_fft = white_fft * scale[None, :]
    pink = np.fft.irfft(pink_fft, n=n_times, axis=-1)

    # Standardize per channel (mean 0, std 1) with epsilon guard
    pink -= pink.mean(axis=1, keepdims=True)
    std = pink.std(axis=1, keepdims=True)
    eps = 1e-12
    pink /= std + eps

    return pink


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
