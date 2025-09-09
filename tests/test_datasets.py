# Tests for cocofeats.datasets module
import os
import numpy as np
from pathlib import Path
import pytest
import mne

from cocofeats.datasets import (
    replace_brainvision_filename,
    make_dummy_dataset,
    generate_1_over_f_noise,
    get_dummy_raw,
)


# Tests for replace_brainvision_filename function


def test_replace_in_common_infos_only(tmp_path: Path):
    vhdr = tmp_path / "rec.vhdr"
    content = (
        "[Common Infos]\n"
        "DataFile=oldname.eeg\n"
        "MarkerFile=oldname.vmrk\n"
        "\n"
        "[Binary Infos]\n"
        "DataFormat=BINARY\n"
        "DataFile=should_not_change.eeg\n"
    )
    vhdr.write_text(content, encoding="utf-8")

    replace_brainvision_filename(vhdr, "newbase")

    txt = vhdr.read_text(encoding="utf-8")
    # Changed inside [Common Infos]
    assert "DataFile=newbase.eeg" in txt
    assert "MarkerFile=newbase.vmrk" in txt
    # Unchanged in other sections
    assert "DataFile=should_not_change.eeg" in txt


def test_fallback_when_no_common_infos(tmp_path: Path):
    vhdr = tmp_path / "rec.vhdr"
    content = "DataFile=old.eeg\n" "MarkerFile=old.vmrk\n" "[Some Other]\n" "Key=Value\n"
    vhdr.write_text(content, encoding="utf-8")

    replace_brainvision_filename(vhdr, "baseX")

    txt = vhdr.read_text(encoding="utf-8")
    assert "DataFile=baseX.eeg" in txt
    assert "MarkerFile=baseX.vmrk" in txt
    # Other content preserved
    assert "[Some Other]" in txt
    assert "Key=Value" in txt


def test_preserve_crlf_line_endings(tmp_path: Path):
    vhdr = tmp_path / "rec.vhdr"
    # Force CRLF endings on the lines we will replace
    content = (
        b"[Common Infos]\r\n"
        b"DataFile=old.eeg\r\n"
        b"MarkerFile=old.vmrk\r\n"
        b"\r\n"
        b"[Comment]\n"
        b"This is fine.\n"
    )
    vhdr.write_bytes(content)

    replace_brainvision_filename(vhdr, "session01")

    data = vhdr.read_bytes()
    # Check CRLF preserved on replaced lines
    assert b"DataFile=session01.eeg\r\n" in data
    assert b"MarkerFile=session01.vmrk\r\n" in data
    # Unchanged LF in other lines
    assert b"[Comment]\n" in data
    assert b"This is fine.\n" in data


def test_case_insensitive_keys_and_extensions(tmp_path: Path):
    vhdr = tmp_path / "rec.vhdr"
    content = "[Common Infos]\n" "datafile=OLD.EEG\n" "markerfile=OLD.VMRK\n"
    vhdr.write_text(content, encoding="utf-8")

    # Provide newname with extension (case-insensitive) and directories
    replace_brainvision_filename(vhdr, "subdir/session01.EeG")

    txt = vhdr.read_text(encoding="utf-8")
    assert "DataFile=session01.eeg" in txt
    assert "MarkerFile=session01.vmrk" in txt


def test_latin1_roundtrip(tmp_path: Path):
    vhdr = tmp_path / "rec.vhdr"
    # Include a Latin-1 only char (ó) in a comment
    latin1_text = (
        "[Common Infos]\r\n"
        "DataFile=old.eeg\r\n"
        "MarkerFile=old.vmrk\r\n"
        "; Comentario con acent\u00f3\r\n"  # ó
    )
    # Write as latin-1
    vhdr.write_bytes(latin1_text.encode("latin-1"))

    replace_brainvision_filename(vhdr, "ses01")

    # Ensure file still readable as latin-1 and ó preserved
    raw = vhdr.read_bytes()
    decoded = raw.decode("latin-1")
    assert "Comentario con acentó" in decoded
    assert "DataFile=ses01.eeg" in decoded
    assert "MarkerFile=ses01.vmrk" in decoded


def test_no_keys_no_change(tmp_path: Path):
    vhdr = tmp_path / "rec.vhdr"
    content = "[Common Infos]\n" "Version=1.0\n" "SomeOtherKey=Value\n"
    vhdr.write_text(content, encoding="utf-8")
    before = vhdr.read_bytes()

    replace_brainvision_filename(vhdr, "anything")

    after = vhdr.read_bytes()
    assert before == after  # nothing to change, file untouched content-wise


def test_missing_file_raises(tmp_path: Path):
    vhdr = tmp_path / "does_not_exist.vhdr"
    with pytest.raises(FileNotFoundError):
        replace_brainvision_filename(vhdr, "base")


def test_fallback_replaces_anywhere_when_no_common_infos(tmp_path: Path):
    # Ensure that if [Common Infos] is absent, *any* occurrences are changed
    vhdr = tmp_path / "rec.vhdr"
    content = "[Other]\n" "DataFile=old.eeg\n" "MarkerFile=old.vmrk\n"
    vhdr.write_text(content, encoding="utf-8")

    replace_brainvision_filename(vhdr, "X")

    txt = vhdr.read_text(encoding="utf-8")
    assert "DataFile=X.eeg" in txt
    assert "MarkerFile=X.vmrk" in txt


# Tests for make_dummy_dataset function


def _write(p: Path, text: str = "dummy\n") -> Path:
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(text, encoding="utf-8")
    return p


def test_creates_expected_structure_and_counts(tmp_path: Path):
    ex = _write(tmp_path / "example.dat", "content\n")

    root = tmp_path / "out"
    # Grid: NSUBS=2, NRUNS=2, others=1 → 4 files per example
    make_dummy_dataset(
        EXAMPLE=str(ex),
        ROOT=str(root),
        NSUBS=2,
        NTASKS=1,
        NSESSIONS=1,
        NACQS=1,
        NRUNS=2,
        DATASET="DUMMY",
        PATTERN="T%task%/S%session%/sub%subject%_%acquisition%_%run%",
    )

    files = list(root.rglob("*.dat"))
    assert len(files) == 4

    # 0-based indices with zero-pad width inferred from counts (=1 here)
    expect1 = root / "TTA0" / "SSE0" / "subSU0_AC0_0.dat"
    expect2 = root / "TTA0" / "SSE0" / "subSU1_AC0_1.dat"
    assert expect1.exists()
    assert expect2.exists()


def test_indices_zero_based_and_zero_padded(tmp_path: Path):
    # NSUBS=12 → subjects SU00..SU11 (width=2)
    ex = _write(tmp_path / "example.bin", "x\n")
    root = tmp_path / "outpad"

    make_dummy_dataset(
        EXAMPLE=str(ex),
        ROOT=str(root),
        NSUBS=12,
        NTASKS=1,
        NSESSIONS=1,
        NACQS=1,
        NRUNS=1,
        DATASET="Demo",
        PATTERN="T%task%/S%session%/sub%subject%_%acquisition%_%run%",
    )

    files = list(root.rglob("*.bin"))
    assert len(files) == 12

    p0 = root / "TTA0" / "SSE0" / "subSU00_AC0_0.bin"
    p11 = root / "TTA0" / "SSE0" / "subSU11_AC0_0.bin"
    p12 = root / "TTA0" / "SSE0" / "subSU12_AC0_0.bin"  # should NOT exist (range goes 0..11)
    assert p0.exists()
    assert p11.exists()
    assert not p12.exists()


def test_multiple_examples_copied_per_combination(tmp_path: Path):
    ex1 = _write(tmp_path / "ex.txt", "a\n")
    ex2 = _write(tmp_path / "ex.dat", "b\n")
    root = tmp_path / "multi"

    make_dummy_dataset(
        EXAMPLE=[str(ex1), str(ex2)],
        ROOT=str(root),
        NSUBS=1,
        NTASKS=1,
        NSESSIONS=1,
        NACQS=1,
        NRUNS=2,
        DATASET="D",
        PATTERN="T%task%/S%session%/sub%subject%_%acquisition%_%run%",
    )

    base0 = root / "TTA0" / "SSE0" / "subSU0_AC0_0"
    base1 = root / "TTA0" / "SSE0" / "subSU0_AC0_1"
    assert (base0.with_suffix(".txt")).exists()
    assert (base0.with_suffix(".dat")).exists()
    assert (base1.with_suffix(".txt")).exists()
    assert (base1.with_suffix(".dat")).exists()


def test_brainvision_headers_updated(tmp_path: Path):
    # Minimal BrainVision trio: .vhdr/.vmrk headers + .eeg blob
    vhdr = tmp_path / "template.vhdr"
    vmrk = tmp_path / "template.vmrk"
    eeg = tmp_path / "template.eeg"

    vhdr.write_text(
        "[Common Infos]\n"
        "DataFile=oldname.eeg\n"
        "MarkerFile=oldname.vmrk\n"
        "\n[Binary Infos]\n"
        "DataFormat=BINARY\n",
        encoding="utf-8",
    )
    vmrk.write_text(
        "[Common Infos]\n" "Codepage=UTF-8\n",
        encoding="utf-8",
    )
    eeg.write_bytes(b"\x00" * 8)

    root = tmp_path / "bv"
    make_dummy_dataset(
        EXAMPLE=[str(vhdr), str(vmrk), str(eeg)],
        ROOT=str(root),
        NSUBS=1,
        NTASKS=1,
        NSESSIONS=1,
        NACQS=1,
        NRUNS=1,
        DATASET="BV",
        PATTERN="T%task%/S%session%/sub%subject%_%acquisition%_%run%",
    )

    gen_base = root / "TTA0" / "SSE0" / "subSU0_AC0_0"
    out_vhdr = gen_base.with_suffix(".vhdr")
    out_vmrk = gen_base.with_suffix(".vmrk")
    out_eeg = gen_base.with_suffix(".eeg")

    assert out_vhdr.exists()
    assert out_vmrk.exists()
    assert out_eeg.exists()

    txt = out_vhdr.read_text(encoding="utf-8")
    assert "DataFile=subSU0_AC0_0.eeg" in txt
    assert "MarkerFile=subSU0_AC0_0.vmrk" in txt


def test_custom_prefixes_applied(tmp_path: Path):
    ex = _write(tmp_path / "file.bin", "x\n")
    root = tmp_path / "custom"

    prefixes = {"subject": "S", "session": "Sess", "task": "T", "acquisition": "A", "run": "R"}

    make_dummy_dataset(
        EXAMPLE=str(ex),
        ROOT=str(root),
        NSUBS=1,
        NTASKS=1,
        NSESSIONS=1,
        NACQS=1,
        NRUNS=1,
        DATASET="X",
        PREFIXES=prefixes,
        PATTERN="T%task%/S%session%/sub%subject%_%acquisition%_%run%",
    )

    expected = root / "TT0" / "SSess0" / "subS0_A0_0.bin"
    assert expected.exists()


def test_raises_when_example_missing(tmp_path: Path):
    root = tmp_path / "err"
    missing = tmp_path / "nope.dat"
    with pytest.raises(FileNotFoundError):
        make_dummy_dataset(
            EXAMPLE=str(missing),
            ROOT=str(root),
            NSUBS=1,
            NTASKS=1,
            NSESSIONS=1,
            NACQS=1,
            NRUNS=1,
        )


# Test for generate_1_over_f_noise function


def _welch_psd(x: np.ndarray, sfreq: float, n_seg: int = 1024, overlap: float = 0.5):
    """
    Minimal Welch PSD estimate for 1D signal x (time,), returns (freqs, psd).
    """
    x = np.asarray(x, float)
    n = x.size
    step = int(n_seg * (1 - overlap))
    if step <= 0:
        step = n_seg // 2 or 1
    # Build segments
    starts = np.arange(0, max(n - n_seg + 1, 1), step, dtype=int)
    if starts.size == 0:
        starts = np.array([0], dtype=int)
    window = np.hanning(n_seg)
    wnorm = (window ** 2).sum()
    acc = None
    for s in starts:
        seg = x[s : s + n_seg]
        if seg.size < n_seg:
            # zero-pad last segment
            seg = np.pad(seg, (0, n_seg - seg.size))
        seg = seg * window
        spec = np.fft.rfft(seg)
        psd = (np.abs(spec) ** 2) / (wnorm * sfreq)
        acc = psd if acc is None else (acc + psd)
    psd = acc / starts.size
    freqs = np.fft.rfftfreq(n_seg, d=1.0 / sfreq)
    return freqs, psd


def _slope_loglog(freqs: np.ndarray, psd: np.ndarray) -> float:
    """
    Fit slope of log10(PSD) vs log10(freq) over a mid-band region.
    Excludes DC and highest 10% of bins to avoid edge effects.
    """
    valid = freqs > 0
    freqs = freqs[valid]
    psd = psd[valid]
    if freqs.size < 8:
        # too few points; fall back to simple fit
        pass
    # Use middle band: 10%..90% to avoid edges
    lo = int(0.10 * freqs.size)
    hi = int(0.90 * freqs.size)
    if hi <= lo:
        lo, hi = 0, freqs.size
    f = freqs[lo:hi]
    p = psd[lo:hi]
    slope, _ = np.polyfit(np.log10(f), np.log10(p + 1e-24), 1)
    return slope


def test_shape_and_zscore_stats():
    n_channels, n_times = 6, 20_000
    x = generate_1_over_f_noise(n_channels, n_times, exponent=1.0, sfreq=250.0, random_state=0)
    assert x.shape == (n_channels, n_times)
    means = x.mean(axis=1)
    stds = x.std(axis=1)
    assert np.allclose(means, 0.0, atol=3e-2)
    assert np.allclose(stds, 1.0, atol=3e-2)


def test_reproducibility_seed_and_generator():
    # Seed reproducibility
    a1 = generate_1_over_f_noise(3, 8192, exponent=1.0, random_state=123)
    a2 = generate_1_over_f_noise(3, 8192, exponent=1.0, random_state=123)
    b = generate_1_over_f_noise(3, 8192, exponent=1.0, random_state=124)
    assert np.allclose(a1, a2)
    assert not np.allclose(a1, b)

    # Generator reproducibility
    g1 = np.random.default_rng(999)
    g2 = np.random.default_rng(999)
    c1 = generate_1_over_f_noise(2, 8192, exponent=0.5, random_state=g1)
    c2 = generate_1_over_f_noise(2, 8192, exponent=0.5, random_state=g2)
    assert np.allclose(c1, c2)


@pytest.mark.parametrize(
    "exp, tol",
    [
        (0.0, 0.20),  # white ≈ flat
        (0.5, 0.40),  # pinkish
        (1.0, 0.40),  # pink
    ],
)
def test_spectral_slope_matches_exponent(exp, tol):
    # Average slope across channels to reduce variance
    sfreq = 200.0
    n_times = 32768
    n_channels = 8
    x = generate_1_over_f_noise(n_channels, n_times, exponent=exp, sfreq=sfreq, random_state=7)

    slopes = []
    for ch in range(n_channels):
        freqs, psd = _welch_psd(x[ch], sfreq=sfreq, n_seg=2048, overlap=0.5)
        slopes.append(_slope_loglog(freqs, psd))
    mean_slope = float(np.mean(slopes))
    # For 1/f^alpha, slope ≈ -alpha
    print(f"Exponent: {exp}, mean slope: {mean_slope:.2f} (tol {tol})")
    assert abs(mean_slope + exp) < tol


def test_dc_component_near_zero_after_standardization():
    n_times = 4096
    x = generate_1_over_f_noise(4, n_times, exponent=1.0, random_state=0)
    dc = np.fft.rfft(x, axis=-1)[..., 0].real
    assert np.allclose(dc, 0.0, atol=1e-10)


def test_slope_consistency_across_sfreq():
    # The spectral slope is unitless; it should be consistent across sfreq choices.
    n_times = 32768
    exp = 0.8
    a = generate_1_over_f_noise(3, n_times, exponent=exp, sfreq=120.0, random_state=42)[0]
    b = generate_1_over_f_noise(3, n_times, exponent=exp, sfreq=360.0, random_state=42)[0]

    fa, pa = _welch_psd(a, sfreq=120.0, n_seg=2048, overlap=0.5)
    fb, pb = _welch_psd(b, sfreq=360.0, n_seg=2048, overlap=0.5)
    sa = _slope_loglog(fa, pa)
    sb = _slope_loglog(fb, pb)
    assert abs(sa - sb) < 0.2  # slopes should be close regardless of sampling rate


@pytest.mark.parametrize("nc, nt", [(-1, 100), (0, 100), (2, 0), (2, -10)])
def test_invalid_sizes_raise(nc, nt):
    with pytest.raises(ValueError):
        _ = generate_1_over_f_noise(nc, nt, exponent=1.0, random_state=0)


# Tests for get_dummy_raw function


def test_basic_shape_and_info():
    n_ch, sfreq, stop, n_events = 4, 200.0, 2.5, 10  # n_times = round(200*2.5)=500
    raw, events = get_dummy_raw(
        NCHANNELS=n_ch, SFREQ=sfreq, STOP=stop, NUMEVENTS=n_events, random_state=0
    )

    assert isinstance(raw, mne.io.BaseRaw)
    assert raw.info["sfreq"] == sfreq
    assert raw.get_data().shape == (n_ch, 500)

    # channel names and types
    assert raw.ch_names == [f"EEG{idx:03d}" for idx in range(n_ch)]
    # assert all(kind == "eeg" for kind in mne.io.pick.channel_type(raw.info, picks=range(n_ch)))
    ch_types = raw.get_channel_types()  # <- robust across MNE versions
    assert ch_types == ["eeg"] * n_ch

    # events: shape and columns
    assert events.shape == (n_events, 3)
    assert np.all(events[:, 1] == 0)  # middle column zeros
    assert np.all(events[:, 2] == 1)  # default event_id

    # events strictly increasing and within range
    assert np.all(np.diff(events[:, 0]) > 0)
    assert events[0, 0] >= 0
    assert events[-1, 0] < raw.n_times


def test_events_evenly_spaced_and_exact_count():
    # Choose parameters so n_times is large compared to NUMEVENTS
    raw, events = get_dummy_raw(NCHANNELS=2, SFREQ=100.0, STOP=10.0, NUMEVENTS=25, random_state=1)
    # evenly spaced in integer samples (monotonic, no duplicates, exact count)
    samples = events[:, 0]
    assert len(samples) == len(np.unique(samples))
    diffs = np.diff(samples)
    # Allow small variation due to integer rounding, but most diffs should be close to the median
    med = int(np.median(diffs))
    assert med > 0
    assert np.mean(np.abs(diffs - med)) <= 1.0


def test_zscore_stats_reasonable():
    raw, _ = get_dummy_raw(NCHANNELS=3, SFREQ=250.0, STOP=8.0, NUMEVENTS=8, random_state=42)
    X = raw.get_data()
    means = X.mean(axis=1)
    stds = X.std(axis=1)
    assert np.allclose(means, 0.0, atol=3e-2)
    assert np.allclose(stds, 1.0, atol=3e-2)


def test_reproducibility_seed_and_generator():
    a_raw, a_ev = get_dummy_raw(3, 200.0, 5.0, 7, random_state=123)
    b_raw, b_ev = get_dummy_raw(3, 200.0, 5.0, 7, random_state=123)
    c_raw, c_ev = get_dummy_raw(3, 200.0, 5.0, 7, random_state=124)

    assert np.allclose(a_raw.get_data(), b_raw.get_data())
    assert np.array_equal(a_ev, b_ev)

    # Different seed should differ with overwhelming probability
    assert not np.allclose(a_raw.get_data(), c_raw.get_data()) or not np.array_equal(a_ev, c_ev)

    # Reproducibility with Generator
    g1 = np.random.default_rng(999)
    g2 = np.random.default_rng(999)
    r1, e1 = get_dummy_raw(2, 128.0, 4.0, 5, random_state=g1)
    r2, e2 = get_dummy_raw(2, 128.0, 4.0, 5, random_state=g2)
    assert np.allclose(r1.get_data(), r2.get_data())
    assert np.array_equal(e1, e2)


def test_event_id_custom():
    raw, events = get_dummy_raw(2, 100.0, 3.0, 6, event_id=7, random_state=0)
    assert np.all(events[:, 2] == 7)


def test_numevents_equal_to_samples():
    # n_times = round(50 * 2.0) = 100
    raw, events = get_dummy_raw(1, 50.0, 2.0, 100, random_state=0)
    samples = events[:, 0]
    assert samples[0] == 0
    assert samples[-1] == raw.n_times - 1  # last valid index
    assert np.all(np.diff(samples) == 1)  # one per sample


@pytest.mark.parametrize(
    "kwargs",
    [
        dict(NCHANNELS=0),
        dict(SFREQ=0.0),
        dict(STOP=0.0),
        dict(NUMEVENTS=0),
    ],
)
def test_invalid_parameters_raise(kwargs):
    base = dict(NCHANNELS=2, SFREQ=100.0, STOP=2.0, NUMEVENTS=2)
    base.update(kwargs)
    with pytest.raises(ValueError):
        _ = get_dummy_raw(**base)


def test_numevents_exceeds_samples_raises():
    # n_times = round(10 * 0.5) = 5; request 6 events → error
    with pytest.raises(ValueError):
        _ = get_dummy_raw(NCHANNELS=1, SFREQ=10.0, STOP=0.5, NUMEVENTS=6, random_state=0)
