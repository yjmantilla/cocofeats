# tests/test_replace_brainvision_filename.py
import os
from pathlib import Path
import pytest
from cocofeats.datasets import replace_brainvision_filename, make_dummy_dataset


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
