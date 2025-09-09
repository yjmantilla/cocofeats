# tests/test_replace_brainvision_filename.py
import os
from pathlib import Path
import pytest
from cocofeats.datasets import replace_brainvision_filename


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
    content = (
        "DataFile=old.eeg\n"
        "MarkerFile=old.vmrk\n"
        "[Some Other]\n"
        "Key=Value\n"
    )
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
    content = (
        "[Common Infos]\n"
        "datafile=OLD.EEG\n"
        "markerfile=OLD.VMRK\n"
    )
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
    content = (
        "[Common Infos]\n"
        "Version=1.0\n"
        "SomeOtherKey=Value\n"
    )
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
    content = (
        "[Other]\n"
        "DataFile=old.eeg\n"
        "MarkerFile=old.vmrk\n"
    )
    vhdr.write_text(content, encoding="utf-8")

    replace_brainvision_filename(vhdr, "X")

    txt = vhdr.read_text(encoding="utf-8")
    assert "DataFile=X.eeg" in txt
    assert "MarkerFile=X.vmrk" in txt
