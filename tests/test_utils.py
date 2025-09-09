# tests/test_get_num_digits.py
import pytest
import os
from cocofeats.utils import get_num_digits, get_path, find_minimal_unique_root
import ntpath
import posixpath
from pathlib import PureWindowsPath, PurePosixPath


# Tests for get_num_digits function

@pytest.mark.parametrize(
    "n, expected",
    [
        (0, 1),
        (5, 1),
        (9, 1),
        (10, 2),
        (123, 3),
        (-1, 1),
        (-999, 3),
    ],
)
def test_safe_mode_basic(n, expected):
    assert get_num_digits(n, method="safe") == expected


@pytest.mark.parametrize(
    "n, expected",
    [
        (1, 1),
        (9, 1),
        (10, 2),
        (12345, 5),
        (-12345, 5),
    ],
)
def test_fast_mode_basic(n, expected):
    assert get_num_digits(n, method="fast") == expected


def test_zero_in_fast_mode():
    assert get_num_digits(0, method="fast") == 1


def test_large_number_consistency():
    n = 10 ** 50  # 51 digits
    safe = get_num_digits(n, method="safe")
    fast = get_num_digits(n, method="fast")
    # Both should agree
    assert safe == fast == 51


def test_invalid_method_raises():
    with pytest.raises(ValueError) as excinfo:
        get_num_digits(123, method="wrong")
    assert "Unknown method" in str(excinfo.value)


def test_negative_numbers_consistency():
    for n in [-1, -12, -123456]:
        safe = get_num_digits(n, method="safe")
        fast = get_num_digits(n, method="fast")
        assert safe == fast
        assert safe == len(str(abs(n)))


# Tests for get_path function

def test_returns_same_path_for_string_no_mount():
    path = "/data/files"
    assert get_path(path) == "/data/files"


def test_returns_same_path_for_string_with_mount_ignored():
    path = "/data/files"
    assert get_path(path, mount="local") == "/data/files"


def test_returns_mounted_value_when_dict_and_mount_present():
    paths = {"local": "/mnt/local/data", "remote": "/mnt/remote/data"}
    assert get_path(paths, mount="local") == "/mnt/local/data"
    assert get_path(paths, mount="remote") == "/mnt/remote/data"


def test_raises_keyerror_when_mount_missing_in_dict():
    paths = {"local": "/mnt/local/data"}
    with pytest.raises(KeyError):
        _ = get_path(paths, mount="remote")


def test_returns_dict_when_mount_is_none_and_path_is_dict():
    # NOTE: Given the current implementation, if `path` is a dict and `mount` is None,
    # the function returns the dictionary itself.
    # If you want it to *always* return a string, update the implementation accordingly.
    paths = {"local": "/mnt/local/data"}
    assert get_path(paths, mount=None) is paths


# Helper functions for testing find_minimal_unique_root


def _pick_flavor(paths) -> tuple:
    """Return (pmod, PurePath, use_windows) based on the inputs."""
    s = [os.fspath(p) for p in paths]
    looks_win = [("\\" in x) or (len(x) >= 2 and x[1] == ":") for x in s]
    use_windows = all(looks_win)
    pmod = ntpath if use_windows else posixpath
    PurePath = PureWindowsPath if use_windows else PurePosixPath
    return pmod, PurePath, use_windows


def _is_prefix_path(prefix, path) -> bool:
    pmod, _, _ = _pick_flavor([prefix, path])
    prefix = pmod.normpath(os.fspath(prefix))
    path = pmod.normpath(os.fspath(path))
    try:
        common = pmod.commonpath([prefix, path])
    except ValueError:
        # e.g., different drives on Windows → no common path
        return False
    return common == prefix


def _relpaths_from_root(root, filepaths) -> list[str]:
    pmod, _, _ = _pick_flavor([root, *filepaths])
    root = pmod.normpath(os.fspath(root))
    return [pmod.relpath(pmod.normpath(os.fspath(p)), root) for p in filepaths]


def _has_unique_relpaths(root, filepaths) -> bool:
    rels = _relpaths_from_root(root, filepaths)
    return len(rels) == len(set(rels))


def _parent_dir(path) -> str | None:
    pmod, PurePath, use_windows = _pick_flavor([path])
    path = pmod.normpath(os.fspath(path))
    parent = pmod.dirname(path)

    # Treat filesystem roots as having no parent
    if parent == path:
        return None
    if use_windows:
        drv = ntpath.splitdrive(path)[0]
        # Drive root like "C:\" (normalized) has no parent
        if path in (drv + "\\", drv + "/"):
            return None
    else:
        if path == "/":
            return None
    return parent
    return None if parent == path else parent

# Tests for find_minimal_unique_root function

@pytest.mark.parametrize(
    "filepaths",
    [
        # Same directory, different filenames
        ["/data/p/x.txt", "/data/p/y.txt"],
        # Different directories, same filename
        ["/data/project1/images/cat.png", "/data/project2/images/cat.png"],
        # Deeper structures that only diverge late
        ["/data/A/B/C/file.bin", "/data/A/B/D/file2.bin", "/data/A/E/F/file3.bin"],
        # Mixed files across a broader tree
        ["/mnt/a/x/a.txt", "/mnt/a/y/b.txt", "/mnt/b/x/c.txt", "/mnt/b/z/d.txt"],
    ],
)
def test_minimal_unique_root_properties_posix(filepaths, monkeypatch):
    # Force POSIX-style behavior for consistency on all OSes during tests
    monkeypatch.setenv("PYTHONIOENCODING", "utf-8")

    # Precondition: all absolute POSIX-like paths
    assert all(p.startswith("/") for p in filepaths)

    root = find_minimal_unique_root(filepaths)

    # 1) Root must be a common ancestor of all files
    assert all(_is_prefix_path(root, p) for p in filepaths), "Returned root is not a common ancestor."

    # 2) Relative paths from root must be unique
    assert _has_unique_relpaths(root, filepaths), "Relative paths are not unique from the returned root."

    # 3) Minimality: parent of root must NOT satisfy uniqueness (or root is already top-level)
    parent = _parent_dir(root)
    if parent is not None:
        assert not _has_unique_relpaths(parent, filepaths), (
            "Returned root is not minimal: its parent still yields unique relative paths."
        )


def test_identical_paths_edge_case():
    # When inputs contain identical paths, uniqueness is impossible from any higher root.
    # The function should fall back to the full common path (which equals that path).
    filepaths = ["/data/p/x.txt", "/data/p/x.txt"]
    root = find_minimal_unique_root(filepaths)

    # Must be a common ancestor (trivially true if equals the path)
    assert _is_prefix_path(root, filepaths[0])

    # From that root, relpaths are not unique — that's okay; the function promises fallback.
    # We only require that the returned root equals the full common path in this scenario.
    common = os.path.commonpath(filepaths)
    assert os.path.normpath(root) == os.path.normpath(common)


def test_single_path_trivial_case():
    # With a single path, the minimal unique root can be the path itself
    # (since there is nothing to disambiguate).
    filepaths = ["/only/one/file.txt"]
    root = find_minimal_unique_root(filepaths)

    assert _is_prefix_path(root, filepaths[0])
    # Relative path uniqueness is trivially satisfied
    assert _has_unique_relpaths(root, filepaths)


@pytest.mark.parametrize(
    "filepaths",
    [
        ["C:\\data\\p\\x.txt", "C:\\data\\p\\y.txt"],
        ["C:\\data\\proj1\\images\\cat.png", "C:\\data\\proj2\\images\\cat.png"],
    ],
)
def test_minimal_unique_root_windows_like_paths(filepaths):
    # These are Windows-like paths; the implementation should handle them
    # if it relies on os.path/commonpath properly. If your implementation
    # is POSIX-only, mark these xfail or normalize inputs before calling.
    root = find_minimal_unique_root(filepaths)

    # Common-ancestor + uniqueness properties
    assert all(_is_prefix_path(root, p) for p in filepaths)
    assert _has_unique_relpaths(root, filepaths)

    # Minimality: parent should fail uniqueness (or root has no parent)
    parent = _parent_dir(root)
    if parent is not None:
        assert not _has_unique_relpaths(parent, filepaths)

if __name__ == "__main__":
    #pytest.main([__file__])
    test_identical_paths_edge_case()