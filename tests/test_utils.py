# tests/test_get_num_digits.py
import pytest

from cocofeats.utils import get_num_digits, get_path


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
