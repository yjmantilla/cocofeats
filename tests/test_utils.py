# tests/test_get_num_digits.py
import math
import pytest

from cocofeats.utils import get_num_digits


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
    n = 10**50  # 51 digits
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
