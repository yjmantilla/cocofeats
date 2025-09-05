import math
from typing import Literal


def get_num_digits(n: int, method: Literal["safe", "fast"] = "safe") -> int:
    """Return the number of decimal digits in an integer.

    Parameters
    ----------
    n : int
        The integer whose digits should be counted.
    method : {"safe", "fast"}, default="safe"
        - "safe": Uses string conversion (`len(str(abs(n)))`).
          Always correct, even for very large integers.
        - "fast": Uses ``log10``.
          Faster for large numbers, but depends on floating-point precision.
          Still safe for most practical integer ranges.

    Returns
    -------
    int
        The number of digits in the absolute value of ``n``.

    Examples
    --------
    >>> get_num_digits(0)
    1
    >>> get_num_digits(12345)
    5
    >>> get_num_digits(-999, method="fast")
    3
    """
    if n == 0:
        return 1

    n_abs = abs(n)

    if method == "safe":
        return len(str(n_abs))

    if method == "fast":
        return int(math.log10(n_abs)) + 1

    raise ValueError(f"Unknown method: {method!r}. Use 'safe' or 'fast'.")
