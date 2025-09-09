import math
from pathlib import PureWindowsPath, PurePosixPath
from typing import Literal, Sequence, Union
import os, ntpath, posixpath
from typing import Sequence, Union

PathLike = Union[str, os.PathLike]


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


def get_path(path: str | dict[str, str], mount: str | None = None) -> str:
    """
    Retrieve the resolved path based on a given mount key or return the path directly.

    Parameters
    ----------
    path : str or dict of str
        A file path string or a dictionary mapping mount names to paths.
    mount : str, optional
        The mount key to use if ``path`` is a dictionary. If None or if ``path``
        is not a dictionary, the function returns ``path`` directly.

    Returns
    -------
    str
        The resolved path.

    Raises
    ------
    KeyError
        If ``mount`` is provided but not found in the dictionary ``path``.

    Examples
    --------
    >>> get_path("/data/files")
    '/data/files'

    >>> paths = {"local": "/mnt/local/data", "remote": "/mnt/remote/data"}
    >>> get_path(paths, mount="local")
    '/mnt/local/data'
    """
    if isinstance(path, dict) and mount is not None:
        return path[mount]
    return path

def find_minimal_unique_root(
    filepaths: Sequence[PathLike],
    *,
    style: str = "auto",   # {'auto','posix','windows'}
    strict: bool = True    # if True, reject ambiguous/mixed inputs
) -> str:
    """
    Find the shallowest root such that relative paths from it are unique.

    Parameters
    ----------
    filepaths : sequence of path-like
        Absolute or relative paths (all expected to use the same style).
    style : {'auto', 'posix', 'windows'}, optional
        Path style rules to apply. 'auto' infers from inputs:
        - any path with a drive letter or backslash ⇒ windows
        - otherwise ⇒ posix
        Mixed styles raise if strict=True.
    strict : bool, optional
        If True, raise on mixed/ambiguous separators for the chosen style.
        If False, coerce separators:
           - windows: convert '/' → '\\'
           - posix: convert '\\' → '/'

    Returns
    -------
    str
        Normalized minimal root path as a string.

    Raises
    ------
    ValueError
        If filepaths is empty, or styles are mixed with strict=True.
    """
    if not filepaths:
        raise ValueError("filepaths must be non-empty")

    raw = [os.fspath(p) for p in filepaths]

    def looks_windows(s: str) -> bool:
        # Drive letter or backslash suggests Windows intent
        return ("\\" in s) or (len(s) >= 2 and s[1] == ":")

    # Decide style
    if style == "auto":
        flags = [looks_windows(s) for s in raw]
        if any(flags) and not all(flags):
            if strict:
                raise ValueError("Mixed POSIX/Windows styles are not supported (strict=True).")
            # Lenient: pick windows if any looks windows
            use_windows = True
        else:
            use_windows = all(flags)
    elif style == "windows":
        use_windows = True
    elif style == "posix":
        use_windows = False
    else:
        raise ValueError("style must be 'auto', 'posix', or 'windows'.")

    # Pick modules/classes per style
    pmod = ntpath if use_windows else posixpath
    PurePath = PureWindowsPath if use_windows else PurePosixPath

    # Separator coercion if not strict
    paths = []
    for s in raw:
        if strict:
            if use_windows:
                # ok to have either '/' or '\\' on Windows
                pass
            else:
                # POSIX: backslash should not be a separator
                if "\\" in s:
                    raise ValueError(
                        f"Backslash found in POSIX path '{s}' (strict=True). "
                        "Use strict=False to coerce '\\' → '/'."
                    )
            paths.append(s)
        else:
            if use_windows:
                paths.append(s.replace("/", "\\"))   # normalize separators
            else:
                paths.append(s.replace("\\", "/"))   # treat backslash as separator

    # Normalize with the chosen flavor
    paths = [pmod.normpath(s) for s in paths]

    # Upper bound common prefix
    try:
        common_prefix = pmod.commonpath(paths)
    except ValueError:
        # e.g., different drives on Windows → no common path
        common_prefix = ""

    split = [PurePath(s).parts for s in paths]
    prefix_len = len(PurePath(common_prefix).parts)

    # Try shallowest→deepest; first that yields unique relpaths wins
    for i in range(1, prefix_len + 1):
        rels = []
        for parts in split:
            tail = parts[i:]
            rels.append(pmod.join(*tail) if tail else "")
        if len(rels) == len(set(rels)):
            head = split[0][:i]
            return pmod.normpath(pmod.join(*head))

    return pmod.normpath(common_prefix) if common_prefix else common_prefix