import math
import ntpath
import os
import posixpath
from collections.abc import Sequence
from pathlib import PurePosixPath, PureWindowsPath
from typing import Literal
from cocofeats.definitions import PathLike
from pathlib import Path

from cocofeats.loggers import get_logger

log = get_logger(__name__)

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
    log.debug("get_num_digits: called", n=n, method=method)

    if n == 0:
        log.debug("get_num_digits: zero has one digit")
        return 1

    n_abs = abs(n)

    if method == "safe":
        digits = len(str(n_abs))
        log.debug("get_num_digits: computed (safe)", n=n, digits=digits)
        return digits

    if method == "fast":
        digits = int(math.log10(n_abs)) + 1
        log.debug("get_num_digits: computed (fast)", n=n, digits=digits)
        return digits

    log.debug("get_num_digits: invalid method", method=method)
    raise ValueError(f"Unknown method: {method!r}. Use 'safe' or 'fast'.")


def get_path(path: str | dict[str, str], mount_point: str | None = None) -> str:
    """
    Retrieve the resolved path based on a given mount_point key or return the path directly.

    Parameters
    ----------
    path : str or dict of str
        A file path string or a dictionary mapping mount_point names to paths.
    mount_point : str, optional
        The mount_point key to use if ``path`` is a dictionary. If None or if ``path``
        is not a dictionary, the function returns ``path`` directly.

    Returns
    -------
    str
        The resolved path.

    Raises
    ------
    KeyError
        If ``mount_point`` is provided but not found in the dictionary ``path``.

    Examples
    --------
    >>> get_path("/data/files")
    '/data/files'

    >>> paths = {"local": "/mnt/local/data", "remote": "/mnt/remote/data"}
    >>> get_path(paths, mount_point="local")
    '/mnt/local/data'
    """
    log.debug("get_path: called", mount_point=mount_point, path_type=type(path).__name__)
    if isinstance(path, dict) and mount_point is not None:
        try:
            resolved = path[mount_point]
            log.debug("get_path: resolved via mount_point", mount_point=mount_point, resolved=resolved)
            return resolved
        except KeyError:
            log.debug("get_path: missing mount_point key", mount_point=mount_point, available=list(path.keys()))
            raise
    log.debug("get_path: returning direct path", path=path)
    return path


def replace_bids_suffix(path: PathLike, new_suffix: str, new_ext: str) -> Path:
    """
    Replace the suffix and extension of a BIDS-like filename.

    Parameters
    ----------
    path : PathLike
        Original file path (``str`` or ``os.PathLike``).
    new_suffix : str
        New suffix (without leading underscore).
    new_ext : str
        New extension. Can handle multi-part extensions (e.g. ``".nii.gz"``). 
        Must include the leading dot.

    Returns
    -------
    Path
        Path object with replaced suffix and extension.

    Notes
    -----
    - Handles multi-part extensions (e.g., ``.nii.gz``).
    - Splits on the last underscore to isolate suffix **only if the file already
      has an extension**. If no extension is present, keeps the full name.
    """
    path = Path(path)
    log.debug("Received path for suffix replacement", path=str(path))

    stem = path.name
    exts = "".join(path.suffixes)   # ".nii.gz" or ""
    base = stem[: -len(exts)] if exts else stem
    log.debug("Split stem into base and exts", base=base, exts=exts)

    prefix = base
    if exts and "_" in base:
        prefix, _ = base.rsplit("_", 1)
        log.debug("Detected existing suffix, isolating prefix", prefix=prefix)
    elif not exts and "_" in base:
        log.info("No extension present, keeping full base as prefix", prefix=prefix)
    else:
        log.debug("No underscore found, using full base", prefix=prefix)

    new_name = f"{prefix}_{new_suffix}{new_ext}"
    log.debug("Constructed new filename", new_name=new_name)

    return path.with_name(new_name)


def find_unique_root(
    filepaths: Sequence[PathLike],
    *,
    style: str = "auto",   # {'auto','posix','windows'}
    strict: bool = True,   # if True, reject ambiguous/mixed inputs
    mode: str = "minimal", # {'minimal','maximal'}
) -> str:
    """
    Find the shallowest or deepest root such that relative paths from it are unique.

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
        - Windows: convert ``/`` → ``\\``
        - POSIX: convert ``\\`` → ``/``
    mode : {'minimal', 'maximal'}, optional
        - 'minimal': return the shallowest root that ensures unique relpaths.
        - 'maximal': return the deepest such root.

    Returns
    -------
    str
        Normalized root path as a string.

    Raises
    ------
    ValueError
        If filepaths is empty, styles are mixed with strict=True,
        or mode is invalid.
    """
    if not filepaths:
        log.debug("find_unique_root: empty input")
        raise ValueError("filepaths must be non-empty")

    raw = [os.fspath(p) for p in filepaths]
    log.debug("find_unique_root: called", n_paths=len(raw), style=style, strict=strict, mode=mode)

    def looks_windows(s: str) -> bool:
        # Drive letter or backslash suggests Windows intent
        return ("\\" in s) or (len(s) >= 2 and s[1] == ":")

    # Decide style
    if style == "auto":
        flags = [looks_windows(s) for s in raw]
        if any(flags) and not all(flags):
            if strict:
                log.debug("find_unique_root: mixed styles rejected (strict=True)")
                raise ValueError("Mixed POSIX/Windows styles are not supported (strict=True).")
            # Lenient: pick windows if any looks windows
            use_windows = True
            log.debug("find_unique_root: mixed styles coerced to windows (lenient)")
        else:
            use_windows = all(flags)
            log.debug(
                "find_unique_root: inferred style",
                inferred="windows" if use_windows else "posix",
            )
    elif style == "windows":
        use_windows = True
        log.debug("find_unique_root: explicit style", chosen="windows")
    elif style == "posix":
        use_windows = False
        log.debug("find_unique_root: explicit style", chosen="posix")
    else:
        log.debug("find_unique_root: invalid style", style=style)
        raise ValueError("style must be 'auto', 'posix', or 'windows'.")

    # Pick modules/classes per style
    pmod = ntpath if use_windows else posixpath
    PurePath = PureWindowsPath if use_windows else PurePosixPath

    # Separator coercion if not strict
    paths = []
    for s in raw:
        if strict:
            if not use_windows and "\\" in s:
                log.debug(
                    "find_unique_root: backslash in POSIX path with strict=True", path=s
                )
                raise ValueError(
                    f"Backslash found in POSIX path '{s}' (strict=True). "
                    "Use strict=False to coerce '\\' → '/'."
                )
            paths.append(s)
        else:
            if use_windows:
                paths.append(s.replace("/", "\\"))  # normalize separators
            else:
                paths.append(s.replace("\\", "/"))  # treat backslash as separator
    if not strict:
        log.debug("find_unique_root: coerced separators", use_windows=use_windows)

    # Normalize with the chosen flavor
    paths = [pmod.normpath(s) for s in paths]
    log.debug(
        "find_unique_root: normalized paths sample",
        sample=paths[:3] if len(paths) > 3 else paths,
    )

    # Upper bound common prefix
    try:
        common_prefix = pmod.commonpath(paths)
    except ValueError:
        # e.g., different drives on Windows → no common path
        common_prefix = ""
        log.debug("find_unique_root: no common path (e.g., different drives)")

    split = [PurePath(s).parts for s in paths]
    prefix_len = len(PurePath(common_prefix).parts)
    log.debug(
        "find_unique_root: prefix info", common_prefix=common_prefix, prefix_len=prefix_len
    )

    # Try shallowest→deepest; collect all candidates
    candidates = []
    for i in range(1, prefix_len + 1):
        rels = []
        for parts in split:
            tail = parts[i:]
            rels.append(pmod.join(*tail) if tail else "")
        unique = len(rels) == len(set(rels))
        log.debug("find_unique_root: candidate check", depth=i, unique=unique)
        if unique:
            head = split[0][:i]
            chosen = pmod.normpath(pmod.join(*head))
            candidates.append((i, chosen))

    if candidates:
        if mode == "minimal":
            chosen = candidates[0][1]  # shallowest
            log.debug("find_unique_root: selected minimal root", root=chosen)
            return chosen
        elif mode == "maximal":
            chosen = candidates[-1][1] # deepest
            log.debug("find_unique_root: selected maximal root", root=chosen)
            return chosen
        else:
            raise ValueError("mode must be 'minimal' or 'maximal'")

    # Fallback — full common path (may be empty)
    fallback = pmod.normpath(common_prefix) if common_prefix else common_prefix
    log.debug("find_unique_root: fallback to common prefix", root=fallback or "(none)")
    return fallback
