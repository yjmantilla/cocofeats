import os
from collections.abc import Callable, Mapping
from typing import IO, Any, NamedTuple

from pydantic import BaseModel

PathLike = str | os.PathLike[str]
RulesLike = Mapping[str, Any] | PathLike | IO[str]


class DatasetConfig(BaseModel):
    """Configuration for a dataset."""

    name: str
    file_pattern: str | dict[str, str]  # path or mountpoint mapping
    exclude_pattern: str | None = None
    drop_split_continuations: bool = True  # drop split-FIF continuation files (keep entry only)
    skip: bool = False
    derivatives_path: str | dict[str, str] | None = None
    vars: dict[str, Any] | None = (
        None  # dataset-level variables; referenced as $var_name in pipeline node args
    )

    class Config:
        extra = "allow"  # allow arbitrary extra fields for user flexibility


class Artifact(NamedTuple):
    """An artifact produced by a node, with its associated writer."""

    item: Any
    writer: Callable[[str], None]  # how to save it


class NodeResult(NamedTuple):
    """The result of a node execution."""

    artifacts: dict[str, Artifact]  # Objects with writers


class SkipDerivative(Exception):
    """Raise from a node to signal that this source file is not applicable for this derivative.

    neurodags catches this exception, writes a ``.skip`` marker file next to where the
    artifact would have been saved, and reports the derivative as **skipped** in
    ``neurodags status`` output — distinct from *missing* (never ran) or *errored*
    (failed unexpectedly).

    Skipped derivatives are not retried on subsequent runs unless the ``.skip`` marker
    is deleted manually or ``overwrite: true`` is set on the derivative.

    Typical use-case: a condition that does not exist in a particular subject's recording.
    Instead of raising a generic ``ValueError`` (which neurodags treats as an error),
    raise ``SkipDerivative`` to record that the absence is intentional::

        from neurodags.definitions import SkipDerivative

        if condition_name not in found_conditions:
            raise SkipDerivative(
                f"Condition '{condition_name}' not present in this recording."
            )

    The message is written verbatim into the ``.skip`` marker file for later inspection.
    """


# In general we want at least one artifact, which should be
# an xarray DataArray with dimensions and coordinates fully populated
# and optional metadata in attrs (json-serializable)
