import os
from collections.abc import Callable, Mapping
from typing import IO, Any, NamedTuple

from pydantic import BaseModel

PathLike = str | os.PathLike[str]
RulesLike = Mapping[str, Any] | PathLike | IO[str]


class DatasetConfig(BaseModel):
    name: str
    file_pattern: str | dict[str, str]  # path or mountpoint mapping
    exclude_pattern: str | None = None
    skip: bool = False
    derivatives_path: str | dict[str, str] | None = None

    class Config:
        extra = "allow"  # allow arbitrary extra fields for user flexibility


class Artifact(NamedTuple):
    item: Any
    writer: Callable[[str], None]  # how to save it


class NodeResult(NamedTuple):
    artifacts: dict[str, Artifact]  # Objects with writers


# In general we want at least one artifact, which should be
# an xarray DataArray with dimensions and coordinates fully populated
# and optional metadata in attrs (json-serializable)
