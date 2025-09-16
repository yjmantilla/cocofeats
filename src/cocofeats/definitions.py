from dataclasses import dataclass
import os
from typing import Any
from typing import IO, Any
from collections.abc import Mapping


PathLike = str | os.PathLike[str]
RulesLike = Mapping[str, Any] | PathLike | IO[str]

@dataclass
class DatasetConfig:
    label: str
    pattern: str | dict[str, str] # Path or Mount Point to Path mapping
    exclude_filter: str | None = None
    skip: bool = False
