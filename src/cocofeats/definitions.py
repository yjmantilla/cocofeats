from dataclasses import dataclass
import os
from typing import Any
from typing import IO, Any
from collections.abc import Mapping
from typing import Union, Dict, Optional
from pathlib import Path
from pydantic import BaseModel, validator, root_validator
from typing import Any, Dict, Callable, NamedTuple
import xarray as xr


PathLike = str | os.PathLike[str]
RulesLike = Mapping[str, Any] | PathLike | IO[str]

class DatasetConfig(BaseModel):
    name: str
    file_pattern: Union[str, Dict[str, str]]  # path or mountpoint mapping
    exclude_pattern: Optional[str] = None
    skip: bool = False
    derivatives_path: Optional[Union[str, Dict[str, str]]] = None
    
    class Config:
        extra = "allow"  # allow arbitrary extra fields for user flexibility



class Artifact(NamedTuple):
    item: Any
    writer: Callable[[str], None]   # how to save it

class FeatureResult(NamedTuple):
    artifacts: Dict[str, Artifact]            # Objects with writers

# In general we want at least one artifact, which should be
# an xarray DataArray with dimensions and coordinates fully populated
# and optional metadata in attrs (json-serializable)