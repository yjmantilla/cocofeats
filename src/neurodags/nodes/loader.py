from __future__ import annotations

import hashlib
import importlib.util
import sys
from collections.abc import Iterable
from pathlib import Path
from types import ModuleType

from neurodags.loggers import get_logger

log = get_logger(__name__)

_LOADED_SOURCES: set[Path] = set()


def _unique_module_name(path: Path) -> str:
    digest = hashlib.sha1(str(path).encode("utf-8")).hexdigest()
    return f"_neurodags_nodes_{digest}"


def load_node_definitions(
    paths: Iterable[str | Path], base_dir: Path | None = None
) -> list[ModuleType]:
    """
    Load and execute node definition files.

    The modules are expected to invoke ``register_node`` during import.
    Already imported paths are skipped.
    """
    loaded_modules: list[ModuleType] = []
    base = base_dir or Path.cwd()

    for raw_path in paths:
        if isinstance(raw_path, str):
            raw_path = raw_path.strip()
        if not raw_path:
            continue

        candidate = Path(raw_path).expanduser()
        resolved = (
            (base / candidate).resolve() if not candidate.is_absolute() else candidate.resolve()
        )

        if resolved in _LOADED_SOURCES:
            log.debug("Skipping node definition; already loaded", path=str(resolved))
            continue

        if not resolved.exists():
            raise FileNotFoundError(f"Node definition file not found: {resolved}")

        module_name = _unique_module_name(resolved)
        spec = importlib.util.spec_from_file_location(module_name, resolved)
        if spec is None or spec.loader is None:
            raise ImportError(f"Could not load node definitions from {resolved}")

        module = importlib.util.module_from_spec(spec)
        # Register before exec so that intra-package imports (e.g. `import nodes_utils`
        # inside nodes_aggregation.py) resolve to the already-executing module rather
        # than re-loading it from sys.path and double-registering nodes.
        sys.modules[module_name] = module
        stem = resolved.stem
        if stem not in sys.modules:
            sys.modules[stem] = module
        log.info("Loading custom node definitions", path=str(resolved))
        spec.loader.exec_module(module)  # type: ignore[assignment]
        _LOADED_SOURCES.add(resolved)
        loaded_modules.append(module)

    return loaded_modules
