from __future__ import annotations

import pkgutil
import sys
from collections.abc import Callable, Iterable
from importlib import import_module
from typing import Any

from cocofeats.loggers import get_logger

FeatureCallable = Callable[..., Any]


class FeatureEntry:
    """Bundle a feature callable with its definition."""

    def __init__(
        self,
        name: str,
        func: FeatureCallable,
        definition: dict[str, Any] | None = None,
    ):
        self.name = name
        self.func = func
        self.definition = definition or {}

    def __call__(self, *args, **kwargs):
        return self.func(*args, **kwargs)

    def __repr__(self) -> str:
        return f"<FeatureEntry name={self.name!r} func={self.func!r}>"


# Global registry
_FEATURE_REGISTRY: dict[str, FeatureEntry] = {}

log = get_logger(__name__)


# -------------------------
# Registration API
# -------------------------


def register_feature(
    func: FeatureCallable | None = None,
    *,
    name: str | None = None,
    override: bool = False,
    definition: dict[str, Any] | None = None,
) -> FeatureCallable:
    """Register a feature callable with optional definition.

    Can be used as:

        @register_feature
        def my_feature(...): ...

    or:

        @register_feature(name="alias", definition=defn)

    or programmatically with `register_feature_with_name`.
    """

    def decorator(target: FeatureCallable) -> FeatureCallable:
        key = name or target.__name__
        if not override and key in _FEATURE_REGISTRY and _FEATURE_REGISTRY[key].func is not target:
            raise ValueError(f"Feature '{key}' is already registered")
        elif override and key in _FEATURE_REGISTRY:
            log.info(f"Overriding existing feature registration for '{key}'")

        _FEATURE_REGISTRY[key] = FeatureEntry(key, target, definition)
        return target

    if func is not None:
        return decorator(func)

    return decorator


def register_feature_with_name(
    name: str,
    func: FeatureCallable,
    definition: dict[str, Any] | None = None,
    override: bool = False,
) -> None:
    """Register a feature under a specific name."""
    func.__name__ = name
    register_feature(func, name=name, definition=definition, override=override)


# -------------------------
# Lookup API
# -------------------------


def get_feature(name: str) -> FeatureEntry:
    """Return a registered feature entry by name."""
    try:
        return _FEATURE_REGISTRY[name]
    except KeyError as exc:
        available = ", ".join(sorted(_FEATURE_REGISTRY)) or "<none>"
        raise KeyError(f"Unknown feature '{name}'. Available features: {available}") from exc


def iter_features() -> Iterable[tuple[str, FeatureEntry]]:
    """Yield registered features as (name, FeatureEntry) pairs."""
    return _FEATURE_REGISTRY.items()


def list_features() -> tuple[str, ...]:
    """Return the registered feature names sorted alphabetically."""
    return tuple(sorted(_FEATURE_REGISTRY))


def clear_feature_registry() -> None:
    """Remove all registered features (useful for tests)."""
    _FEATURE_REGISTRY.clear()


def unregister_feature(name: str) -> None:
    """Remove a feature from the registry if present."""
    _FEATURE_REGISTRY.pop(name, None)


# -------------------------
# Convenience
# -------------------------

__all__ = [
    "FeatureEntry",
    "clear_feature_registry",
    "get_feature",
    "iter_features",
    "list_features",
    "register_feature",
    "register_feature_with_name",
    "unregister_feature",
]


def discover(package: str | None = None) -> None:
    """Import all modules in the package to trigger registrations."""
    package_name = package or __name__
    module = sys.modules[package_name]
    if not hasattr(module, "__path__"):
        return

    for mod_info in pkgutil.iter_modules(module.__path__):
        if mod_info.name.startswith("_"):
            continue
        import_module(f"{package_name}.{mod_info.name}")


discover()
