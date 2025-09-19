from __future__ import annotations

from collections.abc import Callable, Iterable
from importlib import import_module
import pkgutil
import sys
from typing import Any

FeatureCallable = Callable[..., Any]

_FEATURE_REGISTRY: dict[str, FeatureCallable] = {}


def register_feature(
    func: FeatureCallable | None = None,
    *,
    name: str | None = None,
    override: bool = False,
) -> FeatureCallable:
    """Register a callable feature factory.

    Can be used as ``@register_feature`` or ``@register_feature(name="alias")``.
    """

    def decorator(target: FeatureCallable) -> FeatureCallable:
        key = name or target.__name__
        if not override and key in _FEATURE_REGISTRY and _FEATURE_REGISTRY[key] is not target:
            raise ValueError(f"Feature '{key}' is already registered")
        _FEATURE_REGISTRY[key] = target
        return target

    if func is not None:
        return decorator(func)

    return decorator


def get_feature(name: str) -> FeatureCallable:
    """Return a registered feature by name."""
    try:
        return _FEATURE_REGISTRY[name]
    except KeyError as exc:
        available = ", ".join(sorted(_FEATURE_REGISTRY)) or "<none>"
        raise KeyError(f"Unknown feature '{name}'. Available features: {available}") from exc


def iter_features() -> Iterable[tuple[str, FeatureCallable]]:
    """Yield registered features as ``(name, callable)`` pairs."""
    return _FEATURE_REGISTRY.items()


def list_features() -> tuple[str, ...]:
    """Return the registered feature names sorted alphabetically."""
    return tuple(sorted(_FEATURE_REGISTRY))


def clear_feature_registry() -> None:
    """Remove all registered features (primarily useful for tests)."""
    _FEATURE_REGISTRY.clear()


def unregister_feature(name: str) -> None:
    """Remove a feature from the registry if present."""
    _FEATURE_REGISTRY.pop(name, None)


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

def register_feature_with_name(name: str, func: Callable) -> None:
    func.__name__ = name
    register_feature(func)  # reuse existing path

# Eagerly discover features so registry is populated on package import.
discover()

__all__ = [
    "register_feature",
    "get_feature",
    "iter_features",
    "list_features",
    "clear_feature_registry",
    "unregister_feature",
    "discover",
    "register_feature_with_name",
]
