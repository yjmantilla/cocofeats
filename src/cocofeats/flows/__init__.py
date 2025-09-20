from __future__ import annotations

from collections.abc import Callable, Iterable
from typing import Any
import sys
import pkgutil
from importlib import import_module

FlowCallable = Callable[..., Any]


class FlowEntry:
    """Bundle a flow callable with its definition."""

    def __init__(
        self,
        name: str,
        func: FlowCallable,
        definition: dict[str, Any] | None = None,
    ):
        self.name = name
        self.func = func
        self.definition = definition or {}

    def __call__(self, *args, **kwargs):
        return self.func(*args, **kwargs)

    def __repr__(self) -> str:
        return f"<FlowEntry name={self.name!r} func={self.func!r}>"


# Global registry
_FLOW_REGISTRY: dict[str, FlowEntry] = {}


# -------------------------
# Registration API
# -------------------------

def register_flow(
    func: FlowCallable | None = None,
    *,
    name: str | None = None,
    override: bool = False,
    definition: dict[str, Any] | None = None,
) -> FlowCallable:
    """Register a flow callable with optional definition.

    Can be used as:

        @register_flow
        def myflow(...): ...

    or:

        @register_flow(name="alias", definition=defn)

    or programmatically with `register_flow_with_name`.
    """

    def decorator(target: FlowCallable) -> FlowCallable:
        key = name or target.__name__
        if (
            not override
            and key in _FLOW_REGISTRY
            and _FLOW_REGISTRY[key].func is not target
        ):
            raise ValueError(f"Flow '{key}' is already registered")

        _FLOW_REGISTRY[key] = FlowEntry(key, target, definition)
        return target

    if func is not None:
        return decorator(func)

    return decorator


def register_flow_with_name(
    name: str,
    func: FlowCallable,
    definition: dict[str, Any] | None = None,
    override: bool = False,
) -> None:
    """Register a flow under a specific name."""
    func.__name__ = name
    register_flow(func, name=name, definition=definition, override=override)


# -------------------------
# Lookup API
# -------------------------

def get_flow(name: str) -> FlowEntry:
    """Return a registered flow entry by name."""
    try:
        return _FLOW_REGISTRY[name]
    except KeyError as exc:
        available = ", ".join(sorted(_FLOW_REGISTRY)) or "<none>"
        raise KeyError(
            f"Unknown flow '{name}'. Available flows: {available}"
        ) from exc


def iter_flows() -> Iterable[tuple[str, FlowEntry]]:
    """Yield registered flows as (name, FlowEntry) pairs."""
    return _FLOW_REGISTRY.items()


def list_flows() -> tuple[str, ...]:
    """Return the registered flow names sorted alphabetically."""
    return tuple(sorted(_FLOW_REGISTRY))


def clear_flow_registry() -> None:
    """Remove all registered flows (useful for tests)."""
    _FLOW_REGISTRY.clear()


def unregister_flow(name: str) -> None:
    """Remove a flow from the registry if present."""
    _FLOW_REGISTRY.pop(name, None)


# -------------------------
# Convenience
# -------------------------

__all__ = [
    "FlowEntry",
    "register_flow",
    "register_flow_with_name",
    "get_flow",
    "iter_flows",
    "list_flows",
    "clear_flow_registry",
    "unregister_flow",
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
