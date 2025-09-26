# dag_exec.py
import glob
import inspect
import re
from pathlib import Path
from typing import Any, Dict, List, Set

import yaml

from cocofeats.definitions import FeatureResult
from cocofeats.features import get_feature
from cocofeats.flows import get_flow, list_flows, register_flow_with_name
from cocofeats.loggers import get_logger
from cocofeats.utils import snake_to_camel

log = get_logger(__name__)
_ID_REF = re.compile(r"^id\.(\d+)$")

# ---------- helpers ----------


def _is_id_ref(x: Any) -> bool:
    return isinstance(x, str) and _ID_REF.match(x) is not None


def _resolve_refs(obj: Any, store: dict[int, Any]) -> Any:
    if _is_id_ref(obj):
        sid = int(_ID_REF.match(obj).group(1))
        if sid not in store:
            raise KeyError(f"Reference {obj} not computed yet")
        return store[sid]
    if isinstance(obj, dict):
        return {k: _resolve_refs(v, store) for k, v in obj.items()}
    if isinstance(obj, list | tuple):
        t = type(obj)
        return t(_resolve_refs(v, store) for v in obj)
    return obj


def _collect_id_refs(obj: Any) -> Set[int]:
    """Find every `id.<n>` string embedded in the given object."""
    refs: Set[int] = set()
    if isinstance(obj, str):
        match = _ID_REF.match(obj)
        if match:
            refs.add(int(match.group(1)))
        return refs
    if isinstance(obj, dict):
        for value in obj.values():
            refs.update(_collect_id_refs(value))
        return refs
    if isinstance(obj, (list, tuple, set)):
        for value in obj:
            refs.update(_collect_id_refs(value))
        return refs
    return refs


def _unwrap_for_arg(val: Any, argname: str) -> Any:
    """Heuristic: unwrap FeatureResult when passing to a primitive function."""
    if isinstance(val, FeatureResult):
        if argname in {"data", "dataset", "da", "ds"} and hasattr(val, "data"):
            return val.data
        if argname in {"mne_object", "meeg", "raw", "epochs"}:
            # Prefer a .fif artifact if present
            for k, art in val.artifacts.items():
                if k == ".fif" or k.endswith(".fif"):
                    return art.item
        # Single-artifact unwrap
        if len(val.artifacts) == 1:
            return next(iter(val.artifacts.values())).item
    return val


def _prep_kwargs(raw_kwargs: dict[str, Any], store: dict[int, Any]) -> dict[str, Any]:
    resolved = _resolve_refs(raw_kwargs or {}, store)
    return {k: _unwrap_for_arg(v, k) for k, v in resolved.items()}


def _topo_order(flow: list[dict[str, Any]]) -> list[int]:
    steps = {s["id"]: s for s in flow}
    dependencies: dict[int, Set[int]] = {}
    for sid, step in steps.items():
        declared = step.get("depends_on") or []
        deps = set(declared)
        if "args" in step:
            deps.update(_collect_id_refs(step.get("args")))
        deps.discard(sid)
        dependencies[sid] = deps
    seen, order = set(), []

    def visit(i: int):
        if i in seen:
            return
        for d in sorted(dependencies.get(i, set())):
            visit(d)
        seen.add(i)
        order.append(i)

    for i in sorted(steps):
        visit(i)
    return order


# Optional: simple cache probe (customize for your FS layout)
def _is_step_cached(
    flow_name: str, step_feature_name: str, reference_base: Path, accept_ambiguous=False
) -> bool:
    """
    Decide if a 'feature' step can be considered already computed.
    By default: look for any file whose prefix matches:
      <reference_base>@<CamelCase(step_feature_name)>.*
    """
    # prefix = f"{reference_base}@{snake_to_camel(step_feature_name)}"
    postfix = f"@{step_feature_name}"
    # return any(reference_base.parent.glob(Path(prefix).name + ".*"))
    candidates = glob.glob(reference_base.as_posix() + postfix if isinstance(reference_base, Path) else reference_base + postfix)  # or use
    if len(candidates) > 1:
        if not accept_ambiguous:
            log.error(
                "Multiple cached artifacts found",
                flow=flow_name,
                feature=step_feature_name,
                reference_base=reference_base,
                candidates=candidates,
            )
            raise RuntimeError(
                f"Multiple cached artifacts found for {reference_base}{postfix}: {candidates}"
            )
        else:
            log.warning(
                "Multiple cached artifacts found, using the first one",
                flow=flow_name,
                feature=step_feature_name,
                reference_base=reference_base,
                candidates=candidates,
            )
    return len(candidates) >= 1, candidates[0] if candidates else None


# --- registration from YAML ---
def register_flows_from_yaml(yaml_path: str) -> list[str]:
    with open(yaml_path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    flow_defs: dict[str, dict] = cfg.get("FeatureDefinitions", {}) or {}
    registered: list[str] = []

    for flow_name, flow_def in flow_defs.items():

        def make_wrapper(flow_name: str, flow_def: dict):
            def wrapper(
                file_path: str,
                reference_base: Path | None = None,
                dataset_config: Any = None,
                mount_point: Path | None = None,
            ) -> FeatureResult:
                return run_flow(
                    flow_def,
                    file_path,
                    reference_base or Path(""),
                    dataset_config=dataset_config,
                    mount_point=mount_point,
                )

            return wrapper

        func = make_wrapper(flow_name, flow_def)

        # pass the definition into the registry
        register_flow_with_name(flow_name, func, definition=flow_def)

        registered.append(flow_name)
        log.info("Registered flow", name=flow_name)

    return registered



def _artifact_candidates_for(prefix: str) -> list[str]:
    return sorted(glob.glob(prefix + ".*"))

def _split_feature_ref(feature_ref: str) -> tuple[str, str | None]:
    if "." in feature_ref:
        base, ext = feature_ref.split(".", 1)
        return base, ext
    return feature_ref, None

# ---------- executor ----------


def run_flow(
    flow_def: dict,
    flow_name: str,
    file_path: str,
    reference_base: Path,
    dataset_config: Any = None,
    mount_point: Path | None = None,
    dry_run: bool = False,
) -> FeatureResult | Dict[str, Any]:
    """
    Evaluate (or dry-run) a flow on a single file.

    When dry_run=True:
      - Do NOT execute any steps.
      - Report, per step, whether the expected artifacts are already present.
      - Return a dict with a 'plan' list (so callers don't confuse it with FeatureResult).

    Normal mode:
      - 'feature' steps: if cached and overwrite=False, skip compute; else compute or recurse.
      - 'function' steps: execute; if it's the last step, save artifacts under '@<FlowName>*'.
    """
    overwrite = bool(flow_def.get("overwrite", False))
    flow = flow_def.get("flow", [])
    order = _topo_order(flow)

    store: dict[int, Any] = {}
    last_result: FeatureResult | None = None
    plan: List[Dict[str, Any]] = [] if dry_run else None

    def _record(**kwargs):
        if plan is not None:
            plan.append(kwargs)

    # In dry-run, also check the final output of the flow upfront
    final_prefix = reference_base.as_posix() + "@" + snake_to_camel(flow_name) if isinstance(reference_base, Path) else reference_base + "@" + snake_to_camel(flow_name)
    if dry_run:
        final_candidates = _artifact_candidates_for(final_prefix)
        _record(
            id="final",
            kind="flow_output",
            name=flow_name,
            prefix=final_prefix,
            cached=len(final_candidates) > 0,
            paths=final_candidates,
        )
    else:
        # Early skip if final already cached and not overwriting
        if not overwrite:
            final_candidates = _artifact_candidates_for(final_prefix)
            if len(final_candidates) > 0:
                log.info(
                    "Last step of the flow is cached",
                    flow=flow_name,
                    final_prefix=final_prefix,
                    reference_base=reference_base,
                    candidate=final_candidates[0],
                )
                return {"cached": final_candidates}

    for sid in order:
        step = next(s for s in flow if s["id"] == sid)

        # 1) 'feature' step
        if "feature" in step:
            feature_ref = step["feature"]

            if feature_ref == "SourceFile":
                store[sid] = file_path
                _record(id=sid, kind="source", name="SourceFile", path=file_path)
                continue

            base_name, ext = _split_feature_ref(feature_ref)
            prefix = reference_base.as_posix() + "@" + snake_to_camel(base_name) if isinstance(reference_base, Path) else reference_base + "@" + snake_to_camel(base_name)
            candidates = _artifact_candidates_for(prefix)
            if ext:
                candidates = [p for p in candidates if p.endswith("." + ext)]

            if dry_run:
                entry: Dict[str, Any] = {
                    "id": sid,
                    "kind": "feature",
                    "name": feature_ref,
                    "prefix": prefix,
                    "cached": len(candidates) > 0,
                    "paths": candidates,
                }
                if base_name in list_flows():
                    sub_def = get_flow(base_name).definition
                    entry["subflow_plan"] = run_flow(
                        sub_def,
                        base_name,
                        file_path,
                        reference_base,
                        dataset_config=dataset_config,
                        mount_point=mount_point,
                        dry_run=True,
                    )
                _record(**entry)
                continue

            # normal execution path
            cached_here = (len(candidates) > 0) and not overwrite
            if cached_here:
                log.debug(
                    "Using cached feature",
                    flow=flow_name,
                    id=sid,
                    feature=feature_ref,
                    paths=candidates,
                )
                store[sid] = candidates[0]  # marker
                if len(candidates) > 1:
                    log.warning(
                        "Multiple cached artifacts found, using the first one",
                        flow=flow_name,
                        feature=feature_ref,
                        reference_base=reference_base,
                        candidates=candidates,
                        selected=candidates[0],
                    )
                last_result = {"cached": candidates}
                continue

            if base_name in list_flows():
                log.debug("Recurse into sub-flow", parent=flow_name, child=base_name)
                sub_result = run_flow(
                    get_flow(base_name).definition,
                    base_name,
                    file_path,
                    reference_base=reference_base,
                    dataset_config=dataset_config,
                    mount_point=mount_point,
                    dry_run=False,
                )
                # sub-flow may return None if already cached
                store[sid] = sub_result
                last_result = sub_result if isinstance(sub_result, FeatureResult) else last_result
            else:
                log.error("Unknown feature in flow", flow=flow_name, id=sid, feature=feature_ref)
                raise ValueError(f"Unknown feature '{feature_ref}' in flow '{flow_name}'")

        # 2) 'function' step
        elif "function" in step:
            func_name = step["function"]

            if dry_run:
                if not sid == order[-1]:
                    _record(
                        id=sid,
                        kind="function",
                        name=func_name,
                        will_compute=True,
                        is_final=(sid == order[-1]),
                    )
                continue
                # not sure what to do if not

            fn = get_feature(func_name)

            extra_args = {}
            if "reference_base" in inspect.signature(fn).parameters:
                extra_args["reference_base"] = Path(reference_base)
            if "dataset_config" in inspect.signature(fn).parameters:
                extra_args["dataset_config"] = dataset_config
            if "mount_point" in inspect.signature(fn).parameters:
                extra_args["mount_point"] = mount_point

            kwargs = _prep_kwargs(step.get("args", {}), store)
            log.debug("Execute function", flow=flow_name, id=sid, function=func_name, kwargs=kwargs)
            res = fn(**kwargs, **extra_args)
            if not isinstance(res, FeatureResult):
                raise TypeError(f"Function {func_name} must return FeatureResult")
            store[sid] = res
            last_result = res

            if sid == order[-1]:
                # save final artifacts under '@<FlowName>'
                reference_path = reference_base.as_posix() + "@" + snake_to_camel(flow_name) if isinstance(reference_base, Path) else reference_base + "@" + snake_to_camel(flow_name)
                for artifact_name, artifact in res.artifacts.items():
                    try:
                        log.info("Processed artifact", name=artifact_name, file=artifact)
                        artifact_path = reference_path + artifact_name
                        artifact.writer(artifact_path)
                        log.info("Saved artifact", file=artifact_path)
                    except Exception as e:
                        log.error(
                            "Error saving artifact",
                            artifact_name=artifact_name,
                            path=reference_path + artifact_name,
                            error=str(e),
                            exc_info=True,
                        )
                        raise

        else:
            raise ValueError(f"Step id={sid} must specify 'feature' or 'function'")

    if dry_run:
        return {
            "flow": flow_name,
            "file": file_path,
            "reference_base": reference_base.as_posix() if isinstance(reference_base, Path) else reference_base,
            "overwrite": overwrite,
            "plan": plan,
        }

    if last_result is None:
        raise RuntimeError(f"Flow '{flow_name}' produced no result")
    return last_result
