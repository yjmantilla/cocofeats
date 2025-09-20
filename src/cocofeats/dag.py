# dag_exec.py
import re
import yaml
from typing import Any, Dict, List, Tuple
from cocofeats.flows import register_flow_with_name, get_flow, list_flows
from cocofeats.features import register_feature, get_feature, register_feature_with_name
from cocofeats.definitions import FeatureResult
from cocofeats.loggers import get_logger
from cocofeats.utils import snake_to_camel
from pathlib import Path
import glob
import inspect

log = get_logger(__name__)
_ID_REF = re.compile(r"^id\.(\d+)$")

# ---------- helpers ----------

def _is_id_ref(x: Any) -> bool:
    return isinstance(x, str) and _ID_REF.match(x) is not None

def _resolve_refs(obj: Any, store: Dict[int, Any]) -> Any:
    if _is_id_ref(obj):
        sid = int(_ID_REF.match(obj).group(1))
        if sid not in store:
            raise KeyError(f"Reference {obj} not computed yet")
        return store[sid]
    if isinstance(obj, dict):
        return {k: _resolve_refs(v, store) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        t = type(obj)
        return t(_resolve_refs(v, store) for v in obj)
    return obj

def _unwrap_for_arg(val: Any, argname: str) -> Any:
    """Heuristic: unwrap FeatureResult when passing to a primitive function."""
    if isinstance(val, FeatureResult):
        if argname in {"data","dataset","da","ds"} and hasattr(val, "data"):
            return val.data
        if argname in {"mne_object","meeg","raw","epochs"}:
            # Prefer a .fif artifact if present
            for k, art in val.artifacts.items():
                if k == ".fif" or k.endswith(".fif"):
                    return art.item
        # Single-artifact unwrap
        if len(val.artifacts) == 1:
            return next(iter(val.artifacts.values())).item
    return val

def _prep_kwargs(raw_kwargs: Dict[str, Any], store: Dict[int, Any]) -> Dict[str, Any]:
    resolved = _resolve_refs(raw_kwargs or {}, store)
    return {k: _unwrap_for_arg(v, k) for k, v in resolved.items()}

def _topo_order(flow: List[Dict[str, Any]]) -> List[int]:
    steps = {s["id"]: s for s in flow}
    seen, order = set(), []
    def visit(i: int):
        if i in seen:
            return
        for d in steps[i].get("depends_on", []) or []:
            visit(d)
        seen.add(i)
        order.append(i)
    for i in sorted(steps):
        visit(i)
    return order

# Optional: simple cache probe (customize for your FS layout)
def _is_step_cached(flow_name: str, step_feature_name: str, reference_base: Path, accept_ambiguous=False) -> bool:
    """
    Decide if a 'feature' step can be considered already computed.
    By default: look for any file whose prefix matches:
      <reference_base>@<CamelCase(step_feature_name)>.*
    """
    #prefix = f"{reference_base}@{snake_to_camel(step_feature_name)}"
    postfix = f"@{step_feature_name}"
    #return any(reference_base.parent.glob(Path(prefix).name + ".*"))
    candidates = glob.glob(reference_base.as_posix() + postfix) # or use
    if len(candidates) > 1:
        if not accept_ambiguous:
            log.error("Multiple cached artifacts found", flow=flow_name, feature=step_feature_name, reference_base=reference_base, candidates=candidates)
            raise RuntimeError(f"Multiple cached artifacts found for {reference_base}{postfix}: {candidates}")
        else:
            log.warning("Multiple cached artifacts found, using the first one", flow=flow_name, feature=step_feature_name, reference_base=reference_base, candidates=candidates)
    return len(candidates) >= 1, candidates[0] if candidates else None


# --- registration from YAML ---
def register_flows_from_yaml(yaml_path: str) -> list[str]:
    with open(yaml_path, "r", encoding="utf-8") as f:
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

# ---------- executor ----------


def run_flow(
    flow_def: dict,
    flow_name: str,
    file_path: str,
    reference_base: Path,
    dataset_config: Any = None,
    mount_point: Path | None = None,
) -> FeatureResult:
    """
    Evaluate a flow on a single file.
    - feature steps: if cached and overwrite=False, assume ready; otherwise compute.
    - function steps: always execute (they are primitives returning FeatureResult).
    """
    overwrite = bool(flow_def.get("overwrite", False))
    flow = flow_def.get("flow", [])
    order = _topo_order(flow)

    store: Dict[int, Any] = {}
    last_result: FeatureResult | None = None

    # check if last step is a feature and cached
    if not overwrite:
        last_step = flow[-1]
        cached, candidate = _is_step_cached(flow_name,flow_name+'.*', reference_base, accept_ambiguous=True)
        if cached:
            log.info("Last step of the flow is cached", flow=flow_name, reference_base=reference_base, candidate=candidate)
            # Return a lightweight marker (you can return paths if you want)
            return None

    for sid in order:
        step = next(s for s in flow if s["id"] == sid)
        # 1) 'feature' step
        if "feature" in step:
            feature_name = step["feature"]

            if feature_name == "SourceFile":
                store[sid] = file_path
                log.debug("Flow step SourceFile", flow=flow_name, id=sid, path=file_path)
                continue

            # feature can be another flow (composite) or a registered primitive feature
            # If cached & not overwrite: we don't need to recompute now; we store a sentinel.
            # If not cached or overwrite: compute.
            cached, candidate = _is_step_cached(flow_name, feature_name, reference_base) if not overwrite else (False, None)
            if cached:
                log.debug("Using cached feature", flow=flow_name, id=sid, feature=feature_name)
                # Store a lightweight marker (you can store paths if you want)
                store[sid] = candidate # just the first candidate (see _is_step_cached)
                last_result = candidate
                continue

            if '.' in feature_name:
                feature_name = feature_name.split('.')[0] # when dependency is given as FeatureName.ext, the feature itself is 'FeatureName'

            if feature_name in list_flows():


                log.debug("Recurse into sub-flow", parent=flow_name, child=feature_name)
                sub_result = run_flow(
                    get_flow(feature_name).definition,
                    feature_name,
                    file_path,
                    reference_base=reference_base.as_posix(),  # pass as str
                    dataset_config=dataset_config,
                    mount_point=mount_point,
                )
                store[sid] = sub_result
                last_result = sub_result

                reference_path = reference_base.as_posix() + '@' + snake_to_camel(feature_name)

                for artifact_name, artifact in sub_result.artifacts.items():
                    log.info("Processed artifact", name=artifact_name, file=artifact)
                    artifact_path = reference_path + artifact_name # has the extension
                    artifact.writer(artifact_path)
                    log.info("Saved artifact", file=artifact_path)

            else:
                log.error("Unknown feature in flow", flow=flow_name, id=sid, feature=feature_name)
                raise ValueError(f"Unknown feature '{feature_name}' in flow '{flow_name}'")

        # 2) 'function' step (primitive function that returns FeatureResult)
        elif "function" in step:

            func_name = step["function"]
            reference_path = reference_base.as_posix() if isinstance(reference_base, Path) else reference_base + '@' + snake_to_camel(func_name)

            extra_args = {}


            fn = get_feature(func_name)


            if 'reference_base' in inspect.signature(fn).parameters:
                extra_args['reference_base'] = Path(reference_base)
            if 'dataset_config' in inspect.signature(fn).parameters:
                extra_args['dataset_config'] = dataset_config
            if 'mount_point' in inspect.signature(fn).parameters:
                extra_args['mount_point'] = mount_point


            kwargs = _prep_kwargs(step.get("args", {}), store)
            log.debug("Execute function", flow=flow_name, id=sid, function=func_name, kwargs=kwargs)
            res = fn(**kwargs, **extra_args)
            if not isinstance(res, FeatureResult):
                raise TypeError(f"Function {func_name} must return FeatureResult")
            store[sid] = res
            last_result = res

            # check if we are in the last step, in that case its saved
            if sid == order[-1]:
                reference_path = reference_base.as_posix() + '@' + flow_name
                for artifact_name, artifact in res.artifacts.items():
                    log.info("Processed artifact", name=artifact_name, file=artifact)
                    artifact_path = reference_path + artifact_name # has the extension
                    artifact.writer(artifact_path)
                    log.info("Saved artifact", file=artifact_path)

        else:
            raise ValueError(f"Step id={sid} must specify 'feature' or 'function'")

    if last_result is None:
        raise RuntimeError(f"Flow '{flow_name}' produced no result")
    return last_result

