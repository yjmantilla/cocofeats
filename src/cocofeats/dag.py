# dag_exec.py
import re
import yaml
from typing import Any, Dict, List, Tuple
from cocofeats.features import register_feature, get_feature, register_feature_with_name
from cocofeats.definitions import FeatureResult
from cocofeats.loggers import get_logger
from cocofeats.utils import snake_to_camel
from pathlib import Path
import glob


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
def _is_step_cached(flow_name: str, step_feature_name: str, reference_base: Path) -> bool:
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
        log.error("Multiple cached artifacts found", flow=flow_name, feature=step_feature_name, reference_base=reference_base, candidates=candidates)
        raise RuntimeError(f"Multiple cached artifacts found for {reference_base}{postfix}: {candidates}")
    return len(candidates) == 1, candidates[0]

# ---------- executor ----------

class FlowRegistry:
    """Holds parsed flows and runs them recursively."""
    def __init__(self, flows: Dict[str, Dict[str, Any]]):
        self.flows = flows  # name -> {"overwrite": bool, "flow": [...]}

    def run_flow_for_file(
        self,
        flow_name: str,
        file_path: str,
        reference_base: Path,
    ) -> FeatureResult:
        """
        Evaluate a flow on a single file.
        - feature steps: if cached and overwrite=False, assume ready; otherwise compute.
        - function steps: always execute (they are primitives returning FeatureResult).
        """
        if flow_name not in self.flows:
            raise KeyError(f"Unknown flow '{flow_name}'")
        fdef = self.flows[flow_name]
        overwrite = bool(fdef.get("overwrite", False))
        flow = fdef.get("flow", [])
        order = _topo_order(flow)

        store: Dict[int, Any] = {}
        last_result: FeatureResult | None = None

        for sid in order:
            step = next(s for s in flow if s["id"] == sid)
            # 1) 'feature' step
            if "feature" in step:
                fname = step["feature"]

                if fname == "SourceFile":
                    store[sid] = file_path
                    log.debug("Flow step SourceFile", flow=flow_name, id=sid, path=file_path)
                    continue

                # feature can be another flow (composite) or a registered primitive feature
                # If cached & not overwrite: we don't need to recompute now; we store a sentinel.
                # If not cached or overwrite: compute.
                cached, candidate = _is_step_cached(flow_name, fname, reference_base) if not overwrite else (False, None)
                if cached:
                    log.debug("Using cached feature", flow=flow_name, id=sid, feature=fname)
                    # Store a lightweight marker (you can store paths if you want)
                    store[sid] = candidate # just the first candidate (see _is_step_cached)
                    last_result = candidate
                    continue
                if fname in self.flows:
                    log.debug("Recurse into sub-flow", parent=flow_name, child=fname)
                    sub_result = self.run_flow_for_file(
                        fname, file_path, reference_base
                    )
                    store[sid] = sub_result
                    last_result = sub_result
                else:
                    # primitive feature callable
                    fn = get_feature(fname)
                    kwargs = _prep_kwargs(step.get("args", {}), store)
                    log.debug("Execute primitive feature", flow=flow_name, id=sid, feature=fname, kwargs=kwargs)
                    res = fn(**kwargs)
                    if not isinstance(res, FeatureResult):
                        raise TypeError(f"Feature {fname} must return FeatureResult")
                    store[sid] = res
                    last_result = res

            # 2) 'function' step (primitive function that returns FeatureResult)
            elif "function" in step:
                func_name = step["function"]
                fn = get_feature(func_name)
                kwargs = _prep_kwargs(step.get("args", {}), store)
                log.debug("Execute function", flow=flow_name, id=sid, function=func_name, kwargs=kwargs)
                res = fn(**kwargs)
                if not isinstance(res, FeatureResult):
                    raise TypeError(f"Function {func_name} must return FeatureResult")
                store[sid] = res
                last_result = res

            else:
                raise ValueError(f"Step id={sid} must specify 'feature' or 'function'")

        if last_result is None:
            raise RuntimeError(f"Flow '{flow_name}' produced no result")
        return last_result

def register_flows_from_yaml_old(yaml_path: str) -> Tuple[FlowRegistry, List[str]]:
    with open(yaml_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    fdefs = cfg.get("FeatureDefinitions", {}) or {}
    reg = FlowRegistry(fdefs)

    # expose thin wrappers so your orchestrator can call them and name outputs with @FlowName
    registered = []
    for fname in fdefs.keys():
        def make_wrapper(flow_name: str):
            def wrapper(file_path: str) -> FeatureResult:
                return reg.run_flow_for_file(flow_name, file_path, reference_base=Path(""))
            wrapper.__name__ = fname  # critical for '@<FlowName>' naming
            return wrapper
        register_feature(fname, make_wrapper(fname))
        registered.append(fname)
        log.info("Registered flow wrapper", name=fname)

    return reg, registered


def register_flows_from_yaml(yaml_path: str) -> Tuple["FlowRegistry", List[str]]:
    with open(yaml_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    fdefs: Dict[str, Any] = cfg.get("FeatureDefinitions", {}) or {}
    reg = FlowRegistry(fdefs)

    registered: List[str] = []
    for flow_name in fdefs.keys():

        def make_wrapper(flow_name: str):
            def wrapper(file_path: str, reference_base: Path) -> FeatureResult:
                # reference_base is the orchestrator’s base path before '@<name>'
                return reg.run_flow_for_file(flow_name, file_path, reference_base)
            wrapper.__name__ = flow_name
            return wrapper

        func = make_wrapper(flow_name)
        register_feature_with_name(flow_name, func)   # ✅ use your helper
        registered.append(flow_name)
        log.info("Registered flow wrapper", name=flow_name)

    return reg, registered