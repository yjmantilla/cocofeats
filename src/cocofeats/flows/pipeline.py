import yaml

from cocofeats.flows import register_flow_with_name
from cocofeats.loggers import get_logger

log = get_logger(__name__)

this_file = __file__
this_yaml = this_file.replace(".py", ".yml")

yaml_definitions = yaml.safe_load(open(this_yaml))

# Register the chain features
for name, chain in yaml_definitions.get("FeatureDefinitions", {}).items():

    def make_wrapper(chain, name=name):  # capture both in closure
        def wrapper(file_path: str, reference_base: str | None = None):
            from pathlib import Path

            from cocofeats.dag import run_flow

            return run_flow(chain, name, file_path, Path(reference_base or ""))

        return wrapper

    func = make_wrapper(chain)
    # pass chain as definition so FlowEntry stores it
    register_flow_with_name(name, func, definition=chain)
    log.info("Registered flow", name=name)
