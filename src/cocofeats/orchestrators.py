import inspect
import os
from collections.abc import Callable
from typing import Any, List

from cocofeats.datasets import get_datasets_and_mount_point_from_pipeline_configuration
from cocofeats.iterators import get_all_files_from_pipeline_configuration
from cocofeats.loggers import get_logger
from cocofeats.utils import get_path
from cocofeats.features import get_feature, list_features
from cocofeats.flows import get_flow, list_flows
from cocofeats.dag import run_flow
import pandas as pd
log = get_logger(__name__)


def iterate_feature_pipeline(
    pipeline_configuration: dict,
    feature: Callable | str,
    max_files_per_dataset: int | None = None,
    dry_run: bool = False,
    only_index: int | List[int] | None = None,
    raise_on_error: bool = False,
) -> None:
    """
    Iterate over all files specified in the pipeline configuration and call a given function on each file.

    Parameters
    ----------
    pipeline_configuration : dict
        The pipeline configuration containing dataset information.
    function_to_call : callable
        The function to call for each file. It should accept at least a 'file_path' argument.
    max_files_per_dataset : int, optional
        Maximum number of files to process per dataset. If None, processes all files.
    **function_kwargs : dict
        Additional keyword arguments to pass to the function being called.

    Returns
    -------
    None
    """
    log.debug("iterate_call_pipeline: called", pipeline_configuration=pipeline_configuration)

    datasets_configs, mount_point = get_datasets_and_mount_point_from_pipeline_configuration(
        pipeline_configuration
    )

    files_per_dataset, all_files, common_roots = get_all_files_from_pipeline_configuration(
        pipeline_configuration, max_files_per_dataset=None #max_files_per_dataset, to obtain all files and then filter by only_index
    )

    log.info("iterate_call_pipeline: starting processing", total_files=len(all_files))

    dry_run_collection = []
    if only_index is not None:
        all_files = [item for item in all_files if item[0] in (only_index if isinstance(only_index, list) else [only_index])]
        if len(all_files) != len(only_index if isinstance(only_index, list) else [only_index]):
            log.warning("Some specified indices in only_index were not found in the files to process.", requested_indices=only_index, found_indices=[item[0] for item in all_files])
            log.warning("Missing indices will be ignored.", missing_indices=list(set(only_index if isinstance(only_index, list) else [only_index]) - set([item[0] for item in all_files])))
            log.warning("Proceeding with available indices.")

    for index, dataset_name, file_path in all_files:
        log.debug("Processing file", index=index, dataset=dataset_name, file_path=file_path)
        try:
            common_root = common_roots.get(dataset_name)
            derivatives_path = datasets_configs[dataset_name].derivatives_path
            if derivatives_path:
                derivatives_path = get_path(derivatives_path, mount_point=mount_point)
                os.makedirs(derivatives_path, exist_ok=True)
                reference_base = (
                    os.path.relpath(file_path, start=common_root) if common_root else file_path
                )
                reference_base = os.path.join(derivatives_path, reference_base)
                os.makedirs(os.path.dirname(reference_base), exist_ok=True)
            else:
                reference_base = file_path
            extra_kwargs = {}

            #feature(file_path, **extra_kwargs)
            # if isinstance(feature, Callable):
            #     feature(file_path, **extra_kwargs)
            if isinstance(feature, str):
                if feature in list_features():
                    feature = get_feature(feature)
                elif feature in list_flows():
                    feature = get_flow(feature)


            if hasattr(feature, "func"):
                signature = inspect.signature(feature.func).parameters
            else:
                signature = inspect.signature(feature).parameters
            if "reference_base" in signature:
                extra_kwargs["reference_base"] = reference_base
            if "dataset_config" in signature:
                extra_kwargs["dataset_config"] = datasets_configs[dataset_name]
            if "mount_point" in signature:
                extra_kwargs["mount_point"] = mount_point

            if feature.name in list_features():
                feature(file_path, **extra_kwargs)
            elif feature.name in list_flows():

                result = run_flow(feature.definition, feature.name, file_path, **extra_kwargs, dry_run=dry_run)
                res = {}
                res['index'] = index
                res['dataset'] = dataset_name
                res['file_path'] = file_path
                res.update(result)
                if dry_run:
                    log.info("Dry run:", **res)
                    dry_run_collection.append(res)


            log.info(
                "Processed file successfully",
                index=index,
                dataset=dataset_name,
                file_path=file_path,
            )
        except Exception as e:
            log.error(
                "Error processing file",
                index=index,
                dataset=dataset_name,
                file_path=file_path,
                error=str(e),
                exc_info=True,
            )
            if raise_on_error:
                raise e

    log.info("iterate_call_pipeline: completed processing")
    if dry_run:
        return pd.DataFrame(dry_run_collection)

if __name__ == "__main__":
    from cocofeats.datasets import generate_dummy_dataset

    dataset_1 = {
        "PATTERN": "sub-%subject%/ses-%session%/sub-%subject%_ses-%session%_task-%task%_acq-%acquisition%_run-%run%",
        "DATASET": "dataset1",
        "NSUBS": 2,
        "NSESSIONS": 2,
        "NTASKS": 2,
        "NACQS": 2,
        "NRUNS": 2,
        "PREFIXES": {"subject": "S", "session": "SE", "task": "T", "acquisition": "A", "run": "R"},
        "ROOT": "_data/dataset1",
    }

    dataset_2 = {
        "PATTERN": "sub-%subject%/ses-%session%/sub-%subject%_ses-%session%_task-%task%_acq-%acquisition%_run-%run%",
        "DATASET": "dataset2",
        "NSUBS": 3,
        "NSESSIONS": 1,
        "NTASKS": 2,
        "NACQS": 1,
        "NRUNS": 2,
        "PREFIXES": {"subject": "S", "session": "SE", "task": "T", "acquisition": "A", "run": "R"},
        "ROOT": "_data/dataset2",
    }

    generate_dummy_dataset(data_params=dataset_1)
    generate_dummy_dataset(data_params=dataset_2)

    datasets = {
        "dataset_1": {
            "name": "Dataset 1",
            "file_pattern": "_data/dataset1/**/*.vhdr",
            "derivatives_path": "_outputs/dataset1",
            "extra_field": "extra_value",  # Example of an extra field
        },
        "dataset_2": {
            "name": "Dataset 2",
            "file_pattern": "_data/dataset2/**/*.vhdr",
            "derivatives_path": "_outputs/dataset2",
            "extra_field": "extra_value",  # Example of an extra field
        },
    }

    pipeline_input = {
        "datasets": datasets,
        "mount_point": None,
    }
    files_per_dataset, all_files, common_roots = get_all_files_from_pipeline_configuration(
        pipeline_input, max_files_per_dataset=None
    )

    print("Files per dataset:", files_per_dataset)
    print("All files with indices:", all_files)
    print("Common roots:", common_roots)

    # Now run the feature extraction
    from cocofeats.features import get_feature, list_features

    print("Registered features:", list_features())

    if False:
        for feature in ["spectrum", "basic_preprocessing"]:
            iterate_feature_pipeline(
                pipeline_configuration=pipeline_input,
                feature=get_feature(feature),
                max_files_per_dataset=2,
            )

    # 1) Load and register flows
    # 2) Use your existing iterate_feature_pipeline per flow name
    from cocofeats.features import get_feature
    from cocofeats.flows import get_flow

    # "BasicPrep1", "CheckLineFrequency",
    # "BasicPrep1", "CheckLineFrequency", "InterFeatureDependence",
    for flow_name in ["SpectrumArrayWelch", "SpectrumArrayMultitaper",]:
        df = iterate_feature_pipeline(
            pipeline_configuration=pipeline_input,
            feature=flow_name,  # the thin wrapper
            max_files_per_dataset=None,
            dry_run = False,
            only_index = [3,5],
            raise_on_error = True,
        )
