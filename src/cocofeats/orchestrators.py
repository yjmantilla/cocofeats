import os
from cocofeats.datasets import get_datasets_and_mount_point_from_pipeline_configuration
from cocofeats.features.base import FeatureBase
from cocofeats.loggers import get_logger
from cocofeats.loaders import load_configuration
from cocofeats.utils import get_path, replace_bids_suffix, find_unique_root, snake_to_camel
from cocofeats.iterators import get_all_files_from_pipeline_configuration
from cocofeats.writers import save_dict_to_json
from cocofeats.definitions import Artifact, FeatureResult
from typing import Any, Callable
import xarray as xr
log = get_logger(__name__)

def iterate_feature_pipeline(
    pipeline_configuration: dict,
    feature: Callable,
    max_files_per_dataset: int | None = None,
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
        pipeline_configuration, max_files_per_dataset=max_files_per_dataset
    )

    log.info("iterate_call_pipeline: starting processing", total_files=len(all_files))

    for index, dataset_name, file_path in all_files:
        log.debug("Processing file", index=index, dataset=dataset_name, file_path=file_path)
        try:
            common_root = common_roots.get(dataset_name)
            output_path = datasets_configs[dataset_name].derivatives_path
            if output_path:
                output_path = get_path(output_path, mount_point=mount_point)
                os.makedirs(output_path, exist_ok=True)
                reference_path = os.path.relpath(file_path, start=common_root) if common_root else file_path
                reference_path = os.path.join(output_path, reference_path)
                os.makedirs(os.path.dirname(reference_path), exist_ok=True)
            else:
                reference_path = file_path

            output = feature(file_path)
            reference_path = reference_path + '@' + snake_to_camel(feature.__name__)

            # Write artifacts
            for artifact_name, artifact in output.artifacts.items():
                log.info("Processed artifact", name=artifact_name, file=artifact)
                artifact_path = reference_path + artifact_name # has the extension
                artifact.writer(artifact_path)
                log.info("Saved artifact", file=artifact_path)

            log.info("Processed file successfully", index=index, dataset=dataset_name, file_path=file_path)
        except Exception as e:
            log.error("Error processing file", index=index, dataset=dataset_name, file_path=file_path, error=str(e))

    log.info("iterate_call_pipeline: completed processing")

if __name__ == "__main__":
    from cocofeats.datasets import generate_dummy_dataset

    dataset_1 = dict(
        PATTERN = "sub-%subject%/ses-%session%/sub-%subject%_ses-%session%_task-%task%_acq-%acquisition%_run-%run%",
        DATASET = "dataset1",
        NSUBS = 2,
        NSESSIONS = 2,
        NTASKS = 2,
        NACQS = 2,
        NRUNS = 2,
        PREFIXES = dict(subject="S", session="SE", task="T", acquisition="A", run="R"),
        ROOT = "_data/dataset1",
   )

    dataset_2 = dict(
        PATTERN = "sub-%subject%/ses-%session%/sub-%subject%_ses-%session%_task-%task%_acq-%acquisition%_run-%run%",
        DATASET = "dataset2",
        NSUBS = 3,
        NSESSIONS = 1,
        NTASKS = 2,
        NACQS = 1,
        NRUNS = 2,
        PREFIXES = dict(subject="S", session="SE", task="T", acquisition="A", run="R"),
        ROOT = "_data/dataset2",
    )

    generate_dummy_dataset(data_params=dataset_1)
    generate_dummy_dataset(data_params=dataset_2)

    datasets = dict(
        dataset_1={
            "name": "Dataset 1",
            "file_pattern": "_data/dataset1/**/*.vhdr",
            "derivatives_path": "_outputs/dataset1",
            "extra_field": "extra_value",  # Example of an extra field
        },
        dataset_2={
            "name": "Dataset 2",
            "file_pattern": "_data/dataset2/**/*.vhdr",
            "derivatives_path": "_outputs/dataset2",
            "extra_field": "extra_value",  # Example of an extra field
        },
    )

    
    pipeline_input = dict(
        datasets=datasets, 
        mount_point=None,
    )
    files_per_dataset, all_files, common_roots = get_all_files_from_pipeline_configuration(
        pipeline_input, max_files_per_dataset=None
    )
    
    print("Files per dataset:", files_per_dataset)
    print("All files with indices:", all_files)
    print("Common roots:", common_roots)
    

    [print(x[2]) for x in all_files]


    
    from cocofeats.utils import replace_bids_suffix, find_unique_root

    # outputs
    outputs = [x[2].replace(".vhdr", "_report.html") for x in all_files]
    root= find_unique_root([x[2] for x in all_files], mode="minimal")
    outputs = [os.path.join("_outputs", os.path.relpath(y, start=root)) for y in outputs]



    # write files to a config.yml for snakemake
    files_dict = {
        'pairs': [{'input':x[2], 'output':y} for x,y in zip(all_files, outputs)],
        }
    import yaml
    with open("pipeline_config.yml", "w") as f:
        yaml.dump(files_dict, f)

    
    
    
    from cocofeats.features.spectral import spectrum
    from cocofeats.features.preprocessing import basic_preprocessing

    for feature in [spectrum, basic_preprocessing]:
        iterate_feature_pipeline(
            pipeline_configuration=pipeline_input,
            feature=feature,
            max_files_per_dataset=None,
        )