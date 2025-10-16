import inspect
import os
from collections.abc import Callable
from pathlib import Path
from typing import Any, List

from cocofeats.datasets import get_datasets_and_mount_point_from_pipeline_configuration
from cocofeats.iterators import get_all_files_from_pipeline_configuration
from cocofeats.loggers import get_logger
from cocofeats.utils import get_path
from cocofeats.nodes import get_node, list_nodes
from cocofeats.features import get_feature, list_features
from cocofeats.dag import collect_feature_for_dataframe, run_feature
from cocofeats.loaders import load_configuration
from cocofeats.features.pipeline import register_features_from_dict
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


    config_dict = load_configuration(pipeline_configuration) if isinstance(pipeline_configuration, str) else pipeline_configuration

    if "FeatureDefinitions" in config_dict:
        register_features_from_dict(config_dict)
    else:
        log.warning("No 'FeatureDefinitions' found in the configuration. Skipping feature registration.")
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

    # TODO: applying max_files after only_index, not sure about it though, maybe they should be incompatible
    # now apply max_files_per_dataset if specified
    if max_files_per_dataset is not None:
        filtered_files = []
        dataset_file_count = {dataset: 0 for dataset in files_per_dataset.keys()}
        for item in all_files:
            dataset_name = item[1]
            if dataset_file_count[dataset_name] < max_files_per_dataset:
                filtered_files.append(item)
                dataset_file_count[dataset_name] += 1
        all_files = filtered_files
        log.debug("iterate_call_pipeline: applied max_files_per_dataset filter", max_files_per_dataset=max_files_per_dataset, total_files=len(all_files), per_dataset=dataset_file_count)

    feature_entry = None
    node_callable: Callable | None = None
    feature_label = None

    if isinstance(feature, str):
        feature_label = feature
        if feature in list_nodes():
            node_callable = get_node(feature)
        elif feature in list_features():
            feature_entry = get_feature(feature)
        else:
            raise KeyError(f"Unknown feature or node '{feature}'")
    elif hasattr(feature, "definition") and hasattr(feature, "func"):
        feature_entry = feature
        feature_label = feature.name
    elif callable(feature):
        node_callable = feature
        feature_label = getattr(feature, "__name__", "<callable>")
    else:
        raise TypeError("feature must be a registered feature name, node name, FeatureEntry, or callable node")

    node_parameters = {}
    if node_callable is not None:
        node_parameters = inspect.signature(node_callable).parameters

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
            reference_base_path = Path(reference_base)

            if node_callable is not None:
                node_kwargs: dict[str, Any] = {}
                if "reference_base" in node_parameters:
                    node_kwargs["reference_base"] = reference_base
                if "dataset_config" in node_parameters:
                    node_kwargs["dataset_config"] = datasets_configs[dataset_name]
                if "mount_point" in node_parameters:
                    node_kwargs["mount_point"] = mount_point

                node_callable(file_path, **node_kwargs)
            else:
                try:
                    result = run_feature(
                        feature_entry.definition,
                        feature_entry.name,
                        file_path,
                        reference_base=reference_base_path,
                        dataset_config=datasets_configs[dataset_name],
                        mount_point=mount_point,
                        dry_run=dry_run,
                    )
                    if dry_run:
                        res = {
                            "index": index,
                            "dataset": dataset_name,
                            "file_path": file_path,
                        }
                        res.update(result)
                        log.info("Dry run:", **res)
                        dry_run_collection.append(res)
                except Exception as e:
                    log.error(
                        "Error running feature",
                        index=index,
                        dataset=dataset_name,
                        file_path=file_path,
                        feature=feature_label,
                        error=str(e),
                        exc_info=True,
                    )
                    if raise_on_error:
                        raise e
                    continue

            log.info(
                "Processed file successfully",
                index=index,
                dataset=dataset_name,
                file_path=file_path,
                feature=feature_label,
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


def build_feature_dataframe(
    pipeline_configuration: dict | str,
    *,
    include_features: List[str] | None = None,
    max_files_per_dataset: int | None = None,
    only_index: int | List[int] | None = None,
    raise_on_error: bool = False,
) -> pd.DataFrame:
    """
    Assemble a dataframe by collecting feature artifacts for every file in a pipeline configuration.

    Parameters
    ----------
    pipeline_configuration : dict | str
        Pipeline configuration or path to it.
    include_features : list[str], optional
        Restrict collection to this explicit subset of feature names.
    max_files_per_dataset : int, optional
        Limit the number of files processed per dataset.
    only_index : int | list[int], optional
        Restrict collection to specific indices (matching iterate_feature_pipeline behaviour).
    raise_on_error : bool, optional
        Re-raise exceptions encountered while collecting features; defaults to False.

    Returns
    -------
    pandas.DataFrame
        A dataframe with one row per processed file and columns derived from collected feature artifacts.
    """

    log.debug("build_feature_dataframe: called", pipeline_configuration=pipeline_configuration)

    config_dict = load_configuration(pipeline_configuration) if isinstance(pipeline_configuration, str) else pipeline_configuration

    if "FeatureDefinitions" not in config_dict:
        log.warning("No 'FeatureDefinitions' found in the configuration. Returning empty dataframe.")
        return pd.DataFrame()

    register_features_from_dict(config_dict)

    feature_definitions: dict[str, dict] = config_dict.get("FeatureDefinitions", {}) or {}
    selected_features: list[str] = []
    
    include_features = config_dict.get("FeatureList", []) if include_features is None else include_features

    for feature_name, feature_def in feature_definitions.items():
        if include_features is not None and feature_name not in include_features:
            continue
        if not feature_def.get("for_dataframe", True):
            continue
        selected_features.append(feature_name)
    if include_features:
        missing_features = sorted(set(include_features) - set(selected_features))
        if missing_features:
            log.warning("Some requested features are either undefined or flagged out of dataframe collection.", missing_features=missing_features)

    if not selected_features:
        log.warning("No features eligible for dataframe collection were found. Returning empty dataframe.")
        return pd.DataFrame(columns=["index", "dataset", "file_path"])

    datasets_configs, mount_point = get_datasets_and_mount_point_from_pipeline_configuration(
        pipeline_configuration
    )

    files_per_dataset, all_files, common_roots = get_all_files_from_pipeline_configuration(
        pipeline_configuration, max_files_per_dataset=max_files_per_dataset
    )
    log.debug("build_feature_dataframe: enumerated files", total_files=len(all_files), per_dataset=files_per_dataset)

    if only_index is not None:
        index_filter = only_index if isinstance(only_index, list) else [only_index]
        filtered = [item for item in all_files if item[0] in index_filter]
        if len(filtered) != len(index_filter):
            detected = [item[0] for item in filtered]
            missing = list(set(index_filter) - set(detected))
            log.warning("Some indices requested for dataframe collection were not found.", requested=index_filter, missing=missing)
        all_files = filtered

    rows: list[dict[str, Any]] = []
    for index, dataset_name, file_path in all_files:
        try:
            dataset_config = datasets_configs[dataset_name]
            common_root = common_roots.get(dataset_name)
            reference_base = _build_reference_base(file_path, dataset_config, common_root, mount_point)
            row: dict[str, Any] = {
                "index": index,
                "dataset": dataset_name,
                "file_path": file_path,
            }
            for feature_name in selected_features:
                feature_def = feature_definitions.get(feature_name, {}) or {}
                try:
                    row.update(
                        collect_feature_for_dataframe(
                            feature_def,
                            feature_name,
                            file_path,
                            reference_base=reference_base,
                            dataset_config=dataset_config,
                            mount_point=mount_point,
                        )
                    )
                except Exception as feature_error:
                    log.error(
                        "Error collecting feature for dataframe",
                        feature=feature_name,
                        index=index,
                        dataset=dataset_name,
                        file_path=file_path,
                        error=str(feature_error),
                        exc_info=True,
                    )
                    if raise_on_error:
                        raise
                    row[f"{feature_name}__error"] = str(feature_error)
            rows.append(row)
            log.debug(
                "Collected dataframe row",
                index=index,
                dataset=dataset_name,
                file_path=file_path,
                collected_features=len(selected_features),
            )
        except Exception as error:
            log.error(
                "Error collecting dataframe row",
                index=index,
                dataset=dataset_name,
                file_path=file_path,
                error=str(error),
                exc_info=True,
            )
            if raise_on_error:
                raise

    return pd.DataFrame(rows)


def _build_reference_base(file_path: str, dataset_config, common_root: str | None, mount_point: Path | None) -> Path:
    derivatives_path = dataset_config.derivatives_path
    if derivatives_path:
        derivatives_path = get_path(derivatives_path, mount_point=mount_point)
        os.makedirs(derivatives_path, exist_ok=True)
        reference_base = os.path.relpath(file_path, start=common_root) if common_root else file_path
        reference_base = os.path.join(derivatives_path, reference_base)
        os.makedirs(os.path.dirname(reference_base), exist_ok=True)
    else:
        reference_base = file_path
    return Path(reference_base)

if __name__ == "__main__":
    # Parse args and run iterate_feature_pipeline
    import argparse
    parser = argparse.ArgumentParser(description="Iterate feature pipeline.")
    parser.add_argument("config", type=str, help="Path to the pipeline configuration file (YAML or JSON).")
    parser.add_argument("--max_files_per_dataset", type=int, default=None, help="Maximum number of files to process per dataset.")
    parser.add_argument("--dry_run", action="store_true", help="If set, perform a dry run without actual processing.")
    parser.add_argument("--only_index", type=int, nargs='*', default=None, help="Only process files with these indices.")
    parser.add_argument("--raise_on_error", action="store_true", help="If set, raise exceptions on errors instead of logging them.")
    parser.add_argument("--make_final_dataframe", action="store_true", help="If set, build the final feature dataframe after processing.")
    parser.add_argument(
        "--dataframe_output",
        type=str,
        default=None,
        help="Optional path where the dataframe should be written (CSV by default, or Parquet if the extension is .parquet).",
    )
    parser.add_argument(
        "--dataframe_features",
        nargs="*",
        default=None,
        help="Optional subset of feature names to include when building the dataframe.",
    )
    args = parser.parse_args()

    from cocofeats.loaders import load_configuration
    pipeline_configuration = load_configuration(args.config)

    feature_list = pipeline_configuration.get("FeatureList", [])
    if not feature_list:
        log.error("No features specified in the pipeline configuration under 'FeatureList'. Exiting.")
        exit(1)

    if not args.make_final_dataframe:  # and args.dataframe_output:
        for feature in feature_list:
            iterate_feature_pipeline(
                pipeline_configuration=pipeline_configuration,
                feature=feature,
                max_files_per_dataset=args.max_files_per_dataset,
                dry_run=args.dry_run,
                only_index=args.only_index,
                raise_on_error=True if args.raise_on_error else False,
            )

    if args.make_final_dataframe:
        log.info("Building feature dataframe", features=args.dataframe_features)
        dataframe = build_feature_dataframe(
            pipeline_configuration=pipeline_configuration,
            include_features=args.dataframe_features,
            max_files_per_dataset=args.max_files_per_dataset,
            only_index=args.only_index,
            raise_on_error=True if args.raise_on_error else False,
        )
        if args.dataframe_output:
            output_path = Path(args.dataframe_output)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            if output_path.suffix.lower() == ".parquet":
                try:
                    dataframe.to_parquet(output_path, index=False)
                except (ImportError, ValueError) as parquet_error:
                    log.warning(
                        "Parquet export failed; saving as CSV instead.",
                        path=str(output_path),
                        error=str(parquet_error),
                    )
                    csv_fallback = output_path.with_suffix(".csv")
                    dataframe.to_csv(csv_fallback, index=False)
                    log.info("Saved feature dataframe", path=str(csv_fallback), rows=len(dataframe), columns=list(dataframe.columns))
                else:
                    log.info("Saved feature dataframe", path=str(output_path), rows=len(dataframe), columns=list(dataframe.columns))
            else:
                dataframe.to_csv(output_path, index=False)
                log.info("Saved feature dataframe", path=str(output_path), rows=len(dataframe), columns=list(dataframe.columns))
        else:
            log.info("Feature dataframe built (not saved to disk)", rows=len(dataframe), columns=list(dataframe.columns))
            print(dataframe)
