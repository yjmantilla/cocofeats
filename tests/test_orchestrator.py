from cocofeats.datasets import generate_dummy_dataset
from cocofeats.orchestrators import (
    get_all_files_from_pipeline_configuration,
    get_datasets_and_mount_point_from_pipeline_configuration,
    iterate_feature_pipeline,
)
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
