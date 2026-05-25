# datasets.yml Reference

`datasets.yml` decouples data sources from processing logic, so the same dataset definitions can be reused across multiple pipelines.

## Structure

```yaml
<dataset_key>:
  name: <string>
  file_pattern: <string or mount-point map>
  derivatives_path: <string or mount-point map>
  exclude_pattern: <glob string>             # optional
  skip: <bool>                               # optional, default False
  vars: {<key>: <value>, ...}               # optional, dataset-level variables
```

## Fields

### `name`

Human-readable dataset identifier. Used in logging and dataframe output.

### `file_pattern`

Glob pattern for finding source files. Supports mount-point maps for environment-specific paths:

```yaml
# Single path
file_pattern: data/**/*.vhdr

# Environment-specific (mount points)
file_pattern:
  local: data/**/*.vhdr
  hpc: /cluster/BIDS/**/*.vhdr
```

The active mount point is set by `mount_point` in `pipeline.yml`.

### `derivatives_path`

Root directory where derivative files are written. Mirrors the source file's subdirectory structure under this root:

```yaml
derivatives_path:
  local: outputs/
  hpc: /cluster/scratch/out
```

### `exclude_pattern`

Optional glob to exclude files matching the pattern from processing.

### `skip`

Set `skip: True` to temporarily disable a dataset without removing it.

### `vars`

Dataset-level variables that are substituted into pipeline node args at runtime. Any string arg value matching `$identifier` is replaced by the corresponding value from `vars`.

```yaml
eeg_adhd_epilepsy_EO_baseline:
  name: eeg_adhd_epilepsy_EO_baseline
  file_pattern:
    local: data/**/*@CleanedPrepRaw.fif
  derivatives_path:
    local: outputs/features_conditions/EO_baseline
  vars:
    condition_name: EO_baseline
    epoch_duration: 2.0
```

Then in the pipeline YAML:

```yaml
CleanedPrep:
  save: False
  nodes:
    - id: 0
      derivative: SourceFile
    - id: 1
      node: extract_condition_epochs
      args:
        condition_name: $condition_name   # → "EO_baseline" from the active dataset entry
        epoch_duration: $epoch_duration   # → 2.0
```

**Substitution rules:**
- Only whole-string values are substituted: `$condition_name` → substituted; `/path/$HOME/file` → untouched.
- Variables may be any YAML type: string, int, float, bool, list.
- Referencing an undefined variable raises a `KeyError` at runtime with the list of available vars.
- `vars` only applies to node `args`; it has no effect on `derivative:` step references or other YAML keys.

**Primary use case — condition-per-dataset-entry**: Each dataset entry encodes a different condition. The active entry drives both the `derivatives_path` (where features are written) and the `$condition_name` arg (which condition to epoch). Activating a different entry changes both in one step, with no pipeline YAML edits needed.

```yaml
# step-1_dataset.yml
eeg_adhd_epilepsy_EO_baseline:
  vars: { condition_name: EO_baseline }
  derivatives_path: { local: outputs/features/EO_baseline }
  ...

eeg_adhd_epilepsy_EC_baseline:
  skip: true
  vars: { condition_name: EC_baseline }
  derivatives_path: { local: outputs/features/EC_baseline }
  ...
```

**Other uses**: multi-site studies (different `line_freq` per site), cohort-specific parameters (different `epoch_duration` for pediatric vs adult), or any parameter that logically belongs to the dataset rather than the pipeline logic.

## Multiple Datasets

List any number of datasets. All are processed when running `run_pipeline`:

```yaml
dataset1:
  name: StudyA
  file_pattern:
    local: data/studyA/**/*.vhdr
  derivatives_path:
    local: outputs/studyA/

dataset2:
  name: StudyB
  file_pattern:
    local: data/studyB/**/*.vhdr
  derivatives_path:
    local: outputs/studyB/
```

## Referencing in pipeline.yml

```yaml
# pipeline.yml
datasets: datasets.yml    # path to datasets file
mount_point: local        # which mount point to activate
```

`datasets` can be an absolute or relative path. Relative paths are resolved from the pipeline YAML location.
