# CLI Reference

NeuroDAGs installs a unified `neurodags` command with several subcommands. Every subcommand that accepts a pipeline file also accepts `-d/--datasets <path>` to override the datasets YAML defined inside the pipeline file — useful when the same pipeline is run against different dataset collections.

## Global Options

These flags apply to every subcommand and must be placed *before* the subcommand name:

```bash
neurodags --log-level WARNING run pipeline.yml       # suppress INFO output
neurodags --log-level DEBUG run pipeline.yml         # verbose output
neurodags --log-file run.jsonl run pipeline.yml      # also write logs to JSONL file
neurodags --log-level WARNING --log-file run.jsonl run pipeline.yml
```

| Flag | Default | Description |
|------|---------|-------------|
| `--log-level LEVEL` | `INFO` (or `$LOG_LEVEL`) | Console verbosity: `DEBUG`, `INFO`, `WARNING`, `ERROR` |
| `--log-file PATH` | none | Write all log events to PATH in JSONL format (one JSON object per line) |

The JSONL log file can be loaded directly as a dataframe:

```python
import pandas as pd
df = pd.read_json("run.jsonl", lines=True)
# columns: event, level, logger, timestamp, ... plus any bound context keys
```

## Validation

Load and summarise the configuration without running anything. Prints the resolved datasets and the list of derivatives that will be executed.

```bash
neurodags validate pipeline.yml              # load config, print summary
neurodags validate pipeline.yml -d alt.yml   # override datasets
```

See {doc}`pipeline_yaml` for all pipeline keys and {doc}`datasets_yaml` for dataset fields.

## Execution

Run derivatives defined in `DerivativeList`. When no `--derivative` flag is given, all derivatives in `DerivativeList` are run in dependency order.

```bash
neurodags run pipeline.yml                          # run all derivatives in DerivativeList
neurodags run pipeline.yml --derivative CleanedEEG  # run one derivative
neurodags run pipeline.yml --derivative A --derivative B  # run several
```

**Parallelism:**

```bash
neurodags run pipeline.yml --n-jobs 4          # 4 parallel workers
neurodags run pipeline.yml --n-jobs -1         # all cores
neurodags run pipeline.yml --n-jobs 4 --joblib-backend loky --joblib-prefer processes
```

**Subset / error control:**

```bash
neurodags run pipeline.yml --max-files-per-dataset 10
neurodags run pipeline.yml --only-index 0 5 12   # process only files at these indices
neurodags run pipeline.yml --skip-errors          # skip files that already have a .error marker
neurodags run pipeline.yml --raise-on-error       # stop immediately on first failure
```

See {doc}`parallelism` for full details on parallel execution and error handling.

## Dry Run

Inspect the planned execution without running any nodes. Reports each file, derivative, and whether the output is already cached. Useful for verifying the DAG before a long run, checking which files need recomputation, or debugging path issues.

```bash
neurodags dry-run pipeline.yml                             # all derivatives
neurodags dry-run pipeline.yml --derivative CleanedEEG    # one derivative
neurodags dry-run pipeline.yml --output plan.csv          # save to CSV
neurodags dry-run pipeline.yml --output plan.parquet      # or Parquet
neurodags dry-run pipeline.yml --n-jobs 4                 # parallel file resolution
neurodags dry-run pipeline.yml --skip-errors              # exclude errored files
```

See {doc}`inspection` for the plan format and how to interpret cached / missing / errored states.

## Status

Quick at-a-glance summary of done / missing / errored counts per derivative — no CSV required.

```bash
neurodags status pipeline.yml                        # summary table for all derivatives
neurodags status pipeline.yml --derivative Alpha     # filter to one derivative
neurodags status pipeline.yml --list-errors          # print errored file paths + .error file paths
neurodags status pipeline.yml --list-missing         # print paths of not-yet-computed files
neurodags status pipeline.yml --list-errors --list-missing
neurodags status pipeline.yml --n-jobs 4             # parallelize the underlying dry-run
```

Example output:

```
config: /abs/path/pipeline.yml
files:  42

Derivative               total   done  missing  errored
───────────────────────────────────────────────────────
Alpha                       42     30       10        2
Beta                        42     25       15        2
───────────────────────────────────────────────────────
Total                       84     55       25        4

2 error(s) found.  Run with --list-errors for details.
```

Exit code `0` when no errors; `1` when at least one `.error` marker exists — suitable for CI or shell scripts:

```bash
neurodags status pipeline.yml || echo "pipeline has failures"
```

See {doc}`inspection` for status definitions and `.error` marker behaviour.

## File Count

Print the total number of unique input files the pipeline will process. Useful for sanity-checking datasets before a long run.

```bash
neurodags count pipeline.yml                          # total files across all derivatives
neurodags count pipeline.yml --derivative CleanedEEG  # count for a specific derivative
```

## Dataframe Assembly

Collect derivatives marked `for_dataframe: True` into a flat CSV or Parquet file, one row per file (wide) or one row per value (long).

```bash
neurodags dataframe pipeline.yml --format wide --output features.csv
neurodags dataframe pipeline.yml --format long --output features.parquet
neurodags dataframe pipeline.yml --include-derivative PowerSpectrum --include-derivative BandPower
neurodags dataframe pipeline.yml --max-files-per-dataset 5
neurodags dataframe pipeline.yml --n-jobs 4    # parallel file-level collection
neurodags dataframe pipeline.yml --n-jobs -1   # all cores
```

Parallelism is per-file (each worker collects all derivatives for one file), so both wide and long formats work correctly. Artifact loading is IO-bound so threading is used internally.

See {doc}`dataframe_assembly` for format details and how to mark derivatives for dataframe inclusion.

## DAG Visualization

Render the pipeline or a single derivative as a Mermaid diagram. The pipeline-level view shows one node per derivative with inter-derivative edges; the derivative-level view shows every computation node inside one derivative.

```bash
neurodags dag pipeline.yml                                        # print Mermaid text to stdout
neurodags dag pipeline.yml --html pipeline_dag.html               # export to standalone HTML
neurodags dag pipeline.yml --html pipeline_dag.html --open        # export and open in browser
neurodags dag pipeline.yml --derivative CleanedEEG --html d.html  # single-derivative DAG
```

See {doc}`inspection` for a full walkthrough of DAG visualization.

## File Explorer

Launch an interactive Dash-Plotly browser for `.fif` (MNE) and `.nc` (NetCDF/xarray) files.

```bash
neurodags view path/to/file.fif   # MNE raw / epochs explorer
neurodags view path/to/file.nc    # xarray DataArray / Dataset explorer
```

Features: variable selector for multi-variable Datasets, dimension-aware slicing dropdowns, plot types: Line, Scatter, Bar, Heatmap.

See {doc}`inspection` for the full feature list.

## SLURM / HPC Scripts

Generate ready-to-submit SLURM array job scripts. Three submission patterns are available:

| Pattern | Description |
|---------|-------------|
| `per-file` (default) | One array job per pipeline run; each task processes one file across all derivatives |
| `flat` | One array job where each task is a unique (file, derivative) pair |
| `chained` | One array job per derivative, chained with `--dependency=afterok` in topological order |

```bash
neurodags slurm-script pipeline.yml                              # per-file (default)
neurodags slurm-script pipeline.yml --pattern flat
neurodags slurm-script pipeline.yml --pattern chained
neurodags slurm-script pipeline.yml --output run_array.sh        # write to file
neurodags slurm-script pipeline.yml --derivative CleanedEEG      # restrict to one derivative
```

See {doc}`hpc` for full details on each pattern and how to submit.

## TUI (Terminal User Interface)

Requires `pip install neurodags[tui]`. Provides tabs for configuration, execution, dry-run, status, dataframe assembly, DAG visualization, and file inspection — all without leaving the terminal.

```bash
neurodags tui                          # launch empty, load config interactively
neurodags tui pipeline.yml             # launch with config pre-loaded
neurodags tui pipeline.yml -d alt.yml  # with datasets override
```

See {doc}`tui` for a full walkthrough.

## Per-subcommand Dataset Override

All subcommands that take a pipeline file also accept:

| Flag | Description |
|------|-------------|
| `-d / --datasets <path>` | Override the `datasets` key in the pipeline YAML |
