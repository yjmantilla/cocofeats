# Changelog

## Unreleased

### Fixed

- **Sub-derivative cache respected when parent has `overwrite: True`**: previously,
  a derivative with `overwrite: True` forced `cached_here = False` for all of its
  sub-derivative inputs, causing them to re-execute even when they had `overwrite: False`
  and valid cached files on disk.  The cache check now uses the *child* derivative's own
  `overwrite` flag rather than the parent's, so only the derivative that explicitly sets
  `overwrite: True` is recomputed.  (`dag.run_derivative`)

- **`{"cached": [...]}` dict no longer leaks into node arguments**: when a sub-derivative
  early-returns its internal `{"cached": [path, ...]}` sentinel (e.g. because it hit its
  own cache), the parent previously stored that dict raw in `store[sid]` and passed it
  on to downstream node functions as if it were a real value — causing `AttributeError`
  at runtime.  The parent now resolves the cached dict to the matching path string before
  storing, so downstream nodes always receive a proper path or `NodeResult`.
  (`dag.run_derivative`)

### Changed

- **`neurodags status` exit code**: now exits `1` when any derivatives are missing or errored
  (previously only errored triggered a non-zero exit). Enables use in CI and shell dependency
  chains: `neurodags status pipeline.yml || sbatch resubmit.sh`.

- **`neurodags status --format json`**: new flag emits machine-readable JSON with `config`,
  `n_files`, per-derivative counts, `grand_total`, and `complete` boolean. Useful for scripted
  post-cluster checks and quota estimation.

### Changed

- **`neurodags count` renamed to `neurodags count-inputs`**: clarifies that the command
  counts source (input) files the pipeline will process, not output files or derivative
  instances. One input file may produce multiple output files depending on the derivatives.
  All generated SLURM templates, documentation, and tests updated accordingly.

### Added

- **Dataset-level variables (`vars:`)**: dataset entries in `datasets.yml` can now
  declare a `vars:` block of arbitrary key-value pairs.  Any pipeline node arg whose
  string value matches `$identifier` is substituted with the corresponding value from
  the active dataset entry's `vars` at runtime, after `id.N` reference resolution.
  Only whole-string values are substituted — embedded `$` in paths or other strings
  is left untouched.  Variables may be any YAML type (string, int, float, bool,
  list).  Referencing an undefined variable raises `KeyError` with the list of
  available vars.  Primary use case: encoding a condition name (or any
  dataset-specific parameter) in the dataset entry so that activating a different
  entry changes both `derivatives_path` and pipeline behaviour in one step, with no
  pipeline YAML edits required.  (`definitions.DatasetConfig.vars`,
  `dag._resolve_vars`, `dag._prep_kwargs`)

- **In-memory multi-artifact selection**: when a node returns a `NodeResult` with
  multiple artifacts (e.g. a splitter that produces one artifact per condition),
  downstream derivatives can now select a specific artifact using the existing
  dot-extension syntax — `derivative: SplitterName.condA.fif` — even when the
  splitter has not yet been written to disk.  Previously this selection only worked
  for on-disk (cached) artifacts; the in-memory path passed the full `NodeResult`
  and relied on the `_unwrap_for_arg` heuristic, which returned the first matching
  artifact regardless of the requested suffix.  The fix applies the same suffix
  filter to the in-memory `NodeResult` that was already applied to on-disk
  candidates, making both paths consistent.  A warning is logged when the requested
  suffix is absent from the splitter's artifacts.  (`dag.run_derivative`)

- **Config snapshot on `neurodags run`**: before executing any derivatives, the pipeline
  YAML, `new_definitions` file(s), and datasets YAML are copied to
  `derivatives_path/code/`.  A `neurodags_env.json` file is also written with the
  installed neurodags version, git commit of the source repo (when installed from a
  checkout), and a UTC timestamp.  Skipped on dry runs; failures are warnings, never
  errors.  (`orchestrators._snapshot_pipeline_config`)

## 0.1.0

- Initial release of the template.
