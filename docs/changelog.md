# Changelog

## Unreleased

### Added

- **`SkipDerivative` exception**: nodes can now raise `SkipDerivative` to signal that a
  source file is intentionally not processable by a given derivative — distinct from an
  unexpected error. neurodags catches this, writes a `.skip` marker file alongside where
  the artifact would have been saved, and propagates the skip to all parent derivatives
  that depend on it (each writing their own `.skip` marker). Skipped derivatives are not
  retried on subsequent runs unless the `.skip` file is deleted or `overwrite: true` is
  set. **Motivation:** in multi-condition studies, some subjects may not have undergone
  every condition; without `SkipDerivative`, their missing conditions showed as *missing*
  in `neurodags status` — indistinguishable from derivatives that simply had not run yet,
  which made pipeline completion state ambiguous. (`definitions.SkipDerivative`,
  `dag.run_derivative`, `neurodags.SkipDerivative`)

- **`neurodags status` reports skipped derivatives**: the status table now includes a
  *skipped* column alongside *done*, *missing*, and *errored*. Derivatives with a `.skip`
  marker are reported as skipped — not missing — so pipeline operators can distinguish
  "will never compute for this file" from "has not run yet". The note at the bottom of
  the table explains what skipped means. JSON output (`--format json`) also includes the
  skipped count per derivative. (`cli._status_classify`, `cli._cmd_status`)

### Fixed

- **`aggregate_across_dimension` no longer silently drops non-finite values** (#19): a
  reduction such as `mean`/`std` runs with xarray's default `skipna=True`, so NaN values
  along the aggregated dimension were dropped with no record — a per-channel mean over 2
  surviving epochs looked identical downstream to one over 13, and since missingness is
  often not-at-random (e.g. failed FOOOF fits concentrating in one condition) this was a
  data-quality/leakage hazard. The node now takes `on_dropped` (`"warn"` default →
  `log.warning` with the dropped count; `"raise"` to fail fast; `"ignore"` for the old
  behaviour) and an opt-in `emit_counts` flag that attaches `n_used`/`n_dropped`
  coordinates to the aggregated array so per-value reliability is queryable. Fires only
  when a reduction actually skips NaN (float data, `skipna` in effect); pipelines with no
  non-finite values are unchanged. (`operations.aggregate_across_dimension`)

- **Split-FIF continuations no longer scanned as separate source files**: MNE splits
  recordings larger than ~2 GB across multiple `.fif` files, of which only the first
  (*entry*) file is an independent recording — the continuations are stitched back in
  transparently by `mne.io.read_raw_fif(entry)`. The file scanner used a bare
  `glob.glob`, so every continuation was returned as its own source file and the pipeline
  ran derivatives on partial data, emitting duplicate/garbage rows. The scanner now drops
  continuations while keeping the entry, for both conventions: BIDS `_split-01_` (kept) vs
  `_split-02_`+ (dropped, zero-padding aware), and plain-mne `name.fif` (kept) vs
  `name-1.fif`/`name-2.fif`+ (dropped). Detection is filename-based and cheap (no file
  reads); the plain-mne `-N` rule only fires when the entry file is also present in the
  scan, so legitimately named files such as `sub-01_..._run-2_meg.fif` are never dropped.
  Controlled by the new `drop_split_continuations` flag (default `True`) — a per-dataset
  field on `DatasetConfig` (opt out one dataset) and a parameter on
  `get_files_from_pattern` / `get_all_files_across_datasets` /
  `get_all_files_from_pipeline_configuration` (global switch). The number of dropped
  continuations is logged. (`iterators.find_split_continuations`,
  `iterators.get_files_from_pattern`, `definitions.DatasetConfig.drop_split_continuations`)

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

- **DAG HTML visualization uses ELK layout by default**: Mermaid diagrams now use the ELK
  layout engine (orthogonal edge routing, crossing minimisation) instead of dagre with bezier
  curves. Significantly cleaner for dense pipelines. Use `--layout dagre` for offline use.
  The raw Mermaid text output (`neurodags dag` without `--html`) is unchanged.

- **`neurodags count` renamed to `neurodags count-inputs`**: clarifies that the command
  counts source (input) files the pipeline will process, not output files or derivative
  instances. One input file may produce multiple output files depending on the derivatives.
  All generated SLURM templates, documentation, and tests updated accordingly.

### Added

- **`neurodags dag --layout`**: new flag for HTML DAG output selecting the layout engine.
  `elk` (default) uses orthogonal routing via ELK — requires CDN access. `dagre` uses
  right-angle step edges with no CDN dependency — suitable for offline environments.
  Also available as `layout=` in the Python API (`pipeline_to_html`, `derivative_to_html`,
  `save_mermaid_html`).

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
