# Changelog

## Unreleased

### Added

- **Config snapshot on `neurodags run`**: before executing any derivatives, the pipeline
  YAML, `new_definitions` file(s), and datasets YAML are copied to
  `derivatives_path/code/`.  A `neurodags_env.json` file is also written with the
  installed neurodags version, git commit of the source repo (when installed from a
  checkout), and a UTC timestamp.  Skipped on dry runs; failures are warnings, never
  errors.  (`orchestrators._snapshot_pipeline_config`)

## 0.1.0

- Initial release of the template.
