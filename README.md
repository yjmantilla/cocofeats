# cocofeats

A slurm friendly MEEG feature extraction package leveraging bids-like data organization and DAG processing.

- Bids-like data organization
- Slurm friendly
- Reusage of existing derivatives through DAG processing.
- Yaml configuration

## Quickstart

```bash
# Clone and rename the repository

# Create and activate a virtual environment
python -m venv .venv
.venv\Scripts\activate  # on Windows
# source .venv/bin/activate  # on Linux/macOS

# Install in development mode with extras
pip install -U pip
pip install -e .[dev,test,docs]

# Run quality checks
ruff check src/ (fix with `ruff check src/ --fix` if needed)
black --check . (fix with `black .` if needed)
pytest -q

# To debug pytest, use:
pytest -q --pdb
pytest -s -q --no-cov --pdb

# Build docs
sphinx-build -b html docs docs/_build/html -W --keep-going

# Clean docs
sphinx-build -M clean docs docs/_build/html

or

rm -rf docs/_build

```

## Documentation

- Local build: `docs/_build/html/index.html`
- Hosted docs: configure GitHub Pages and set the URL in `pyproject.toml` under `[project.urls]`.
- [Docs](https://yjmantilla.github.io/cocofeats/)

## Contributing

See [`CONTRIBUTING.md`](CONTRIBUTING.md).

## License

MIT. See [`LICENSE`](LICENSE).
