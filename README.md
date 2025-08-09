# package_template

A modern Python package template featuring:

- Modern `pyproject.toml` (PEP 621) with `hatchling`
- Source layout under `src/`
- Testing with `pytest` and coverage
- Linting with `ruff`, formatting with `black`, typing with `mypy`
- Sphinx documentation with the `furo` theme, MyST Markdown, and `sphinx-gallery`
- GitHub Actions CI for lint, type-check, tests, and docs build
- Pre-commit hooks

## Quickstart

```bash
# Clone and rename the repository
# Replace 'package_template' everywhere (folder/package names and pyproject)

# Create and activate a virtual environment
python -m venv .venv
.venv\Scripts\activate  # on Windows
# source .venv/bin/activate  # on Linux/macOS

# Install in development mode with extras
pip install -U pip
pip install -e .[dev,test,docs]

# Run quality checks
ruff check .
black --check .
mypy src
pytest -q

# Build docs
sphinx-build -b html docs docs/_build/html -W --keep-going
```

## Documentation

- Local build: `docs/_build/html/index.html`
- Hosted docs: configure GitHub Pages and set the URL in `pyproject.toml` under `[project.urls]`.

## Contributing

See [`CONTRIBUTING.md`](CONTRIBUTING.md).

## License

MIT. See [`LICENSE`](LICENSE).
