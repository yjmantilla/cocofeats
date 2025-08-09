# Installation

```bash
pip install package_template
```

For development:

```bash
git clone https://github.com/your-org/package_template
cd package_template
python -m venv .venv
.venv\\Scripts\\activate  # Windows
# source .venv/bin/activate  # macOS/Linux
pip install -e .[dev,test,docs]
pre-commit install
```
