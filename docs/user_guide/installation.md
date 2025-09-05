# Installation

```bash
pip install cocofeats
```

For development:

```bash
git clone https://github.com/your-org/cocofeats
cd cocofeats
python -m venv .venv
.venv\\Scripts\\activate  # Windows
# source .venv/bin/activate  # macOS/Linux
pip install -e .[dev,test,docs]
pre-commit install
```
