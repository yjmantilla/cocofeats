import os
import subprocess
import sys
from datetime import datetime
from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as _v

project = "cocofeats"

# Full version from installed dist (hatch-vcs stamps this)
try:
    release = _v("cocofeats")  # matches [project].name in pyproject
except PackageNotFoundError:
    release = "0+unknown"

# Short X.Y
version = ".".join(release.split(".")[:2])


# Append commit hash to the HTML title (nice for dev builds)
def _git_short():
    try:
        return (
            subprocess.check_output(
                ["git", "rev-parse", "--short", "HEAD"],
                stderr=subprocess.DEVNULL,
            )
            .decode()
            .strip()
        )
    except Exception:
        return ""


_commit = _git_short()
html_title = f"{project} {release}" + (f" ({_commit})" if _commit else "")


# Add src to sys.path for autodoc
sys.path.insert(0, os.path.abspath("../src"))

project = "cocofeats"
author = "Your Name"
current_year = datetime.now().year
copyright = f"{current_year}, {author}"

# -- General configuration ------------------------------------------------
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.autosummary",
    "sphinx.ext.viewcode",
    "sphinx_autodoc_typehints",
    "myst_parser",
    "sphinx_gallery.gen_gallery",
]

autosummary_generate = True
napoleon_google_docstring = False
napoleon_numpy_docstring = True

templates_path = ["_templates"]
exclude_patterns = [
    "_build",
    "Thumbs.db",
    ".DS_Store",
]

# -- Options for HTML output ----------------------------------------------
html_theme = "furo"
html_static_path = ["_static"]

# MyST settings
myst_enable_extensions = [
    "deflist",
    "fieldlist",
    "colon_fence",
]

# Sphinx-gallery configuration
sphinx_gallery_conf = {
    "examples_dirs": ["examples"],
    "gallery_dirs": ["auto_examples"],
    "filename_pattern": r"/plot_.*\.py$",
    "download_all_examples": False,
}
