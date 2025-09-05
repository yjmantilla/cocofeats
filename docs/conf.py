# conf.py
import os
import subprocess
import sys
from datetime import datetime
from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as _v
from pathlib import Path

project = "cocofeats"

# Full version from installed dist (hatch-vcs stamps this)
try:
    release = _v("cocofeats")
except PackageNotFoundError:
    release = "0+unknown"

# Short X.Y
version = ".".join(release.split(".")[:2])


def _git_short() -> str:
    try:
        return (
            subprocess.check_output(
                ["git", "rev-parse", "--short", "HEAD"], stderr=subprocess.DEVNULL
            )
            .decode()
            .strip()
        )
    except Exception:
        return ""


_commit = _git_short()
html_title = f"{project} {release}" + (f" ({_commit})" if _commit else "")

# If you use a src/ layout, add it to sys.path for things like sphinx.ext.autodoc (AutoAPI doesn't need it)
sys.path.insert(0, os.path.abspath("../src"))

author = "Yorguin-José Mantilla-Ramos"
current_year = datetime.now().year
copyright = f"{current_year}, {author}"

# -- General configuration ------------------------------------------------
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.autosummary",
    #    "sphinx.ext.viewcode",
    "sphinx.ext.linkcode",
    "sphinx_autodoc_typehints",
    "myst_parser",
    "sphinx_gallery.gen_gallery",
    "autoapi.extension",
]

autosummary_generate = True
napoleon_google_docstring = False
napoleon_numpy_docstring = True

templates_path = ["_templates"]
exclude_patterns = [
    "_build",
    "Thumbs.db",
    ".DS_Store",
    # AutoAPI will still crawl ../src; these patterns are for the docs/ tree
]

# -- AutoAPI configuration -----------------------------------------------
# Point to your actual source root (adjust if your package lives elsewhere)
autoapi_type = "python"
autoapi_dirs = ["../src"]  # or ["../src/cocofeats"] to target just the package
autoapi_root = "api"  # generated top-level folder under docs
autoapi_add_toctree_entry = True  # add an "API Reference" entry automatically
autoapi_generate_api_docs = True
autoapi_member_order = "bysource"  # preserve source order
autoapi_python_use_implicit_namespaces = True  # if you have any namespace pkgs
autoapi_python_class_content = "both"  # class + __init__ docs
autoapi_keep_files = False  # useful for debugging

# Be generous so you truly get “the whole API”
autoapi_options = [
    "members",
    "undoc-members",
    "private-members",  # include _private
    "special-members",  # e.g. __call__, __iter__
    "inherited-members",
    "show-inheritance",
    "show-module-summary",
    # "imported-members",   # enable if you also want re-exported/imported names
]

# Ignore docs generation for tests/examples if desired (pattern applies to source crawl)
autoapi_ignore = [
    "*/tests/*",
    "*_test.py",
    "*/conftest.py",
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


# Adjust these to your org/repo/default branch
GITHUB_USER = "yjmantilla"
GITHUB_REPO = "cocofeats"
GITHUB_BRANCH = "main"  # or "master"


def linkcode_resolve(domain, info):
    if domain != "py":
        return None
    modname = info.get("module")
    fullname = info.get("fullname")
    if not modname:
        return None
    try:
        mod = importlib.import_module(modname)
    except Exception:
        return None
    obj = mod
    for part in (fullname or "").split("."):
        obj = getattr(obj, part, None)
        if obj is None:
            break

    try:
        fn = inspect.getsourcefile(obj) or inspect.getfile(obj)
        source, lineno = inspect.getsourcelines(obj)
    except Exception:
        try:
            fn = inspect.getsourcefile(mod) or inspect.getfile(mod)
            source, lineno = inspect.getsourcelines(mod)
        except Exception:
            return None

    # Make path relative to repo root
    fn = os.path.relpath(fn, start=Path(__file__).resolve().parents[1])

    linespec = f"#L{lineno}-L{lineno+len(source)-1}"
    return f"https://github.com/{GITHUB_USER}/{GITHUB_REPO}/blob/{GITHUB_BRANCH}/{fn}{linespec}"
