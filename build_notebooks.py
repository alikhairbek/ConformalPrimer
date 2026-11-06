"""
build_notebooks.py -- generate the notebooks and conformal_toolbox.py from their sources

The notebook sources are kept as plain Python files in "percent" format
(`# %%` starts a code cell, `# %% [markdown]` a text cell) in nbsrc/.  This script

  1. writes the reference numbers of the last recorded run (reference_results.json) into primer.py,
  2. embeds primer.py in the first code cell of every notebook, so that each notebook also runs
     on its own (Colab, Kaggle) without the repository,
  3. writes NN_name.ipynb next to primer.py,
  4. writes ConformalPrimer_all_in_one.ipynb, which runs the six notebooks in sequence and
     packs everything they produce into one zip file, and
  5. writes conformal_toolbox.py, a copy of the toolbox section of primer.py that needs only NumPy/SciPy.

Usage:  python build_notebooks.py
"""

import ast
import glob
import json
import os
import re

import nbformat
from nbformat.v4 import new_code_cell, new_markdown_cell, new_notebook

HERE = os.path.dirname(os.path.abspath(__file__))
BEGIN = "# >>> reference results (written by build_notebooks.py -- do not edit by hand)"
END = "# <<< reference results"

SETUP = '''#@title Setup: install RDKit if needed and write the helper module primer.py {{ display-mode: "form" }}
# This cell is identical in all notebooks. It makes the notebook self-contained:
# on Colab or Kaggle nothing else has to be uploaded.
import importlib, importlib.util, os, subprocess, sys, time

if importlib.util.find_spec("rdkit") is None:
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "rdkit"])
    except Exception as exc:
        raise RuntimeError("RDKit is not installed and could not be installed with pip. On Kaggle, switch "
                           "'Internet' on in the notebook settings (right-hand panel) and run this cell again.") from exc
    importlib.invalidate_caches()

PRIMER_SOURCE = r\'\'\'{primer}\'\'\'

if not os.path.exists("primer.py"):
    with open("primer.py", "w", encoding="utf-8") as fh:
        fh.write(PRIMER_SOURCE)
elif open("primer.py", encoding="utf-8").read() != PRIMER_SOURCE:
    print("NOTE: a file primer.py that differs from the copy embedded in this notebook lies next to the notebook,\\n"
          "      and that FILE is used. If you did not edit it yourself, delete primer.py and run this cell again\\n"
          "      to get the version that belongs to this notebook.")

if "primer" in sys.modules:                 # the cell is run again: make sure that the module is the file on disk
    P = importlib.reload(sys.modules["primer"])
else:
    import primer as P
try:                                        # show figures inside the notebook (some older Matplotlib versions need this)
    get_ipython().run_line_magic("matplotlib", "inline")
except Exception:
    pass
T_START = time.time()
print("primer", P.__version__, "ready")'''

HIDDEN = {"cellView": "form", "jupyter": {"source_hidden": True}, "_kg_hide-input": True, "tags": ["hide-input"]}

ALL_IN_ONE_INTRO = r"""# Conformal Prediction for Physical Chemists: all six notebooks in one run

This notebook runs the six tutorial notebooks one after the other and packs everything they produce
(figures as PNG and PDF, all numbers as JSON) into **`ConformalPrimer_outputs.zip`**.

| part | topic |
|---|---|
| 1 | Two models, and why an MAE is not an error bar |
| 2 | Split conformal prediction from scratch (with CV+ for small data sets) |
| 3 | Adaptive intervals: normalized scores and conformalized quantile regression |
| 4 | Mondrian conformal prediction: a guarantee for every reaction class |
| 5 | When the guarantee breaks: exchangeability in practice |
| 6 | From intervals to decisions: selection with a guaranteed false discovery rate |

**How to run it**

* *Google Colab*: open the notebook and choose **Runtime → Run all**. No GPU is needed.
* *Kaggle*: switch **Internet on** in the notebook settings, then **Run All**. The zip file appears under *Output*.
* *Locally*: `pip install numpy pandas scipy scikit-learn matplotlib rdkit jupyterlab`, then run all cells.

The run takes about 8 minutes on Kaggle and about 12 minutes on Colab (free CPU runtimes, October 2026). The data (two tables of the RDB7 data set, 3.8 MB each) are
downloaded from Zenodo (DOI 10.5281/zenodo.6618262) on first use. The last cell compares every number with the
reference run and reports how many agree.
"""

ALL_IN_ONE_FINAL = '''# Pack the results and compare them with the reference run
import glob, json, zipfile

print("Comparison with the reference run")
summary = P.check_all()
all_agree = summary["different"] == 0 and summary["missing"] == 0

archive = "ConformalPrimer_outputs.zip"
run_info = {"environment": P.environment(), "minutes": round((time.time() - T_START) / 60, 2),
            "comparison_with_reference": summary, "reference_environment": P.REFERENCE_RESULTS.get("_environment", {}),
            "all_numbers_agree_with_reference": bool(all_agree)}
with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as z:
    for path in sorted(glob.glob("outputs/*")):
        z.write(path)
    z.writestr("run_info.json", json.dumps(run_info, indent=1))
print(f"\\nwrote {archive} ({os.path.getsize(archive) / 1e6:.1f} MB) after {(time.time() - T_START) / 60:.1f} minutes")
if "google.colab" in sys.modules and "KAGGLE_KERNEL_RUN_TYPE" not in os.environ:      # on Colab: offer the file for download
    try:
        from google.colab import files
        files.download(archive)
    except Exception:
        print("Download the file from the file browser on the left.")'''


def update_reference_block():
    """Insert reference_results.json (if present) into primer.py as a Python literal."""
    path = os.path.join(HERE, "primer.py")
    src = open(path, encoding="utf-8").read()
    ref_file = os.path.join(HERE, "reference_results.json")
    ref = json.load(open(ref_file, encoding="utf-8")) if os.path.exists(ref_file) else {}
    block = f"{BEGIN}\nREFERENCE_RESULTS.update({ref!r})\n{END}\n" if ref else f"{BEGIN}\n{END}\n"
    assert BEGIN in src
    src = re.sub(re.escape(BEGIN) + r".*?" + re.escape(END) + r"\n", lambda _: block, src, flags=re.S)
    open(path, "w", encoding="utf-8").write(src)
    return src


def write_toolbox(primer_src):
    """conformal_toolbox.py = the toolbox section of primer.py, importable without RDKit or the data."""
    body = primer_src.split("# >>> toolbox\n", 1)[1].split("# <<< toolbox", 1)[0]
    header = '''"""
conformal_toolbox.py -- the conformal prediction functions of the ConformalPrimer notebooks

Split conformal prediction, the Beta law of the coverage, CV+, normalized scores, conformalized
quantile regression, Mondrian quantiles, weighted conformal quantiles, conformal p-values and the
Benjamini–Hochberg procedure. Each function is derived step by step in the notebook named in its
docstring. The file needs only NumPy (and SciPy for `coverage_law`) and can be copied into any project.

Generated from primer.py by build_notebooks.py; edit primer.py, not this file.
"""

import numpy as np

'''
    path = os.path.join(HERE, "conformal_toolbox.py")
    open(path, "w", encoding="utf-8").write(header + body.strip("\n") + "\n")
    return path


def parse_percent(text):
    """Split a percent-format source file into (kind, source) cells."""
    cells, kind, buf = [], None, []

    def flush():
        if kind is None:
            return
        body = "\n".join(buf).strip("\n")
        if kind == "markdown":
            body = body.strip()
            for opener in ('r"""', '"""'):
                if body.startswith(opener):
                    body = body[len(opener):]
                    break
            if body.endswith('"""'):
                body = body[:-3]
            body = body.strip("\n")
        if body.strip() or kind == "setup":
            cells.append((kind, body))

    for line in text.splitlines():
        if line.startswith("# %%"):
            flush()
            tag = line[4:].strip()
            kind = {"[markdown]": "markdown", "[setup]": "setup", "[hidden]": "hidden"}.get(tag, "code")
            buf = []
        else:
            buf.append(line)
    flush()
    return cells


def new_nb():
    nb = new_notebook()
    nb.metadata.update({
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python"},
        "colab": {"provenance": [], "toc_visible": True},
    })
    return nb


def add_cells(nb, cells, primer_src, with_setup=True):
    for kind, body in cells:
        if kind == "markdown":
            nb.cells.append(new_markdown_cell(body))
        elif kind == "setup":
            if with_setup:
                nb.cells.append(new_code_cell(SETUP.format(primer=primer_src), metadata=dict(HIDDEN)))
        elif kind == "hidden":
            nb.cells.append(new_code_cell(body, metadata=dict(HIDDEN)))
        else:
            nb.cells.append(new_code_cell(body))


def finish(nb, prefix, out):
    for i, cell in enumerate(nb.cells):          # stable ids -> clean diffs
        cell["id"] = f"{prefix}-{i:03d}"
    nbformat.validate(nb)
    nbformat.write(nb, out)
    print("wrote", os.path.relpath(out, HERE), f"({len(nb.cells)} cells)")
    return out


def build():
    primer_src = update_reference_block()
    # the module is embedded in a raw triple-quoted string: make sure that this string evaluates to the module
    assert "'''" not in primer_src, "primer.py must not contain triple single quotes"
    assert primer_src.endswith("\n") and not primer_src.rstrip("\n").endswith("\\")
    assert ast.literal_eval("r'''" + primer_src + "'''") == primer_src
    written = [write_toolbox(primer_src)]
    print("wrote conformal_toolbox.py")

    sources = []
    for src_path in sorted(glob.glob(os.path.join(HERE, "nbsrc", "nb*.py"))):
        text = open(src_path, encoding="utf-8").read()
        ast.parse(text)                                   # an unbalanced triple quote in a source file is caught here
        name = re.search(r"^# NOTEBOOK: (\S+)", text, flags=re.M).group(1)
        sources.append((name, parse_percent(text)))

    for name, cells in sources:
        nb = new_nb()
        add_cells(nb, cells, primer_src)
        written.append(finish(nb, name[:2], os.path.join(HERE, f"{name}.ipynb")))

    nb = new_nb()
    nb.cells.append(new_markdown_cell(ALL_IN_ONE_INTRO))
    nb.cells.append(new_code_cell(SETUP.format(primer=primer_src), metadata=dict(HIDDEN)))
    for name, cells in sources:
        add_cells(nb, cells, primer_src, with_setup=False)
    nb.cells.append(new_markdown_cell("# Results archive and comparison with the reference run"))
    nb.cells.append(new_code_cell(ALL_IN_ONE_FINAL))
    written.append(finish(nb, "all", os.path.join(HERE, "ConformalPrimer_all_in_one.ipynb")))
    return written


if __name__ == "__main__":
    build()
