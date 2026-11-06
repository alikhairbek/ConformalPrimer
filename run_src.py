"""
run_src.py -- run a notebook source file (nbsrc/nbNN.py) as a plain script, for quick development

    python run_src.py nb03            # prints what the notebook would display; figures go to outputs/

The notebooks themselves are generated from the same sources by build_notebooks.py.
"""
import os
import sys

import matplotlib
matplotlib.use("Agg")

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import primer as P                       # noqa: E402
from build_notebooks import parse_percent  # noqa: E402


def display(obj):
    if hasattr(obj, "to_string"):
        print(obj.to_string())
    elif hasattr(obj, "data") and hasattr(obj.data, "to_string"):     # pandas Styler
        print(obj.data.to_string())
    else:
        print(obj)
    print()


if __name__ == "__main__":
    for name in sys.argv[1:]:
        path = os.path.join(HERE, "nbsrc", name if name.endswith(".py") else name + ".py")
        cells = parse_percent(open(path, encoding="utf-8").read())
        code = "\n\n".join(body for kind, body in cells if kind in ("code", "hidden"))
        os.chdir(HERE)
        exec(compile(code, path, "exec"), {"P": P, "display": display, "__name__": "__main__"})
