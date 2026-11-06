"""
run_all.py -- execute the notebooks from top to bottom and collect what they produce

    python run_all.py                 run the six notebooks
    python run_all.py 02 05           run only the notebooks whose names start with 02 and 05
    python run_all.py --all-in-one    run ConformalPrimer_all_in_one.ipynb instead
    python run_all.py --record        make the numbers of this run the new reference: run the six notebooks,
                                      store their numbers in reference_results.json, rebuild the notebooks
                                      (which embeds the numbers) and run them once more, so that the executed
                                      copies show the comparison with the new reference

Executed copies (with outputs) are written to executed/, figures and numbers to outputs/.
On Colab or Kaggle you do not need this script: open a notebook and choose "Run all".
"""

import glob
import json
import os
import sys
import time

import nbformat
from nbclient import NotebookClient
from nbclient.exceptions import CellExecutionError

HERE = os.path.dirname(os.path.abspath(__file__))
NAMES = ("nb1", "nb2", "nb3", "nb4", "nb5", "nb6")


def run(path):
    """Execute one notebook; the executed copy is written even if a cell fails."""
    nb = nbformat.read(path, as_version=4)
    t0 = time.time()
    error = None
    try:
        NotebookClient(nb, timeout=3600, kernel_name="python3", resources={"metadata": {"path": HERE}}).execute()
    except CellExecutionError as exc:
        error = exc
    os.makedirs(os.path.join(HERE, "executed"), exist_ok=True)
    nbformat.write(nb, os.path.join(HERE, "executed", os.path.basename(path)))
    if error is not None:
        raise error
    return time.time() - t0


def run_many(notebooks):
    for path in notebooks:
        print(f"running {os.path.basename(path)} ...", end=" ", flush=True)
        print(f"done in {run(path):.0f} s")


def record():
    """Store the numbers in outputs/nb1..nb6_results.json as the new reference."""
    sys.path.insert(0, HERE)
    import primer as P

    reference = {}
    for name in NAMES:
        with open(os.path.join(HERE, "outputs", f"{name}_results.json"), encoding="utf-8") as fh:
            payload = json.load(fh)
        flat = P.flatten(payload["results"])
        reference[name] = {k: v for k, v in flat.items() if v is None or (isinstance(v, (int, float)) and not isinstance(v, bool))}
        reference["_environment"] = {k: v for k, v in payload["environment"].items() if k != "platform"}
        if any(not verdict.endswith("the Zenodo file") or verdict.startswith("DIFFERENT") for verdict in payload.get("data_status", {}).values()):
            raise SystemExit("the data of this run were not the Zenodo tables: no reference recorded")
    with open(os.path.join(HERE, "reference_results.json"), "w", encoding="utf-8") as fh:
        json.dump(reference, fh, indent=1, ensure_ascii=False, sort_keys=True)
    print("recorded", sum(len(v) for k, v in reference.items() if k in NAMES), "reference numbers from", len(NAMES), "notebooks")


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    if "--all-in-one" in sys.argv:
        notebooks = [os.path.join(HERE, "ConformalPrimer_all_in_one.ipynb")]
    else:
        notebooks = sorted(glob.glob(os.path.join(HERE, "0[1-9]_*.ipynb")))
        if args:
            notebooks = [p for p in notebooks if any(os.path.basename(p).startswith(a) for a in args)]
    run_many(notebooks)
    if "--record" in sys.argv:
        if "--all-in-one" in sys.argv or args:
            raise SystemExit("--record needs a run of all six notebooks")
        record()
        import build_notebooks
        build_notebooks.build()
        print("second pass, with the new reference embedded:")
        run_many(notebooks)
