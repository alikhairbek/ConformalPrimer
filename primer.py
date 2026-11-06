"""
primer.py -- shared plumbing for the ConformalPrimer notebooks
==============================================================

Everything that is *not* the subject of the tutorial lives in sections 1-3 and 5-6,
so that the notebooks can concentrate on conformal prediction:

    1. data        download and check the RDB7 reaction-barrier tables
    2. features    turn atom-mapped reaction SMILES into numbers
    3. models      the two point predictors used throughout the tutorial
    4. toolbox     the conformal functions that the notebooks derive one by one
    5. plotting    one figure style for all notebooks
    6. results     save numbers, compare them with the reference run

Section 4 is self-contained (NumPy and SciPy only) and can be copied into other projects;
`conformal_toolbox.py` in the repository is exactly that section.

The module needs NumPy, pandas, SciPy, scikit-learn, Matplotlib and RDKit.
Nothing in it uses a GPU; the slowest step takes well under a minute on two cores.
"""

from __future__ import annotations

import glob
import hashlib
import inspect
import json
import os
import platform
import shutil
import sys
import time
import urllib.request
from types import SimpleNamespace

import numpy as np
import pandas as pd

__version__ = "0.3.0"

# --------------------------------------------------------------------------------------
# 0. Constants
# --------------------------------------------------------------------------------------

SEED = 0            # one seed for every random choice that defines the problem
ALPHA = 0.10        # default miscoverage level: 1 - ALPHA = 90% intervals
DIFFICULTY_TREES = 100   # size of the model that predicts |residual|
DIFFICULTY_FLOOR = 0.5   # sigma(x) is never smaller than this fraction of the median out-of-fold residual
R_KCAL = 1.987204e-3     # gas constant in kcal/(mol K)
KB_OVER_H = 2.083662e10  # Boltzmann constant / Planck constant in 1/(s K)

# RDB7: Spiekermann, Pattanaik, Green, Sci. Data 2022, 9, 417 (CC BY 4.0).
# The two tables used here are in version 1.0.1 of the Zenodo record, DOI 10.5281/zenodo.6618262.
# (Version 1.1.0, DOI 10.5281/zenodo.13328872, only adds a file with heats of formation.)
ZENODO_RECORD = "6618262"
DATA_FILES = {
    "ccsdtf12_dz.csv": "CCSD(T)-F12a/cc-pVDZ-F12//ωB97X-D3/def2-TZVP barrier heights (reference)",
    "wb97xd3.csv": "ωB97X-D3/def2-TZVP barrier heights (the cheap level used for Δ-learning)",
}
DATA_URLS = [
    "https://zenodo.org/records/{rec}/files/{name}?download=1",
    "https://zenodo.org/api/records/{rec}/files/{name}/content",
]
REQUIRED_COLUMNS = ["idx", "rsmi", "psmi", "rinchi", "dE0", "dHrxn298", "rmg_family"]
# SHA-256 of the files as served by Zenodo (checked on 11 October 2026) ...
OFFICIAL_SHA256 = {
    "ccsdtf12_dz.csv": "d5db1487a1cb9324f5a2b4a0044a3130194be02525059cb6a029b5fa69ec3141",
    "wb97xd3.csv": "ac1d7d06437f79593a224f9b7b1b2af4d5c1f4335f729dbdab2f0a6b86f36f8e",
}
# ... and fingerprints of their *content* (see `table_fingerprint`), which do not depend on
# line endings or on how a spreadsheet program re-writes the numbers.
REFERENCE_FINGERPRINTS = {
    "ccsdtf12_dz.csv": "ffc3f46925de474f8465a3483f225e0e929a5ee0f3314daf097113f99de717a6",
    "wb97xd3.csv": "0e705bba9602f483838e736caeb18e545e872da547a618b36c3b2992bd12ce74",
}
ENERGY_DECIMALS = 5      # energies are rounded to 1e-5 kcal/mol on loading (see `load_rdb7`)

ELEMENTS = ["H", "C", "N", "O"]                       # RDB7 contains only these
HEAVY = ["C", "N", "O"]
PAIRS_ALL = [f"{a}-{b}" for i, a in enumerate(ELEMENTS) for b in ELEMENTS[i:]]
PAIRS_HEAVY = [f"{a}-{b}" for i, a in enumerate(HEAVY) for b in HEAVY[i:]]

CLASS_ORDER = [
    "H-transfer isomerization",
    "skeletal rearrangement",
    "fragmentation",
    "carbene product",
    "zwitterionic product",
]
CLASS_SHORT = {
    "H-transfer isomerization": "H transfer",
    "skeletal rearrangement": "skeletal",
    "fragmentation": "fragmentation",
    "carbene product": "carbene",
    "zwitterionic product": "zwitterion",
}

MODEL_LABELS = {"structure": "Structure-only model", "delta": "Δ-learning model"}


def log(msg: str, verbose: bool = True) -> None:
    if verbose:
        print(msg, flush=True)


def _digest(*parts) -> str:
    """Short hash of the given strings / bytes: used for the names of cache files."""
    h = hashlib.sha256()
    for part in parts:
        h.update(part if isinstance(part, bytes) else str(part).encode())
        h.update(b"|")
    return h.hexdigest()[:16]


# --------------------------------------------------------------------------------------
# 1. Data
# --------------------------------------------------------------------------------------

def table_fingerprint(table: pd.DataFrame) -> str:
    """SHA-256 over idx, SMILES, the two energies (5 decimals) and the family label: identifies the
    *content* of a table, independently of line endings, float formatting or row order."""
    h = hashlib.sha256()
    cols = table[["idx", "rsmi", "psmi", "dE0", "dHrxn298", "rmg_family"]].sort_values("idx")
    for r in cols.itertuples(index=False):
        fam = "" if (isinstance(r.rmg_family, float) and np.isnan(r.rmg_family)) or r.rmg_family is pd.NA else str(r.rmg_family)
        h.update(f"{int(r.idx)}|{r.rsmi}|{r.psmi}|{r.dE0:.5f}|{r.dHrxn298:.5f}|{fam}\n".encode())
    return h.hexdigest()


def _file_sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _read_table(path: str) -> pd.DataFrame:
    """Read one RDB7 table and make sure that it is one."""
    table = pd.read_csv(path)
    missing = [c for c in REQUIRED_COLUMNS if c not in table.columns]
    if missing:
        raise ValueError(f"{path} is not an RDB7 table (missing columns: {', '.join(missing)})")
    return table


def _is_reference_table(path: str, name: str) -> bool:
    try:
        return table_fingerprint(_read_table(path)) == REFERENCE_FINGERPRINTS[name]
    except Exception:
        return False


def _find_or_download(name: str, data_dir: str, verbose: bool = True) -> str:
    """Return a local path for `name`: look in `data_dir`, then in Kaggle inputs, then download.

    A downloaded file is accepted only if it is the expected table; an incomplete download or an
    error page is discarded, so that it cannot be mistaken for the data on the next run."""
    local = os.path.join(data_dir, name)
    if os.path.exists(local) and os.path.getsize(local) > 0:
        return local
    for hit in sorted(glob.glob(os.path.join("/kaggle/input", "**", name), recursive=True)):
        if _is_reference_table(hit, name):
            log(f"  using {hit}", verbose)
            return hit
    errors = []
    try:
        os.makedirs(data_dir, exist_ok=True)
    except OSError as exc:
        errors.append(f"cannot create the folder '{data_dir}': {exc}")
    for template in DATA_URLS if not errors else []:
        url = template.format(rec=ZENODO_RECORD, name=name)
        part = local + ".part"
        try:
            log(f"  downloading {name} from Zenodo record {ZENODO_RECORD} ...", verbose)
            req = urllib.request.Request(url, headers={"User-Agent": f"ConformalPrimer/{__version__}"})
            with urllib.request.urlopen(req, timeout=60) as resp, open(part, "wb") as out:
                shutil.copyfileobj(resp, out)
            if not _is_reference_table(part, name):
                raise ValueError("the file received is not the expected table (incomplete download or an error page)")
            os.replace(part, local)
            return local
        except Exception as exc:  # try the next address
            errors.append(f"{url}: {exc}")
            if os.path.exists(part):
                os.remove(part)
    raise RuntimeError(
        f"Could not obtain {name}.\n"
        f"Download it by hand from https://doi.org/10.5281/zenodo.{ZENODO_RECORD} "
        f"and put it in the folder '{data_dir}/' next to the notebook (on Kaggle: switch Internet on, "
        f"or attach the file as a dataset).\nTried:\n  " + "\n  ".join(errors)
    )


def load_rdb7(data_dir: str = "data", verbose: bool = True, strict: bool = True) -> pd.DataFrame:
    """Load RDB7 as one table.

    Columns
    -------
    idx, rsmi, psmi, rinchi : reaction index, atom-mapped reactant / product SMILES, reactant InChI
    barrier_cc, dH_cc       : CCSD(T)-F12a barrier height (the quantity we predict) and reaction enthalpy
    barrier_dft, dH_dft     : the same two numbers at the ωB97X-D3 level
    rmg_family              : RMG reaction family, given by the authors for about 12% of the reactions
    All energies are in kcal/mol; barrier heights include zero-point energy.

    The energies are rounded to 1e-5 kcal/mol, so that the numbers of the notebooks do not depend on
    how many digits a particular copy of the files carries. With `strict=True` (default) a table whose
    content differs from the Zenodo file is refused.
    """
    paths = {name: _find_or_download(name, data_dir, verbose) for name in DATA_FILES}
    tables, status = {}, {}
    for name in DATA_FILES:
        table = _read_table(paths[name])
        if table["idx"].duplicated().any():
            raise ValueError(f"{name}: reaction indices are not unique")
        fp = table_fingerprint(table)
        if _file_sha256(paths[name]) == OFFICIAL_SHA256[name]:
            verdict = "byte-identical to the Zenodo file"
        elif fp == REFERENCE_FINGERPRINTS[name]:
            verdict = "same content as the Zenodo file"
        else:
            verdict = "DIFFERENT from the Zenodo file"
            if strict:
                raise ValueError(
                    f"{paths[name]} is not the RDB7 table of Zenodo record {ZENODO_RECORD} (version 1.0.1): its content differs.\n"
                    f"Delete the file to have it downloaded again, or call load_rdb7(strict=False) to use it anyway.")
        for column in ("dE0", "dHrxn298"):                   # the same numbers from every copy of the file
            table[column] = [float(f"{value:.{ENERGY_DECIMALS}f}") for value in table[column]]
        tables[name], status[name] = table, {"fingerprint": fp, "verdict": verdict}
    cc, dft = tables["ccsdtf12_dz.csv"], tables["wb97xd3.csv"]
    df = cc.rename(columns={"dE0": "barrier_cc", "dHrxn298": "dH_cc"})
    low = dft[["idx", "dE0", "dHrxn298"]].rename(columns={"dE0": "barrier_dft", "dHrxn298": "dH_dft"})
    df = df.merge(low, on="idx", how="left", validate="one_to_one").sort_values("idx").reset_index(drop=True)
    if df[["barrier_cc", "barrier_dft", "dH_dft"]].isna().any().any():
        raise ValueError("RDB7 tables are incomplete: some reactions lack a CCSD(T)-F12a or ωB97X-D3 value")
    df.attrs["fingerprints"] = {name: s["fingerprint"] for name, s in status.items()}
    df.attrs["data_status"] = {name: s["verdict"] for name, s in status.items()}
    if verbose:
        print(f"RDB7: {len(df):,} reactions of {df['rinchi'].nunique()} distinct reactants")
        for name, s in status.items():
            print(f"  {name:16s} {s['verdict']}  (content fingerprint {s['fingerprint'][:12]}...)")
    return df


# --------------------------------------------------------------------------------------
# 2. Features
# --------------------------------------------------------------------------------------

def parse_mapped_smiles(smiles: str):
    """RDKit molecule that keeps the explicit, atom-mapped hydrogens of RDB7."""
    from rdkit import Chem

    params = Chem.SmilesParserParams()
    params.removeHs = False
    mol = Chem.MolFromSmiles(smiles, params)
    if mol is None:                       # unusual valence: sanitize as far as possible
        params.sanitize = False
        mol = Chem.MolFromSmiles(smiles, params)
        mol.UpdatePropertyCache(strict=False)
        Chem.SanitizeMol(mol, Chem.SANITIZE_ALL ^ Chem.SANITIZE_PROPERTIES, catchErrors=True)
    return mol


def readable(smiles: str) -> str:
    """Plain SMILES for display: atom-map numbers and explicit hydrogens removed."""
    from rdkit import Chem

    mol = parse_mapped_smiles(smiles)
    for atom in mol.GetAtoms():
        atom.SetAtomMapNum(0)
    try:
        return Chem.MolToSmiles(Chem.RemoveHs(mol))
    except Exception:
        return Chem.MolToSmiles(mol)


def bond_orders(mol) -> dict:
    """{(map number i, map number j): bond order}, aromatic bonds counted as 1.5."""
    table = {}
    for bond in mol.GetBonds():
        i, j = bond.GetBeginAtom().GetAtomMapNum(), bond.GetEndAtom().GetAtomMapNum()
        table[(min(i, j), max(i, j))] = bond.GetBondTypeAsDouble()
    return table


def _pair(symbols: dict, i: int, j: int) -> str:
    a, b = sorted((symbols[i], symbols[j]), key=ELEMENTS.index)
    return f"{a}-{b}"


def count_rings(mol) -> int:
    """Number of independent rings (bonds - atoms + molecules); the same in every RDKit version."""
    from rdkit import Chem

    return mol.GetNumBonds() - mol.GetNumAtoms() + len(Chem.GetMolFrags(mol))


def has_charge_separation(mol) -> bool:
    """True if the Lewis structure carries formal charges, other than those with which carbon monoxide
    and isocyanides are conventionally written ([C-]#[O+], [C-]#[N+]R)."""
    charged = {atom.GetIdx() for atom in mol.GetAtoms() if atom.GetFormalCharge() != 0}
    for bond in mol.GetBonds():
        if bond.GetBondTypeAsDouble() == 3.0:
            a, b = bond.GetBeginAtom(), bond.GetEndAtom()
            for carbon, partner in ((a, b), (b, a)):
                if (carbon.GetSymbol() == "C" and carbon.GetFormalCharge() == -1
                        and partner.GetSymbol() in ("N", "O") and partner.GetFormalCharge() == 1):
                    charged -= {carbon.GetIdx(), partner.GetIdx()}
    return bool(charged)


def describe_reaction(rsmi: str, psmi: str, n_bits: int = 1024, radius: int = 2):
    """Features of one reaction.

    Returns
    -------
    desc    : dict of interpretable descriptors (bonds broken / formed / changed, sizes, rings, ...)
    fp_r    : Morgan count fingerprint of the reactant (hydrogens removed)
    fp_diff : product fingerprint minus reactant fingerprint ("difference fingerprint")
    """
    from rdkit import Chem
    from rdkit.Chem import rdFingerprintGenerator

    r, p = parse_mapped_smiles(rsmi), parse_mapped_smiles(psmi)
    symbols = {a.GetAtomMapNum(): a.GetSymbol() for a in r.GetAtoms()}
    b_r, b_p = bond_orders(r), bond_orders(p)
    broken = [k for k in b_r if k not in b_p]
    formed = [k for k in b_p if k not in b_r]
    changed = [k for k in b_r if k in b_p and b_r[k] != b_p[k]]

    d = {}
    for kind, keys in (("broken", broken), ("formed", formed)):
        for name in PAIRS_ALL:
            d[f"{kind}_{name}"] = 0
        for i, j in keys:
            d[f"{kind}_{_pair(symbols, i, j)}"] += 1
    for name in PAIRS_HEAVY:
        d[f"order_up_{name}"] = 0
        d[f"order_down_{name}"] = 0
    for i, j in changed:
        direction = "order_up_" if b_p[(i, j)] > b_r[(i, j)] else "order_down_"
        d[direction + _pair(symbols, i, j)] += 1

    d["n_broken"], d["n_formed"], d["n_changed"] = len(broken), len(formed), len(changed)
    d["n_reacting_atoms"] = len({m for key in broken + formed + changed for m in key})
    d["n_XH_broken"] = sum("H" in (symbols[i], symbols[j]) for i, j in broken)
    d["n_heavy"] = sum(a.GetAtomicNum() > 1 for a in r.GetAtoms())
    for el in ELEMENTS:
        d[f"n_{el}"] = sum(a.GetSymbol() == el for a in r.GetAtoms())
    d["n_products"] = len(Chem.GetMolFrags(p))
    d["rings_reactant"] = count_rings(r)
    d["rings_product"] = count_rings(p)
    d["delta_rings"] = d["rings_product"] - d["rings_reactant"]
    d["zwitterion_reactant"] = int(has_charge_separation(r))
    d["zwitterion_product"] = int(has_charge_separation(p))
    d["carbene_product"] = int(sum(a.GetNumRadicalElectrons() for a in p.GetAtoms()) > 0)

    generator = rdFingerprintGenerator.GetMorganGenerator(radius=radius, fpSize=n_bits)
    fps = []
    for mol in (r, p):
        heavy = Chem.RemoveHs(mol, sanitize=False)
        heavy.UpdatePropertyCache(strict=False)
        Chem.FastFindRings(heavy)
        fps.append(generator.GetCountFingerprintAsNumPy(heavy).astype(np.int16))
    return d, fps[0], fps[1] - fps[0]


def reaction_class(d) -> str:
    """Five coarse, automatically assigned reaction classes (used as Mondrian categories).

    The rules are applied in this order to the *product* side of the reaction, as RDB7 writes it:
      a product with a divalent carbon (rarely a nitrene or two radical centers)   -> carbene product
      a product written with formal charges (CO and isocyanides not counted)       -> zwitterionic product
      two or three product molecules                                               -> fragmentation
      one product, at least one X-H bond broken                                    -> H-transfer isomerization
      one product, no X-H bond broken                                              -> skeletal rearrangement
    These are bookkeeping classes read off the Lewis structures, not mechanistic classes.
    """
    if d["carbene_product"]:
        return "carbene product"
    if d["zwitterion_product"]:
        return "zwitterionic product"
    if d["n_products"] >= 2:
        return "fragmentation"
    if d["n_XH_broken"] > 0:
        return "H-transfer isomerization"
    return "skeletal rearrangement"


def featurize(df: pd.DataFrame, cache_dir: str = "cache", verbose: bool = True, n_bits: int = 1024, radius: int = 2):
    """Descriptors and fingerprints for every reaction (cached on disk)."""
    import rdkit

    key = _digest("|".join(df["rsmi"]), "|".join(df["psmi"]), __version__, rdkit.__version__, n_bits, radius)
    path = os.path.join(cache_dir, f"features_{key}.npz")
    if os.path.exists(path):
        z = np.load(path)
        desc = pd.DataFrame(z["desc"], columns=[str(c) for c in z["columns"]])
        return desc, z["fp_r"], z["fp_diff"]
    try:
        from rdkit import RDLogger
        RDLogger.DisableLog("rdApp.*")        # RDB7 contains unusual valences; the warnings are expected
    except Exception:
        pass
    t0 = time.time()
    rows, fp_r, fp_diff = [], [], []
    for rsmi, psmi in zip(df["rsmi"], df["psmi"]):
        d, a, b = describe_reaction(rsmi, psmi, n_bits, radius)
        rows.append(d)
        fp_r.append(a)
        fp_diff.append(b)
    desc = pd.DataFrame(rows).astype(np.float32)
    fp_r, fp_diff = np.asarray(fp_r), np.asarray(fp_diff)
    try:
        os.makedirs(cache_dir, exist_ok=True)
        np.savez_compressed(path, desc=desc.values, columns=np.array(list(desc.columns), dtype=str), fp_r=fp_r, fp_diff=fp_diff)
    except OSError:                           # read-only folder: work without a cache
        pass
    log(f"featurized {len(df):,} reactions in {time.time() - t0:.0f} s", verbose)
    return desc, fp_r, fp_diff


# --------------------------------------------------------------------------------------
# 3. Models
# --------------------------------------------------------------------------------------

def make_model(loss: str = "squared_error", **kwargs):
    """Gradient-boosted trees (scikit-learn): 300 trees, learning rate 0.1, no early stopping, all
    other settings at their defaults. `loss` may also be 'absolute_error' or 'quantile'."""
    from sklearn.ensemble import HistGradientBoostingRegressor

    params = dict(loss=loss, max_iter=300, learning_rate=0.1, early_stopping=False, random_state=SEED)
    params.update(kwargs)
    return HistGradientBoostingRegressor(**params)


def group_split(groups, fraction: float = 0.5, seed: int = SEED):
    """Split row indices so that all rows of one group land on the same side.

    Returns (first, second): sorted row indices; `first` holds about `fraction` of the groups."""
    groups = np.asarray(groups)
    unique = np.unique(groups)
    chosen = np.random.default_rng(seed).permutation(len(unique))[: int(round(fraction * len(unique)))]
    in_first = np.isin(groups, unique[chosen])
    return np.flatnonzero(in_first), np.flatnonzero(~in_first)


def grouped_folds(groups, n_folds: int = 5, seed: int = SEED) -> np.ndarray:
    """Fold index (0 ... n_folds-1) for every row; all rows of one group are in the same fold."""
    unique, inverse = np.unique(np.asarray(groups), return_inverse=True)
    fold_of_group = np.random.default_rng(seed).permutation(len(unique)) % n_folds
    return fold_of_group[inverse]


def frequent_columns(fp: np.ndarray, rows: np.ndarray, min_fraction: float = 0.05) -> np.ndarray:
    """Fingerprint bits that are nonzero in at least `min_fraction` of the given (training) rows."""
    return (fp[rows] != 0).mean(axis=0) >= min_fraction


def fit_model(rdb7: SimpleNamespace, name: str, train_rows) -> SimpleNamespace:
    """Fit one of the two point predictors on `train_rows` and predict every reaction.

    name = "structure": structure-only model. Input: descriptors and fingerprints. Output: the barrier.
    name = "delta"    : Δ-learning model. Input: descriptors, ωB97X-D3 barrier, and reaction enthalpy.
                        Output: the correction from the ωB97X-D3 barrier to the CCSD(T)-F12a barrier.
    Everything that is learned (including which fingerprint bits are used) comes from `train_rows` only.
    """
    train_rows = np.asarray(train_rows)
    if name == "structure":
        keep_r = frequent_columns(rdb7.fp_r, train_rows)       # bits present in at least 5% of the training reactions
        keep_d = frequent_columns(rdb7.fp_diff, train_rows)
        X = np.hstack([rdb7.desc.values, rdb7.fp_r[:, keep_r], rdb7.fp_diff[:, keep_d]]).astype(np.float32)
        baseline = np.zeros(rdb7.n)
    elif name == "delta":
        X = np.hstack([rdb7.desc.values, rdb7.df[["barrier_dft", "dH_dft"]].to_numpy(float)]).astype(np.float32)
        baseline = rdb7.df["barrier_dft"].to_numpy(float)
    else:
        raise ValueError(f"unknown model {name!r}")
    target = rdb7.y - baseline                               # what the trees actually learn
    estimator = make_model().fit(X[train_rows], target[train_rows])
    return SimpleNamespace(name=name, label=MODEL_LABELS[name], X=X, baseline=baseline, target=target,
                           estimator=estimator, train=train_rows, mu=estimator.predict(X) + baseline)


_PROBLEMS: dict = {}


def load_problem(data_dir: str = "data", cache_dir: str = "cache", verbose: bool = True) -> SimpleNamespace:
    """Download the data, build the features and fit the two point predictors.

    Returns a namespace `rdb7` with

        rdb7.df, rdb7.desc            the reaction table and its descriptors
        rdb7.y                        CCSD(T)-F12a barrier heights (kcal/mol) -- the target
        rdb7.rxn_class, rdb7.reactant reaction class and reactant identity of every reaction
        rdb7.train, rdb7.hold         row indices of the fixed split (seed 0): half of the *reactants*
                                      on each side, so that no reactant molecule is both trained on and tested
        rdb7.models["structure"|"delta"]   one namespace per model:
            .label     display name
            .X         feature matrix
            .baseline  what is added to the model output (0, or the DFT barrier)
            .train     training rows (= rdb7.train)
            .hold      held-out rows (= rdb7.hold): the pool for calibration and test sets
            .mu        point prediction for every reaction

    The held-out half is never used for fitting; every notebook splits it again into a
    calibration set and a test set.
    """
    key = (os.path.abspath(data_dir), os.path.abspath(cache_dir))
    if key in _PROBLEMS:                                   # already built in this Python session
        log("RDB7 and both models are already in memory", verbose)
        return _PROBLEMS[key]
    t0 = time.time()
    df = load_rdb7(data_dir, verbose)
    desc, fp_r, fp_diff = featurize(df, cache_dir, verbose)
    y = df["barrier_cc"].to_numpy(float)
    reactant = df["rinchi"].to_numpy()
    train, hold = group_split(reactant, 0.5, SEED)
    classes = np.array([reaction_class(r) for r in desc.to_dict("records")])
    rdb7 = SimpleNamespace(df=df, desc=desc, y=y, n=len(df), train=train, hold=hold, models={},
                           rxn_class=classes, reactant=reactant, n_heavy=desc["n_heavy"].to_numpy(int),
                           fp_r=fp_r, fp_diff=fp_diff, cache_dir=cache_dir)
    for name in MODEL_LABELS:
        mod = fit_model(rdb7, name, train)
        mod.hold = hold
        rdb7.models[name] = mod
        if verbose:
            err = np.abs(y - mod.mu)[hold]
            print(f"{mod.label:22s} {mod.X.shape[1]:4d} features | train {len(train)} | held out {len(hold)}"
                  f" | held-out MAE {err.mean():.2f} kcal/mol")
    log(f"ready in {time.time() - t0:.0f} s", verbose)
    _PROBLEMS[key] = rdb7
    return rdb7


def fit_difficulty(rdb7: SimpleNamespace, name: str, n_folds: int = 5, verbose: bool = True) -> np.ndarray:
    """sigma(x) > 0: how large an error to expect for reaction x (Notebook 3 builds this step by step).

    1. Out-of-fold predictions on the training set give honest residuals for the training reactions
       (the folds keep all reactions of one reactant together).
    2. A second, smaller model learns |out-of-fold residual| from the same features.
    3. Its prediction, bounded from below, is sigma(x).
    The held-out reactions are not used anywhere.
    """
    import sklearn

    mod = rdb7.models[name]
    key = _digest(mod.mu.tobytes(), name, n_folds, DIFFICULTY_TREES, DIFFICULTY_FLOOR, SEED, __version__, sklearn.__version__)
    path = os.path.join(rdb7.cache_dir, f"sigma_{name}_{key}.npy")
    if os.path.exists(path):
        return np.load(path)
    fold = grouped_folds(rdb7.reactant[mod.train], n_folds, SEED)
    oof = np.empty(len(mod.train))
    for k in range(n_folds):
        fit_rows, out_rows = mod.train[fold != k], mod.train[fold == k]
        oof[fold == k] = make_model().fit(mod.X[fit_rows], mod.target[fit_rows]).predict(mod.X[out_rows])
    abs_res = np.abs(mod.target[mod.train] - oof)
    sigma_model = make_model(max_iter=DIFFICULTY_TREES).fit(mod.X[mod.train], abs_res)
    floor = DIFFICULTY_FLOOR * np.median(abs_res)
    sigma = np.maximum(sigma_model.predict(mod.X), floor)
    try:
        os.makedirs(rdb7.cache_dir, exist_ok=True)
        np.save(path, sigma)
    except OSError:
        pass
    log(f"{mod.label}: difficulty model fitted (out-of-fold MAE {abs_res.mean():.2f} kcal/mol)", verbose)
    return sigma


# --------------------------------------------------------------------------------------
# 4. Conformal toolbox
# --------------------------------------------------------------------------------------

# >>> toolbox
# The conformal functions that the notebooks derive step by step, collected here for reuse.
# They need only NumPy (and SciPy for `coverage_law`).

def _rank(n: int, alpha: float) -> int:
    """ceil((n+1)(1-alpha)), protected against floating-point round-up (e.g., 75 * 0.68 = 51.00000000000001)."""
    if not 0 < alpha < 1:
        raise ValueError("alpha must lie strictly between 0 and 1")
    return int(np.ceil((n + 1) * (1 - alpha) - 1e-9))


def conformal_quantile(scores, alpha: float = 0.10) -> float:
    """The ceil((n+1)(1-alpha))-th smallest calibration score; infinity if n is too small.   [Notebook 2]"""
    scores = np.asarray(scores, dtype=float).ravel()
    if np.isnan(scores).any():
        raise ValueError("the calibration scores contain NaN")
    n = scores.size
    k = _rank(n, alpha)                                 # rank of the order statistic we need
    if k > n:                                           # fewer than (1-alpha)/alpha calibration points
        return np.inf
    return float(np.partition(scores, k - 1)[k - 1])


def split_conformal(y_cal, mu_cal, mu_test, alpha: float = 0.10):
    """Absolute-residual split conformal interval mu +/- q. Returns (lower, upper, q).   [Notebook 2]"""
    mu_test = np.asarray(mu_test, dtype=float)
    q = conformal_quantile(np.abs(np.asarray(y_cal, dtype=float) - np.asarray(mu_cal, dtype=float)), alpha)
    return mu_test - q, mu_test + q, q


def coverage_law(n_cal: int, alpha: float = 0.10):
    """Beta distribution of the coverage obtained from ONE calibration set of size n_cal.   [Notebook 2]
    Returns None if n_cal is too small for a finite interval at this level."""
    from scipy import stats

    k = _rank(n_cal, alpha)
    if k > n_cal:
        return None
    return stats.beta(k, n_cal + 1 - k)


def cv_plus(y, oof_pred, fold, test_pred, alpha: float = 0.10):
    """CV+ interval (Barber et al., 2021): no separate calibration set is needed.   [Notebook 2]

    y, oof_pred : reference values of the n labeled points and their out-of-fold predictions
    fold        : fold index (0 ... K-1) of each labeled point
    test_pred   : array (K, m); row k holds the predictions for the m test points made by the
                  model that was fitted WITHOUT fold k
    Returns (lower, upper). Worst-case coverage: 1 - 2 alpha for the jackknife+ (K = n) and slightly less
    for K < n (the bound is given in Notebook 2); in practice the coverage is close to 1 - alpha.
    """
    y, oof_pred, fold = np.asarray(y, float), np.asarray(oof_pred, float), np.asarray(fold, int)
    test_pred = np.asarray(test_pred, float)
    n = y.size
    residual = np.abs(y - oof_pred)
    k_hi = _rank(n, alpha)                              # rank for the upper end
    k_lo = int(np.floor((n + 1) * alpha + 1e-9))        # rank for the lower end
    m = test_pred.shape[1]
    if k_hi > n or k_lo < 1:
        return np.full(m, -np.inf), np.full(m, np.inf)
    low = test_pred[fold] - residual[:, None]           # (n, m): what point i "says" about test point j
    high = test_pred[fold] + residual[:, None]
    lower = np.partition(low, k_lo - 1, axis=0)[k_lo - 1]
    upper = np.partition(high, k_hi - 1, axis=0)[k_hi - 1]
    return lower, upper


def normalized_conformal(y_cal, mu_cal, sigma_cal, mu_test, sigma_test, alpha: float = 0.10):
    """Interval mu +/- q * sigma(x) from the score |y - mu| / sigma. Returns (lower, upper, q).   [Notebook 3]"""
    sigma_cal, sigma_test = np.asarray(sigma_cal, dtype=float), np.asarray(sigma_test, dtype=float)
    mu_test = np.asarray(mu_test, dtype=float)
    if (sigma_cal <= 0).any() or (sigma_test <= 0).any():
        raise ValueError("sigma must be positive")
    q = conformal_quantile(np.abs(np.asarray(y_cal, dtype=float) - np.asarray(mu_cal, dtype=float)) / sigma_cal, alpha)
    return mu_test - q * sigma_test, mu_test + q * sigma_test, q


def cqr(y_cal, lo_cal, hi_cal, lo_test, hi_test, alpha: float = 0.10):
    """Conformalized quantile regression (Romano et al., 2019). Returns (lower, upper, q).   [Notebook 3]

    lo_*, hi_* : predictions of a lower and an upper quantile model (e.g., the 5% and 95% quantiles)
    q may be negative: the raw quantile band is then *shrunk* (and where the band is narrower than
    2|q| the interval is empty: lower > upper).
    """
    y_cal = np.asarray(y_cal, dtype=float)
    lo_test, hi_test = np.asarray(lo_test, dtype=float), np.asarray(hi_test, dtype=float)
    q = conformal_quantile(np.maximum(np.asarray(lo_cal, dtype=float) - y_cal, y_cal - np.asarray(hi_cal, dtype=float)), alpha)
    return lo_test - q, hi_test + q, q


def interval_score(y, lower, upper, alpha: float = 0.10) -> float:
    """Mean interval (Winkler) score: width plus (2/alpha) x the distance by which y is missed.   [Notebook 3]
    Smaller is better; it rewards narrow intervals and penalizes misses in proportion to their size."""
    y, lower, upper = np.asarray(y, dtype=float), np.asarray(lower, dtype=float), np.asarray(upper, dtype=float)
    miss = np.maximum(lower - y, 0) + np.maximum(y - upper, 0)
    return float(np.mean((upper - lower) + (2 / alpha) * miss))


def mondrian_quantiles(scores, groups, alpha: float = 0.10) -> dict:
    """One conformal quantile per group: {group label: q}.   [Notebook 4]"""
    scores, groups = np.asarray(scores, float), np.asarray(groups)
    return {g: conformal_quantile(scores[groups == g], alpha) for g in np.unique(groups)}


def weighted_conformal_quantile(scores, w_cal, w_test, alpha: float = 0.10) -> np.ndarray:
    """Weighted conformal quantile (Tibshirani et al., 2019), one value per test point.   [Notebook 5]

    scores, w_cal : calibration scores and their weights w(x_i) = dP_test/dP_cal (any common scale)
    w_test        : weight(s) of the test point(s)
    The result is infinite where the calibration weight at or below the largest score does not reach
    (1 - alpha) of the total weight, the test point included.
    """
    if not 0 < alpha < 1:
        raise ValueError("alpha must lie strictly between 0 and 1")
    scores, w_cal = np.asarray(scores, float).ravel(), np.asarray(w_cal, float).ravel()
    w_test = np.atleast_1d(np.asarray(w_test, float))
    if scores.size != w_cal.size:
        raise ValueError("one weight is needed for every calibration score")
    if np.isnan(scores).any():
        raise ValueError("the calibration scores contain NaN")
    for w in (w_cal, w_test):
        if not np.isfinite(w).all() or (w < 0).any():
            raise ValueError("weights must be finite and non-negative")
    order = np.argsort(scores)
    cum = np.cumsum(w_cal[order])                       # weight at or below each sorted score
    total = (cum[-1] if cum.size else 0.0) + w_test     # total weight, test point included
    need = (1 - alpha) * total                          # mass that must lie at or below the quantile
    idx = np.searchsorted(cum, need * (1 - 1e-9), side="left")      # relative tolerance, like the 1e-9 in the ranks
    padded = np.append(scores[order], np.inf)           # idx == n: the test point's own mass is needed
    result = padded[idx]
    result[total <= 0] = np.inf                         # no weight at all: nothing can be said
    return result


def effective_sample_size(weights) -> float:
    """(sum w)^2 / sum w^2: how many equally weighted calibration points the weighted set is worth.   [Notebook 5]"""
    w = np.asarray(weights, float)
    return float(w.sum() ** 2 / np.sum(w ** 2)) if np.any(w > 0) else 0.0


def conformal_pvalues(score_cal, null_cal, score_test) -> np.ndarray:
    """Conformal p-values for the hypothesis 'this candidate is NOT a hit' (Jin and Candès, 2023).   [Notebook 6]

    score     : any number computed from x alone; smaller = looks more promising
    score_cal, null_cal : scores of the WHOLE calibration set (hits and non-hits, in the proportions
                in which they occur among the candidates) and True for those that are known non-hits
    p_j = (1 + number of calibration non-hits that look at least as promising as j) / (n + 1)
    """
    score_cal, null_cal = np.asarray(score_cal, float), np.asarray(null_cal, bool)
    score_test = np.asarray(score_test, float)
    if np.isnan(score_cal).any() or np.isnan(score_test).any():
        raise ValueError("the scores contain NaN")
    null_scores = np.sort(score_cal[null_cal])
    return (1 + np.searchsorted(null_scores, score_test, side="right")) / (score_cal.size + 1)


def benjamini_hochberg(pvalues, q: float) -> np.ndarray:
    """Boolean mask of the hypotheses rejected by the Benjamini–Hochberg procedure at level q.   [Notebook 6]"""
    if not 0 < q < 1:
        raise ValueError("q must lie strictly between 0 and 1")
    p = np.asarray(pvalues, float)
    m = p.size
    order = np.argsort(p, kind="stable")
    line = q * np.arange(1, m + 1) / m * (1 + 1e-12)    # the tiny factor makes "p equal to the line" count as below it
    passed = np.flatnonzero(p[order] <= line)
    selected = np.zeros(m, dtype=bool)
    if passed.size:
        selected[order[: passed[-1] + 1]] = True
    return selected
# <<< toolbox


def resplits(pool, n_cal: int, n_rep: int, seed: int = SEED):
    """Yield `n_rep` random (calibration, test) partitions of the index array `pool`."""
    pool = np.asarray(pool)
    if not 0 < n_cal < len(pool):
        raise ValueError("n_cal must be smaller than the pool, so that a test set remains")
    rng = np.random.default_rng(seed)
    for _ in range(n_rep):
        perm = rng.permutation(pool)
        yield perm[:n_cal], perm[n_cal:]


def show(function) -> None:
    """Print the source code of a function of this module (so that no notebook hides its ingredients)."""
    print(inspect.getsource(function))


def rate_factor(delta_e, temperature: float = 298.15):
    """Factor by which a rate constant changes when the barrier changes by delta_e (kcal/mol)."""
    return np.exp(np.asarray(delta_e, float) / (R_KCAL * temperature))


# --------------------------------------------------------------------------------------
# 5. Plotting
# --------------------------------------------------------------------------------------

# Color-blind-safe palette of Wong (Nat. Methods 2011, 8, 441); black is reserved for text.
BLUE, VERMILLION, GREEN = "#0072B2", "#D55E00", "#009E73"
PURPLE, ORANGE, SKY = "#CC79A7", "#E69F00", "#56B4E9"
GRAY, LIGHT_GRAY = "#6E6E6E", "#D0D0D0"
MODEL_COLORS = {"structure": VERMILLION, "delta": BLUE}
MODEL_MARKERS = {"structure": "s", "delta": "o"}


def set_style() -> None:
    import matplotlib as mpl

    mpl.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "Liberation Sans", "DejaVu Sans"],
        "font.size": 8, "axes.titlesize": 8, "axes.labelsize": 8,
        "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 7,
        "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
        "xtick.major.size": 3, "ytick.major.size": 3,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.edgecolor": GRAY, "xtick.color": GRAY, "ytick.color": GRAY,
        "xtick.labelcolor": "black", "ytick.labelcolor": "black", "axes.labelcolor": "black",
        "lines.linewidth": 1.4, "lines.markersize": 4,
        "legend.frameon": False, "legend.handlelength": 1.6,
        "figure.dpi": 130, "savefig.dpi": 300, "savefig.bbox": "tight", "savefig.pad_inches": 0.03,
        "pdf.fonttype": 42, "ps.fonttype": 42, "figure.facecolor": "white",
    })


def panels(n: int, height: float = 2.45, width: float = 7.0, **kwargs):
    """A row of n panels, 7 inches wide (two journal columns), laid out so that no label is cut off."""
    import matplotlib.pyplot as plt

    return plt.subplots(1, n, figsize=(width, height), layout="constrained", **kwargs)


def panel_label(ax, letter: str, title: str | None = None) -> None:
    """Bold panel letter just outside the upper left corner of the axes, and optionally the panel title next to it."""
    ax.text(-0.02, 1.04, f"({letter})", transform=ax.transAxes, fontsize=9, fontweight="bold",
            va="bottom", ha="right", color="black")
    if title is not None:
        ax.set_title(title, loc="left")


def savefig(fig, name: str, out_dir: str = "outputs") -> None:
    """Save a figure as 300-dpi PNG and as vector PDF."""
    os.makedirs(out_dir, exist_ok=True)
    fig.savefig(os.path.join(out_dir, f"{name}.png"))
    fig.savefig(os.path.join(out_dir, f"{name}.pdf"), metadata={"CreationDate": None})


# --------------------------------------------------------------------------------------
# 6. Results
# --------------------------------------------------------------------------------------

NOTEBOOKS = ("nb1", "nb2", "nb3", "nb4", "nb5", "nb6")
LIBRARIES = ("numpy", "pandas", "scipy", "scikit-learn", "matplotlib", "rdkit", "primer")
MODEL_DEPENDENT_COUNTS = ("selected", "false discoveries", "hits found", "smallest non-empty")


def _plain(obj):
    """Make NumPy scalars / arrays JSON-serializable."""
    if isinstance(obj, dict):
        return {str(k): _plain(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_plain(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return [_plain(v) for v in obj.tolist()]
    if isinstance(obj, (np.bool_, bool)):
        return bool(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating, float)):
        return None if not np.isfinite(obj) else round(float(obj), 6)
    return obj


def environment() -> dict:
    import matplotlib
    import scipy
    import sklearn

    try:
        import rdkit
        rd = rdkit.__version__
    except Exception:  # pragma: no cover
        rd = "not installed"
    return {"python": sys.version.split()[0], "platform": platform.platform(), "numpy": np.__version__,
            "pandas": pd.__version__, "scipy": scipy.__version__, "scikit-learn": sklearn.__version__,
            "matplotlib": matplotlib.__version__, "rdkit": rd, "primer": __version__}


def flatten(d: dict, prefix: str = "") -> dict:
    flat = {}
    for k, v in d.items():
        key = f"{prefix}/{k}" if prefix else str(k)
        if isinstance(v, dict):
            flat.update(flatten(v, key))
        elif isinstance(v, list):
            flat.update(flatten({str(i): item for i, item in enumerate(v)}, key))
        else:
            flat[key] = v
    return flat


def save_results(name: str, results: dict, rdb7: SimpleNamespace | None = None, out_dir: str = "outputs") -> str:
    """Write the numbers of one notebook to outputs/<name>_results.json."""
    os.makedirs(out_dir, exist_ok=True)
    payload = {"notebook": name, "environment": environment(), "results": _plain(results)}
    if rdb7 is not None:
        payload["data_fingerprints"] = rdb7.df.attrs.get("fingerprints", {})
        payload["data_status"] = rdb7.df.attrs.get("data_status", {})
        payload["n_reactions"] = int(rdb7.n)
    path = os.path.join(out_dir, f"{name}_results.json")
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=1, ensure_ascii=False)
    print(f"saved {path}")
    return path


# Reference values from the reference run are appended below by build_notebooks.py.
REFERENCE_RESULTS: dict = {}

# >>> reference results (written by build_notebooks.py -- do not edit by hand)
REFERENCE_RESULTS.update({'_environment': {'matplotlib': '3.10.0', 'numpy': '2.1.3', 'pandas': '2.2.3', 'primer': '0.3.0', 'python': '3.12.3', 'rdkit': '2026.09.1', 'scikit-learn': '1.6.1', 'scipy': '1.16.3'}, 'nb1': {'barrier_mean': 80.060894, 'class_counts/H-transfer isomerization': 2643, 'class_counts/carbene product': 1985, 'class_counts/fragmentation': 2534, 'class_counts/skeletal rearrangement': 1736, 'class_counts/zwitterionic product': 3028, 'delta/class_mae/H-transfer isomerization': 1.927528, 'delta/class_mae/carbene product': 0.890999, 'delta/class_mae/fragmentation': 1.550066, 'delta/class_mae/skeletal rearrangement': 2.468339, 'delta/class_mae/zwitterionic product': 1.459087, 'delta/class_q90/H-transfer isomerization': 4.795443, 'delta/class_q90/carbene product': 2.151114, 'delta/class_q90/fragmentation': 3.427818, 'delta/class_q90/skeletal rearrangement': 6.133364, 'delta/class_q90/zwitterionic product': 3.252838, 'delta/errors/90': 3.786015, 'delta/errors/95': 5.613923, 'delta/errors/99': 10.763432, 'delta/errors/MAE': 1.620686, 'delta/errors/RMSE': 2.768401, 'delta/errors/largest': 32.966523, 'delta/errors/median': 0.895411, 'delta/excess_kurtosis': 18.034909, 'delta/gaussian_coverage/0.50': 0.729065, 'delta/gaussian_coverage/0.68': 0.832687, 'delta/gaussian_coverage/0.80': 0.88829, 'delta/gaussian_coverage/0.90': 0.927211, 'delta/gaussian_coverage/0.95': 0.94642, 'delta/gaussian_coverage/0.99': 0.970851, 'delta/n_features': 51, 'family_share': 0.124182, 'n_hold': 5935, 'n_reactants': 1135, 'n_reactions': 11926, 'n_train': 5991, 'shared_reactants': 0, 'structure/class_mae/H-transfer isomerization': 8.343394, 'structure/class_mae/carbene product': 7.209082, 'structure/class_mae/fragmentation': 7.555171, 'structure/class_mae/skeletal rearrangement': 10.636322, 'structure/class_mae/zwitterionic product': 9.486534, 'structure/class_q90/H-transfer isomerization': 18.806988, 'structure/class_q90/carbene product': 17.167928, 'structure/class_q90/fragmentation': 17.310724, 'structure/class_q90/skeletal rearrangement': 22.827262, 'structure/class_q90/zwitterionic product': 22.389925, 'structure/errors/90': 20.100311, 'structure/errors/95': 25.765606, 'structure/errors/99': 41.287609, 'structure/errors/MAE': 8.585918, 'structure/errors/RMSE': 12.213441, 'structure/errors/largest': 100.291134, 'structure/errors/median': 5.927732, 'structure/excess_kurtosis': 3.749813, 'structure/gaussian_coverage/0.50': 0.628981, 'structure/gaussian_coverage/0.68': 0.774052, 'structure/gaussian_coverage/0.80': 0.841112, 'structure/gaussian_coverage/0.90': 0.899916, 'structure/gaussian_coverage/0.95': 0.935131, 'structure/gaussian_coverage/0.99': 0.975063, 'structure/n_features': 173}, 'nb2': {'coverage_vs_n/100/hi': 0.950647, 'coverage_vs_n/100/lo': 0.833068, 'coverage_vs_n/100/mean': 0.900555, 'coverage_vs_n/100/theory_hi': 0.9515, 'coverage_vs_n/100/theory_lo': 0.835476, 'coverage_vs_n/100/theory_mean': 0.90099, 'coverage_vs_n/1000/hi': 0.920365, 'coverage_vs_n/1000/lo': 0.879635, 'coverage_vs_n/1000/mean': 0.900135, 'coverage_vs_n/1000/theory_hi': 0.919554, 'coverage_vs_n/1000/theory_lo': 0.878825, 'coverage_vs_n/1000/theory_mean': 0.9001, 'coverage_vs_n/20/hi': 0.988005, 'coverage_vs_n/20/lo': 0.740651, 'coverage_vs_n/20/mean': 0.903421, 'coverage_vs_n/20/theory_hi': 0.987828, 'coverage_vs_n/20/theory_lo': 0.750972, 'coverage_vs_n/20/theory_mean': 0.904762, 'coverage_vs_n/200/hi': 0.937929, 'coverage_vs_n/200/lo': 0.854926, 'coverage_vs_n/200/mean': 0.900169, 'coverage_vs_n/200/theory_hi': 0.938448, 'coverage_vs_n/200/theory_lo': 0.854926, 'coverage_vs_n/200/theory_mean': 0.900498, 'coverage_vs_n/2000/hi': 0.915883, 'coverage_vs_n/2000/lo': 0.883602, 'coverage_vs_n/2000/mean': 0.899968, 'coverage_vs_n/2000/theory_hi': 0.915629, 'coverage_vs_n/2000/theory_lo': 0.883355, 'coverage_vs_n/2000/theory_mean': 0.90005, 'coverage_vs_n/50/hi': 0.964325, 'coverage_vs_n/50/lo': 0.803726, 'coverage_vs_n/50/mean': 0.900327, 'coverage_vs_n/50/theory_hi': 0.967035, 'coverage_vs_n/50/theory_lo': 0.807307, 'coverage_vs_n/50/theory_mean': 0.901961, 'coverage_vs_n/500/hi': 0.926224, 'coverage_vs_n/500/lo': 0.871385, 'coverage_vs_n/500/mean': 0.899947, 'coverage_vs_n/500/theory_hi': 0.926035, 'coverage_vs_n/500/theory_lo': 0.871205, 'coverage_vs_n/500/theory_mean': 0.9002, 'dft_offset/mean': 3.477724, 'dft_offset/share_positive': 0.88466, 'levels/0.50/conformal': 0.500063, 'levels/0.50/conformal_se': 0.000558, 'levels/0.50/gaussian': 0.729065, 'levels/0.68/conformal': 0.679239, 'levels/0.68/conformal_se': 0.000511, 'levels/0.68/gaussian': 0.832687, 'levels/0.80/conformal': 0.799297, 'levels/0.80/conformal_se': 0.000428, 'levels/0.80/gaussian': 0.88829, 'levels/0.90/conformal': 0.899662, 'levels/0.90/conformal_se': 0.000326, 'levels/0.90/gaussian': 0.927211, 'levels/0.95/conformal': 0.950033, 'levels/0.95/conformal_se': 0.000244, 'levels/0.95/gaussian': 0.94642, 'levels/0.99/conformal': 0.990067, 'levels/0.99/conformal_se': 0.000107, 'levels/0.99/gaussian': 0.970851, 'n_needed/pm1': 3450, 'n_needed/pm2': 854, 'n_needed/pm3': 374, 'n_needed/pm5': 131, 'pitfall_quantile/1000/conformal': 0.900075, 'pitfall_quantile/1000/plain': 0.899166, 'pitfall_quantile/20/conformal': 0.90461, 'pitfall_quantile/20/plain': 0.865542, 'pitfall_quantile/200/conformal': 0.900781, 'pitfall_quantile/200/plain': 0.896344, 'pitfall_quantile/50/conformal': 0.901299, 'pitfall_quantile/50/plain': 0.884803, 'pitfall_training/coverage': 0.760284, 'pitfall_training/q': 2.099674, 'sharpness/constant/coverage': 0.89999, 'sharpness/constant/mae': 17.328224, 'sharpness/constant/q': 35.956381, 'sharpness/delta/coverage': 0.899822, 'sharpness/delta/mae': 1.620686, 'sharpness/delta/q': 3.798717, 'sharpness/dft/coverage': 0.900752, 'sharpness/dft/mae': 3.614416, 'sharpness/dft/q': 8.242418, 'sharpness/dft_shifted/coverage': 0.90031, 'sharpness/dft_shifted/mae': 2.822347, 'sharpness/dft_shifted/q': 5.385178, 'sharpness/structure/coverage': 0.900015, 'sharpness/structure/mae': 8.585918, 'sharpness/structure/q': 20.118245, 'single_split/coverage': 0.898683, 'single_split/k': 901, 'single_split/n_cal': 1000, 'single_split/n_test': 4935, 'single_split/q_hat': 3.771071, 'small_data/N=100 | CV+ (K = 10)/coverage': 0.911155, 'small_data/N=100 | CV+ (K = 10)/coverage_sd': 0.030147, 'small_data/N=100 | CV+ (K = 10)/half_width': 5.759923, 'small_data/N=100 | CV+ (K = 10)/half_width_sd': 0.977303, 'small_data/N=100 | split 50/50/coverage': 0.907845, 'small_data/N=100 | split 50/50/coverage_sd': 0.040093, 'small_data/N=100 | split 50/50/half_width': 6.255902, 'small_data/N=100 | split 50/50/half_width_sd': 1.436231, 'small_data/N=100 | training residuals (wrong)/coverage': 0.822685, 'small_data/N=100 | training residuals (wrong)/coverage_sd': 0.045207, 'small_data/N=100 | training residuals (wrong)/half_width': 4.090479, 'small_data/N=100 | training residuals (wrong)/half_width_sd': 0.675308, 'small_data/N=1000 | CV+ (K = 10)/coverage': 0.902787, 'small_data/N=1000 | CV+ (K = 10)/coverage_sd': 0.012303, 'small_data/N=1000 | CV+ (K = 10)/half_width': 4.774937, 'small_data/N=1000 | CV+ (K = 10)/half_width_sd': 0.263609, 'small_data/N=1000 | split 50/50/coverage': 0.90085, 'small_data/N=1000 | split 50/50/coverage_sd': 0.015576, 'small_data/N=1000 | split 50/50/half_width': 4.900471, 'small_data/N=1000 | split 50/50/half_width_sd': 0.344123, 'small_data/N=1000 | training residuals (wrong)/coverage': 0.891713, 'small_data/N=1000 | training residuals (wrong)/coverage_sd': 0.012906, 'small_data/N=1000 | training residuals (wrong)/half_width': 4.530891, 'small_data/N=1000 | training residuals (wrong)/half_width_sd': 0.25961, 'small_data/N=300 | CV+ (K = 10)/coverage': 0.90666, 'small_data/N=300 | CV+ (K = 10)/coverage_sd': 0.018284, 'small_data/N=300 | CV+ (K = 10)/half_width': 5.1184, 'small_data/N=300 | CV+ (K = 10)/half_width_sd': 0.476453, 'small_data/N=300 | split 50/50/coverage': 0.905575, 'small_data/N=300 | split 50/50/coverage_sd': 0.023052, 'small_data/N=300 | split 50/50/half_width': 5.459493, 'small_data/N=300 | split 50/50/half_width_sd': 0.673392, 'small_data/N=300 | training residuals (wrong)/coverage': 0.87148, 'small_data/N=300 | training residuals (wrong)/coverage_sd': 0.022526, 'small_data/N=300 | training residuals (wrong)/half_width': 4.402272, 'small_data/N=300 | training residuals (wrong)/half_width_sd': 0.426196, 'small_data/N=50 | CV+ (K = 10)/coverage': 0.910685, 'small_data/N=50 | CV+ (K = 10)/coverage_sd': 0.039803, 'small_data/N=50 | CV+ (K = 10)/half_width': 6.093728, 'small_data/N=50 | CV+ (K = 10)/half_width_sd': 1.40391, 'small_data/N=50 | jackknife+ (K = N)/coverage': 0.905525, 'small_data/N=50 | jackknife+ (K = N)/coverage_sd': 0.047633, 'small_data/N=50 | jackknife+ (K = N)/half_width': 6.059034, 'small_data/N=50 | jackknife+ (K = N)/half_width_sd': 1.403652, 'small_data/N=50 | split 50/50/coverage': 0.916725, 'small_data/N=50 | split 50/50/coverage_sd': 0.054944, 'small_data/N=50 | split 50/50/half_width': 7.260283, 'small_data/N=50 | split 50/50/half_width_sd': 2.516128, 'small_data/N=50 | training residuals (wrong)/coverage': 0.77505, 'small_data/N=50 | training residuals (wrong)/coverage_sd': 0.076258, 'small_data/N=50 | training residuals (wrong)/half_width': 3.789945, 'small_data/N=50 | training residuals (wrong)/half_width_sd': 1.001647}, 'nb3': {'class_coverage_single_split/H transfer': 0.85224, 'class_coverage_single_split/carbene': 0.968458, 'class_coverage_single_split/fragmentation': 0.924386, 'class_coverage_single_split/skeletal': 0.797059, 'class_coverage_single_split/zwitterion': 0.922601, 'fraction_narrower_than_split': 0.678419, 'pitfall_in_sample/in-sample residuals/coverage': 0.900254, 'pitfall_in_sample/in-sample residuals/easiest fifth': 0.951079, 'pitfall_in_sample/in-sample residuals/hardest fifth': 0.850405, 'pitfall_in_sample/in-sample residuals/skeletal rearrangements': 0.872302, 'pitfall_in_sample/out-of-fold residuals (Section 2)/coverage': 0.900616, 'pitfall_in_sample/out-of-fold residuals (Section 2)/easiest fifth': 0.88976, 'pitfall_in_sample/out-of-fold residuals (Section 2)/hardest fifth': 0.922148, 'pitfall_in_sample/out-of-fold residuals (Section 2)/skeletal rearrangements': 0.905008, 'q/cqr': 0.72819, 'q/normalized': 2.136523, 'q/split': 3.771071, 'raw_quantile_band_coverage': 0.775737, 'sigma/floor': 0.445916, 'sigma/in_sample_mae': 0.912204, 'sigma/oof_mae': 1.645348, 'sigma/share_at_floor': 0.067734, 'sigma/spearman': 0.558497, 'sigma/tenth_error/0': 0.465166, 'sigma/tenth_error/1': 0.552436, 'sigma/tenth_error/2': 0.789108, 'sigma/tenth_error/3': 0.929193, 'sigma/tenth_error/4': 1.138651, 'sigma/tenth_error/5': 1.376384, 'sigma/tenth_error/6': 1.744693, 'sigma/tenth_error/7': 2.140548, 'sigma/tenth_error/8': 2.877148, 'sigma/tenth_error/9': 4.195373, 'sigma/tenth_sigma/0': 0.453751, 'sigma/tenth_sigma/1': 0.560751, 'sigma/tenth_sigma/2': 0.730795, 'sigma/tenth_sigma/3': 0.900941, 'sigma/tenth_sigma/4': 1.112342, 'sigma/tenth_sigma/5': 1.371394, 'sigma/tenth_sigma/6': 1.667395, 'sigma/tenth_sigma/7': 2.098883, 'sigma/tenth_sigma/8': 2.81514, 'sigma/tenth_sigma/9': 4.614093, 'summary/CQR/class: H transfer': 0.899027, 'summary/CQR/class: carbene': 0.93667, 'summary/CQR/class: fragmentation': 0.90406, 'summary/CQR/class: skeletal': 0.864992, 'summary/CQR/class: zwitterion': 0.895997, 'summary/CQR/coverage': 0.901174, 'summary/CQR/interval score': 10.598758, 'summary/CQR/mean half-width': 3.339694, 'summary/CQR/median half-width': 2.886454, 'summary/CQR/own-width fifth 1': 0.953921, 'summary/CQR/own-width fifth 2': 0.936049, 'summary/CQR/own-width fifth 3': 0.897447, 'summary/CQR/own-width fifth 4': 0.873759, 'summary/CQR/own-width fifth 5': 0.844696, 'summary/CQR/sigma fifth 1': 0.96565, 'summary/CQR/sigma fifth 2': 0.924598, 'summary/CQR/sigma fifth 3': 0.909234, 'summary/CQR/sigma fifth 4': 0.875929, 'summary/CQR/sigma fifth 5': 0.830466, 'summary/normalized/class: H transfer': 0.908543, 'summary/normalized/class: carbene': 0.903226, 'summary/normalized/class: fragmentation': 0.895144, 'summary/normalized/class: skeletal': 0.905008, 'summary/normalized/class: zwitterion': 0.894402, 'summary/normalized/coverage': 0.900616, 'summary/normalized/interval score': 10.549804, 'summary/normalized/mean half-width': 3.45226, 'summary/normalized/median half-width': 2.618145, 'summary/normalized/own-width fifth 1': 0.889763, 'summary/normalized/own-width fifth 2': 0.891468, 'summary/normalized/own-width fifth 3': 0.894699, 'summary/normalized/own-width fifth 4': 0.905002, 'summary/normalized/own-width fifth 5': 0.922148, 'summary/normalized/sigma fifth 1': 0.88976, 'summary/normalized/sigma fifth 2': 0.891471, 'summary/normalized/sigma fifth 3': 0.894699, 'summary/normalized/sigma fifth 4': 0.905002, 'summary/normalized/sigma fifth 5': 0.922148, 'summary/split/class: H transfer': 0.854161, 'summary/split/class: carbene': 0.969595, 'summary/split/class: fragmentation': 0.924629, 'summary/split/class: skeletal': 0.804917, 'summary/split/class: zwitterion': 0.925254, 'summary/split/coverage': 0.900827, 'summary/split/interval score': 13.567084, 'summary/split/mean half-width': 3.814366, 'summary/split/median half-width': 3.814366, 'summary/split/sigma fifth 1': 0.995825, 'summary/split/sigma fifth 2': 0.986795, 'summary/split/sigma fifth 3': 0.960906, 'summary/split/sigma fifth 4': 0.895366, 'summary/split/sigma fifth 5': 0.665243, 'widest_tenth_error/others': 1.338438, 'widest_tenth_error/widest': 4.164283}, 'nb4': {'average/Mondrian/H transfer': 0.901451, 'average/Mondrian/all': 0.902753, 'average/Mondrian/carbene': 0.902123, 'average/Mondrian/fragmentation': 0.902897, 'average/Mondrian/mean half-width': 3.91905, 'average/Mondrian/skeletal': 0.905282, 'average/Mondrian/zwitterion': 0.902812, 'average/one q for all/H transfer': 0.854635, 'average/one q for all/all': 0.901098, 'average/one q for all/carbene': 0.969333, 'average/one q for all/fragmentation': 0.924795, 'average/one q for all/mean half-width': 3.822305, 'average/one q for all/skeletal': 0.806187, 'average/one q for all/zwitterion': 0.92524, 'budget/100/class coverage 2.5%': 0.770952, 'budget/100/class coverage 97.5%': 0.996251, 'budget/100/mean finite half-width': 5.487601, 'budget/100/smallest class (mean n_g)': 13.917439, 'budget/100/splits with an infinite interval': 0.036667, 'budget/1000/class coverage 2.5%': 0.849413, 'budget/1000/class coverage 97.5%': 0.94626, 'budget/1000/mean finite half-width': 4.048102, 'budget/1000/smallest class (mean n_g)': 139.174389, 'budget/1000/splits with an infinite interval': 0.0, 'budget/200/class coverage 2.5%': 0.804124, 'budget/200/class coverage 97.5%': 0.980181, 'budget/200/mean finite half-width': 4.49897, 'budget/200/smallest class (mean n_g)': 27.834878, 'budget/200/splits with an infinite interval': 0.0, 'budget/2000/class coverage 2.5%': 0.862724, 'budget/2000/class coverage 97.5%': 0.937224, 'budget/2000/mean finite half-width': 4.0072, 'budget/2000/smallest class (mean n_g)': 278.348778, 'budget/2000/splits with an infinite interval': 0.0, 'budget/50/class coverage 2.5%': 0.737984, 'budget/50/class coverage 97.5%': 0.998583, 'budget/50/mean finite half-width': 5.955044, 'budget/50/smallest class (mean n_g)': 6.958719, 'budget/50/splits with an infinite interval': 0.98, 'budget/500/class coverage 2.5%': 0.83714, 'budget/500/class coverage 97.5%': 0.958185, 'budget/500/mean finite half-width': 4.126196, 'budget/500/smallest class (mean n_g)': 69.587195, 'budget/500/splits with an infinite interval': 0.0, 'class_spread/H transfer/2.5%': 0.857271, 'class_spread/H transfer/97.5%': 0.942937, 'class_spread/H transfer/Beta law 2.5%': 0.859354, 'class_spread/H transfer/Beta law 97.5%': 0.938224, 'class_spread/H transfer/mean coverage': 0.901451, 'class_spread/H transfer/mean n_g': 214, 'class_spread/carbene/2.5%': 0.855476, 'class_spread/carbene/97.5%': 0.942664, 'class_spread/carbene/Beta law 2.5%': 0.854968, 'class_spread/carbene/Beta law 97.5%': 0.942049, 'class_spread/carbene/mean coverage': 0.902123, 'class_spread/carbene/mean n_g': 174, 'class_spread/fragmentation/2.5%': 0.862557, 'class_spread/fragmentation/97.5%': 0.940765, 'class_spread/fragmentation/Beta law 2.5%': 0.861248, 'class_spread/fragmentation/Beta law 97.5%': 0.939092, 'class_spread/fragmentation/mean coverage': 0.902897, 'class_spread/fragmentation/mean n_g': 217, 'class_spread/skeletal/2.5%': 0.846813, 'class_spread/skeletal/97.5%': 0.946089, 'class_spread/skeletal/Beta law 2.5%': 0.845379, 'class_spread/skeletal/Beta law 97.5%': 0.943833, 'class_spread/skeletal/mean coverage': 0.905282, 'class_spread/skeletal/mean n_g': 139, 'class_spread/zwitterion/2.5%': 0.858203, 'class_spread/zwitterion/97.5%': 0.936124, 'class_spread/zwitterion/Beta law 2.5%': 0.863722, 'class_spread/zwitterion/Beta law 97.5%': 0.9358, 'class_spread/zwitterion/mean coverage': 0.902812, 'class_spread/zwitterion/mean n_g': 256, 'combined/Mondrian + normalized/best-covered class': 0.904068, 'combined/Mondrian + normalized/coverage': 0.902064, 'combined/Mondrian + normalized/easiest fifth': 0.89145, 'combined/Mondrian + normalized/hardest fifth': 0.920436, 'combined/Mondrian + normalized/mean half-width': 3.50719, 'combined/Mondrian + normalized/median half-width': 2.664412, 'combined/Mondrian + normalized/worst-covered class': 0.899882, 'combined/Mondrian/best-covered class': 0.905282, 'combined/Mondrian/coverage': 0.902753, 'combined/Mondrian/easiest fifth': 0.990754, 'combined/Mondrian/hardest fifth': 0.720182, 'combined/Mondrian/mean half-width': 3.91905, 'combined/Mondrian/median half-width': 3.592667, 'combined/Mondrian/worst-covered class': 0.901451, 'combined/normalized/best-covered class': 0.907313, 'combined/normalized/coverage': 0.899794, 'combined/normalized/easiest fifth': 0.889104, 'combined/normalized/hardest fifth': 0.92112, 'combined/normalized/mean half-width': 3.44391, 'combined/normalized/median half-width': 2.613038, 'combined/normalized/worst-covered class': 0.894137, 'combined/split (one q)/best-covered class': 0.969333, 'combined/split (one q)/coverage': 0.901098, 'combined/split (one q)/easiest fifth': 0.9958, 'combined/split (one q)/hardest fifth': 0.666955, 'combined/split (one q)/mean half-width': 3.822305, 'combined/split (one q)/median half-width': 3.822305, 'combined/split (one q)/worst-covered class': 0.806187, 'families/n_families_cal': 12, 'families/n_finite': 5, 'families/unseen_in_cal': 3, 'merged/(no family)/calibration reactions (mean)': 879.2, 'merged/(no family)/coverage (mean)': 0.901133, 'merged/(no family)/coverage 2.5%': 0.878782, 'merged/(no family)/coverage 97.5%': 0.921631, 'merged/(no family)/half-width (mean)': 4.040361, 'merged/(no family)/training reactions': 5233, 'merged/(other families)/calibration reactions (mean)': 68.915, 'merged/(other families)/coverage (mean)': 0.907125, 'merged/(other families)/coverage 2.5%': 0.811109, 'merged/(other families)/coverage 97.5%': 0.969068, 'merged/(other families)/half-width (mean)': 1.782723, 'merged/(other families)/training reactions': 380, 'merged/ketoenol/calibration reactions (mean)': 51.885, 'merged/ketoenol/coverage (mean)': 0.909665, 'merged/ketoenol/coverage 2.5%': 0.79846, 'merged/ketoenol/coverage 97.5%': 0.976574, 'merged/ketoenol/half-width (mean)': 2.780448, 'merged/ketoenol/training reactions': 378, 'single_split/coverage_global/H transfer': 0.85224, 'single_split/coverage_global/carbene': 0.968458, 'single_split/coverage_global/fragmentation': 0.924386, 'single_split/coverage_global/skeletal': 0.797059, 'single_split/coverage_global/zwitterion': 0.922601, 'single_split/coverage_mondrian/H transfer': 0.896092, 'single_split/coverage_mondrian/carbene': 0.915888, 'single_split/coverage_mondrian/fragmentation': 0.913043, 'single_split/coverage_mondrian/skeletal': 0.916176, 'single_split/coverage_mondrian/zwitterion': 0.907121, 'single_split/q_class/H transfer': 4.7676, 'single_split/q_class/carbene': 2.382235, 'single_split/q_class/fragmentation': 3.596611, 'single_split/q_class/skeletal': 6.587717, 'single_split/q_class/zwitterion': 3.370481, 'single_split/q_global': 3.771071}, 'nb5': {'alarm/different mix of classes/10': 0.0555, 'alarm/different mix of classes/100': 0.4595, 'alarm/different mix of classes/20': 0.1795, 'alarm/different mix of classes/200': 0.616, 'alarm/different mix of classes/50': 0.222, 'alarm/different mix of classes/500': 0.886, 'alarm/larger molecules/10': 0.0345, 'alarm/larger molecules/100': 0.2465, 'alarm/larger molecules/20': 0.112, 'alarm/larger molecules/200': 0.307, 'alarm/larger molecules/50': 0.114, 'alarm/larger molecules/500': 0.5375, 'alarm/same population (no shift)/10': 0.013, 'alarm/same population (no shift)/100': 0.0475, 'alarm/same population (no shift)/20': 0.0485, 'alarm/same population (no shift)/200': 0.041, 'alarm/same population (no shift)/50': 0.03, 'alarm/same population (no shift)/500': 0.0445, 'alarm/unseen class (skeletal)/10': 0.4785, 'alarm/unseen class (skeletal)/100': 1.0, 'alarm/unseen class (skeletal)/20': 0.8815, 'alarm/unseen class (skeletal)/200': 1.0, 'alarm/unseen class (skeletal)/50': 0.9915, 'alarm/unseen class (skeletal)/500': 1.0, 'alarm_exact/10': 0.013219, 'alarm_exact/100': 0.047323, 'alarm_exact/20': 0.044676, 'alarm_exact/200': 0.039453, 'alarm_exact/50': 0.027494, 'alarm_exact/500': 0.043372, 'alarm_repeated_looks': 0.1385, 'alarm_thresholds/10': 4, 'alarm_thresholds/100': 16, 'alarm_thresholds/20': 5, 'alarm_thresholds/200': 29, 'alarm_thresholds/50': 10, 'alarm_thresholds/500': 65, 'auc/different mix of classes/AUC of the domain classifier': 0.727994, 'auc/different mix of classes/coverage of the Δ-learning intervals': 0.849017, 'auc/larger molecules/AUC of the domain classifier': 1.0, 'auc/larger molecules/coverage of the Δ-learning intervals': 0.869224, 'auc/same population (no shift)/AUC of the domain classifier': 0.479679, 'auc/same population (no shift)/coverage of the Δ-learning intervals': 0.901303, 'auc/unseen class (skeletal)/AUC of the domain classifier': 1.0, 'auc/unseen class (skeletal)/coverage of the Δ-learning intervals': 0.652542, 'clusters/Structure-only model/mean coverage, single reactions': 0.900414, 'clusters/Structure-only model/mean coverage, whole reactants': 0.899878, 'clusters/Structure-only model/standard deviation, single reactions': 0.009585, 'clusters/Structure-only model/standard deviation, whole reactants': 0.01172, 'clusters/Δ-learning model/mean coverage, single reactions': 0.900559, 'clusters/Δ-learning model/mean coverage, whole reactants': 0.900106, 'clusters/Δ-learning model/standard deviation, single reactions': 0.009514, 'clusters/Δ-learning model/standard deviation, whole reactants': 0.01193, 'clusters_extra_scatter': 0.238439, 'dependence/Structure-only model/coverage': 0.895457, 'dependence/Structure-only model/expected': 0.900383, 'dependence/Structure-only model/loss': 0.492619, 'dependence/Structure-only model/loss_se': 0.147338, 'dependence/Structure-only model/width_ratio': 0.983442, 'dependence/Δ-learning model/coverage': 0.894696, 'dependence/Δ-learning model/expected': 0.900383, 'dependence/Δ-learning model/loss': 0.568644, 'dependence/Δ-learning model/loss_se': 0.150703, 'dependence/Δ-learning model/width_ratio': 0.974772, 'mix/Structure-only model | Mondrian by class/coverage': 0.902342, 'mix/Structure-only model | Mondrian by class/mean half-width': 21.553901, 'mix/Structure-only model | Mondrian by class/s.d. of coverage': 0.018016, 'mix/Structure-only model | split conformal (no repair)/coverage': 0.88075, 'mix/Structure-only model | split conformal (no repair)/mean half-width': 20.114122, 'mix/Structure-only model | split conformal (no repair)/s.d. of coverage': 0.016321, 'mix/Structure-only model | weighted, class-frequency ratios/coverage': 0.902108, 'mix/Structure-only model | weighted, class-frequency ratios/mean half-width': 21.795685, 'mix/Structure-only model | weighted, class-frequency ratios/s.d. of coverage': 0.019104, 'mix/Δ-learning model | Mondrian by class/coverage': 0.902183, 'mix/Δ-learning model | Mondrian by class/mean half-width': 5.141913, 'mix/Δ-learning model | Mondrian by class/s.d. of coverage': 0.017977, 'mix/Δ-learning model | split conformal (no repair)/coverage': 0.849017, 'mix/Δ-learning model | split conformal (no repair)/mean half-width': 3.805376, 'mix/Δ-learning model | split conformal (no repair)/s.d. of coverage': 0.016601, 'mix/Δ-learning model | weighted, class-frequency ratios/coverage': 0.9013, 'mix/Δ-learning model | weighted, class-frequency ratios/mean half-width': 5.205457, 'mix/Δ-learning model | weighted, class-frequency ratios/s.d. of coverage': 0.01879, 'mix/Δ-learning model | weighted, learned weights/coverage': 0.89725, 'mix/Δ-learning model | weighted, learned weights/mean half-width': 5.048138, 'mix/Δ-learning model | weighted, learned weights/s.d. of coverage': 0.020643, 'mix_diagnostics/AUC': 0.727994, 'mix_diagnostics/effective sample size': 487.211995, 'recalibration/10/5%': 0.734915, 'recalibration/10/95%': 0.995567, 'recalibration/10/mean coverage': 0.906243, 'recalibration/10/median half-width': 5.256745, 'recalibration/100/5%': 0.853206, 'recalibration/100/95%': 0.946966, 'recalibration/100/mean coverage': 0.903878, 'recalibration/100/median half-width': 4.363069, 'recalibration/20/5%': 0.778675, 'recalibration/20/95%': 0.983554, 'recalibration/20/mean coverage': 0.904117, 'recalibration/20/median half-width': 4.695256, 'recalibration/200/5%': 0.858094, 'recalibration/200/95%': 0.934736, 'recalibration/200/mean coverage': 0.90034, 'recalibration/200/median half-width': 4.25858, 'recalibration/50/5%': 0.827336, 'recalibration/50/95%': 0.956625, 'recalibration/50/mean coverage': 0.899304, 'recalibration/50/median half-width': 4.301581, 'recalibration/500/5%': 0.874006, 'recalibration/500/95%': 0.922047, 'recalibration/500/mean coverage': 0.899964, 'recalibration/500/median half-width': 4.221877, 'same_population/delta': 0.901303, 'same_population/structure': 0.899523, 'single_example/coverage_learned': 0.883333, 'single_example/coverage_plain': 0.843333, 'single_example/coverage_weighted': 0.895, 'single_example/q_plain': 3.829824, 'single_example/share_hard_cal': 0.147333, 'size/Structure-only model/MAE, calibration set': 8.14796, 'size/Structure-only model/MAE, larger molecules': 9.227034, 'size/Structure-only model/coverage on larger molecules': 0.882783, 'size/Structure-only model/half-width needed for 90%': 21.254, 'size/Structure-only model/half-width used': 19.594703, 'size/Δ-learning model/MAE, calibration set': 1.557411, 'size/Δ-learning model/MAE, larger molecules': 1.810033, 'size/Δ-learning model/coverage on larger molecules': 0.869224, 'size/Δ-learning model/half-width needed for 90%': 4.205089, 'size/Δ-learning model/half-width used': 3.624266, 'size_auc': 1.0, 'size_weighted_infinite': 1.0, 'unseen_class/Structure-only model | H transfer/MAE, known classes': 8.525765, 'unseen_class/Structure-only model | H transfer/MAE, unseen class': 12.17448, 'unseen_class/Structure-only model | H transfer/coverage after 50 reference calculations': 0.901505, 'unseen_class/Structure-only model | H transfer/coverage on the unseen class': 0.817255, 'unseen_class/Structure-only model | H transfer/half-width after': 25.827954, 'unseen_class/Structure-only model | H transfer/half-width used': 20.258024, 'unseen_class/Structure-only model | carbene/MAE, known classes': 8.81635, 'unseen_class/Structure-only model | carbene/MAE, unseen class': 14.367148, 'unseen_class/Structure-only model | carbene/coverage after 50 reference calculations': 0.904236, 'unseen_class/Structure-only model | carbene/coverage on the unseen class': 0.739382, 'unseen_class/Structure-only model | carbene/half-width after': 29.251705, 'unseen_class/Structure-only model | carbene/half-width used': 20.268273, 'unseen_class/Structure-only model | fragmentation/MAE, known classes': 8.958574, 'unseen_class/Structure-only model | fragmentation/MAE, unseen class': 14.262648, 'unseen_class/Structure-only model | fragmentation/coverage after 50 reference calculations': 0.904588, 'unseen_class/Structure-only model | fragmentation/coverage on the unseen class': 0.752923, 'unseen_class/Structure-only model | fragmentation/half-width after': 28.974119, 'unseen_class/Structure-only model | fragmentation/half-width used': 20.642053, 'unseen_class/Structure-only model | skeletal/MAE, known classes': 8.437048, 'unseen_class/Structure-only model | skeletal/MAE, unseen class': 20.356517, 'unseen_class/Structure-only model | skeletal/coverage after 50 reference calculations': 0.898935, 'unseen_class/Structure-only model | skeletal/coverage on the unseen class': 0.5, 'unseen_class/Structure-only model | skeletal/half-width after': 39.631001, 'unseen_class/Structure-only model | skeletal/half-width used': 19.210691, 'unseen_class/Structure-only model | zwitterion/MAE, known classes': 8.031977, 'unseen_class/Structure-only model | zwitterion/MAE, unseen class': 18.518988, 'unseen_class/Structure-only model | zwitterion/coverage after 50 reference calculations': 0.901222, 'unseen_class/Structure-only model | zwitterion/coverage on the unseen class': 0.531353, 'unseen_class/Structure-only model | zwitterion/half-width after': 34.976957, 'unseen_class/Structure-only model | zwitterion/half-width used': 18.356577, 'unseen_class/Δ-learning model | H transfer/MAE, known classes': 1.539215, 'unseen_class/Δ-learning model | H transfer/MAE, unseen class': 2.590063, 'unseen_class/Δ-learning model | H transfer/coverage after 50 reference calculations': 0.905271, 'unseen_class/Δ-learning model | H transfer/coverage on the unseen class': 0.772549, 'unseen_class/Δ-learning model | H transfer/half-width after': 6.131933, 'unseen_class/Δ-learning model | H transfer/half-width used': 3.604055, 'unseen_class/Δ-learning model | carbene/MAE, known classes': 1.770994, 'unseen_class/Δ-learning model | carbene/MAE, unseen class': 2.102658, 'unseen_class/Δ-learning model | carbene/coverage after 50 reference calculations': 0.904209, 'unseen_class/Δ-learning model | carbene/coverage on the unseen class': 0.853282, 'unseen_class/Δ-learning model | carbene/half-width after': 5.370352, 'unseen_class/Δ-learning model | carbene/half-width used': 4.061002, 'unseen_class/Δ-learning model | fragmentation/MAE, known classes': 1.631336, 'unseen_class/Δ-learning model | fragmentation/MAE, unseen class': 2.389908, 'unseen_class/Δ-learning model | fragmentation/coverage after 50 reference calculations': 0.906926, 'unseen_class/Δ-learning model | fragmentation/coverage on the unseen class': 0.791894, 'unseen_class/Δ-learning model | fragmentation/half-width after': 5.459833, 'unseen_class/Δ-learning model | fragmentation/half-width used': 3.895433, 'unseen_class/Δ-learning model | skeletal/MAE, known classes': 1.474938, 'unseen_class/Δ-learning model | skeletal/MAE, unseen class': 3.453914, 'unseen_class/Δ-learning model | skeletal/coverage after 50 reference calculations': 0.904171, 'unseen_class/Δ-learning model | skeletal/coverage on the unseen class': 0.652542, 'unseen_class/Δ-learning model | skeletal/half-width after': 8.397182, 'unseen_class/Δ-learning model | skeletal/half-width used': 3.491246, 'unseen_class/Δ-learning model | zwitterion/MAE, known classes': 1.679546, 'unseen_class/Δ-learning model | zwitterion/MAE, unseen class': 2.134405, 'unseen_class/Δ-learning model | zwitterion/coverage after 50 reference calculations': 0.899461, 'unseen_class/Δ-learning model | zwitterion/coverage on the unseen class': 0.861386, 'unseen_class/Δ-learning model | zwitterion/half-width after': 4.795599, 'unseen_class/Δ-learning model | zwitterion/half-width used': 3.942115}, 'nb6': {'experiment/delta/A: predicted barrier below c/FDP, 90th percentile': 0.051586, 'experiment/delta/A: predicted barrier below c/FDR (mean FDP)': 0.045783, 'experiment/delta/A: predicted barrier below c/mean FDP of the non-empty lists': 0.045783, 'experiment/delta/A: predicted barrier below c/power (mean)': 0.93399, 'experiment/delta/A: predicted barrier below c/selected (mean)': 345.146667, 'experiment/delta/A: predicted barrier below c/splits with an empty list': 0.0, 'experiment/delta/B: 90% interval below c/FDP, 90th percentile': 0.0, 'experiment/delta/B: 90% interval below c/FDR (mean FDP)': 0.0, 'experiment/delta/B: 90% interval below c/mean FDP of the non-empty lists': 0.0, 'experiment/delta/B: 90% interval below c/power (mean)': 0.695495, 'experiment/delta/B: 90% interval below c/selected (mean)': 245.243333, 'experiment/delta/B: 90% interval below c/splits with an empty list': 0.0, 'experiment/delta/C: p-value below q, no correction | q=0.02/FDP, 90th percentile': 0.276173, 'experiment/delta/C: p-value below q, no correction | q=0.02/FDR (mean FDP)': 0.220016, 'experiment/delta/C: p-value below q, no correction | q=0.02/mean FDP of the non-empty lists': 0.220016, 'experiment/delta/C: p-value below q, no correction | q=0.02/power (mean)': 0.983485, 'experiment/delta/C: p-value below q, no correction | q=0.02/selected (mean)': 445.956667, 'experiment/delta/C: p-value below q, no correction | q=0.02/splits with an empty list': 0.0, 'experiment/delta/C: p-value below q, no correction | q=0.05/FDP, 90th percentile': 0.462179, 'experiment/delta/C: p-value below q, no correction | q=0.05/FDR (mean FDP)': 0.414612, 'experiment/delta/C: p-value below q, no correction | q=0.05/mean FDP of the non-empty lists': 0.414612, 'experiment/delta/C: p-value below q, no correction | q=0.05/power (mean)': 0.986388, 'experiment/delta/C: p-value below q, no correction | q=0.05/selected (mean)': 596.466667, 'experiment/delta/C: p-value below q, no correction | q=0.05/splits with an empty list': 0.0, 'experiment/delta/C: p-value below q, no correction | q=0.10/FDP, 90th percentile': 0.616568, 'experiment/delta/C: p-value below q, no correction | q=0.10/FDR (mean FDP)': 0.584545, 'experiment/delta/C: p-value below q, no correction | q=0.10/mean FDP of the non-empty lists': 0.584545, 'experiment/delta/C: p-value below q, no correction | q=0.10/power (mean)': 0.995199, 'experiment/delta/C: p-value below q, no correction | q=0.10/selected (mean)': 847.54, 'experiment/delta/C: p-value below q, no correction | q=0.10/splits with an empty list': 0.0, 'experiment/delta/C: p-value below q, no correction | q=0.20/FDP, 90th percentile': 0.75504, 'experiment/delta/C: p-value below q, no correction | q=0.20/FDR (mean FDP)': 0.736369, 'experiment/delta/C: p-value below q, no correction | q=0.20/mean FDP of the non-empty lists': 0.736369, 'experiment/delta/C: p-value below q, no correction | q=0.20/power (mean)': 0.997646, 'experiment/delta/C: p-value below q, no correction | q=0.20/selected (mean)': 1338.24, 'experiment/delta/C: p-value below q, no correction | q=0.20/splits with an empty list': 0.0, 'experiment/delta/C: p-value below q, no correction | q=0.30/FDP, 90th percentile': 0.818615, 'experiment/delta/C: p-value below q, no correction | q=0.30/FDR (mean FDP)': 0.807298, 'experiment/delta/C: p-value below q, no correction | q=0.30/mean FDP of the non-empty lists': 0.807298, 'experiment/delta/C: p-value below q, no correction | q=0.30/power (mean)': 1.0, 'experiment/delta/C: p-value below q, no correction | q=0.30/selected (mean)': 1833.36, 'experiment/delta/C: p-value below q, no correction | q=0.30/splits with an empty list': 0.0, 'experiment/delta/conformal selection | q=0.02/FDP, 90th percentile': 0.037167, 'experiment/delta/conformal selection | q=0.02/FDR (mean FDP)': 0.015907, 'experiment/delta/conformal selection | q=0.02/mean FDP of the non-empty lists': 0.015907, 'experiment/delta/conformal selection | q=0.02/power (mean)': 0.838402, 'experiment/delta/conformal selection | q=0.02/selected (mean)': 300.733333, 'experiment/delta/conformal selection | q=0.02/splits with an empty list': 0.0, 'experiment/delta/conformal selection | q=0.05/FDP, 90th percentile': 0.0758, 'experiment/delta/conformal selection | q=0.05/FDR (mean FDP)': 0.044451, 'experiment/delta/conformal selection | q=0.05/mean FDP of the non-empty lists': 0.044451, 'experiment/delta/conformal selection | q=0.05/power (mean)': 0.922244, 'experiment/delta/conformal selection | q=0.05/selected (mean)': 340.86, 'experiment/delta/conformal selection | q=0.05/splits with an empty list': 0.0, 'experiment/delta/conformal selection | q=0.10/FDP, 90th percentile': 0.151256, 'experiment/delta/conformal selection | q=0.10/FDR (mean FDP)': 0.097613, 'experiment/delta/conformal selection | q=0.10/mean FDP of the non-empty lists': 0.097613, 'experiment/delta/conformal selection | q=0.10/power (mean)': 0.968289, 'experiment/delta/conformal selection | q=0.10/selected (mean)': 379.313333, 'experiment/delta/conformal selection | q=0.10/splits with an empty list': 0.0, 'experiment/delta/conformal selection | q=0.20/FDP, 90th percentile': 0.268969, 'experiment/delta/conformal selection | q=0.20/FDR (mean FDP)': 0.196612, 'experiment/delta/conformal selection | q=0.20/mean FDP of the non-empty lists': 0.196612, 'experiment/delta/conformal selection | q=0.20/power (mean)': 0.982048, 'experiment/delta/conformal selection | q=0.20/selected (mean)': 433.023333, 'experiment/delta/conformal selection | q=0.20/splits with an empty list': 0.0, 'experiment/delta/conformal selection | q=0.30/FDP, 90th percentile': 0.378764, 'experiment/delta/conformal selection | q=0.30/FDR (mean FDP)': 0.296837, 'experiment/delta/conformal selection | q=0.30/mean FDP of the non-empty lists': 0.296837, 'experiment/delta/conformal selection | q=0.30/power (mean)': 0.985516, 'experiment/delta/conformal selection | q=0.30/selected (mean)': 497.91, 'experiment/delta/conformal selection | q=0.30/splits with an empty list': 0.0, 'experiment/structure/A: predicted barrier below c/FDP, 90th percentile': 0.242798, 'experiment/structure/A: predicted barrier below c/FDR (mean FDP)': 0.229084, 'experiment/structure/A: predicted barrier below c/mean FDP of the non-empty lists': 0.229084, 'experiment/structure/A: predicted barrier below c/power (mean)': 0.525224, 'experiment/structure/A: predicted barrier below c/selected (mean)': 240.243333, 'experiment/structure/A: predicted barrier below c/splits with an empty list': 0.0, 'experiment/structure/B: 90% interval below c/FDP, 90th percentile': 0.0, 'experiment/structure/B: 90% interval below c/FDR (mean FDP)': 0.0, 'experiment/structure/B: 90% interval below c/mean FDP of the non-empty lists': 0.0, 'experiment/structure/B: 90% interval below c/power (mean)': 0.001361, 'experiment/structure/B: 90% interval below c/selected (mean)': 0.48, 'experiment/structure/B: 90% interval below c/splits with an empty list': 0.52, 'experiment/structure/C: p-value below q, no correction | q=0.02/FDP, 90th percentile': 0.366717, 'experiment/structure/C: p-value below q, no correction | q=0.02/FDR (mean FDP)': 0.313124, 'experiment/structure/C: p-value below q, no correction | q=0.02/mean FDP of the non-empty lists': 0.313124, 'experiment/structure/C: p-value below q, no correction | q=0.02/power (mean)': 0.60226, 'experiment/structure/C: p-value below q, no correction | q=0.02/selected (mean)': 311.436667, 'experiment/structure/C: p-value below q, no correction | q=0.02/splits with an empty list': 0.0, 'experiment/structure/C: p-value below q, no correction | q=0.05/FDP, 90th percentile': 0.530085, 'experiment/structure/C: p-value below q, no correction | q=0.05/FDR (mean FDP)': 0.48837, 'experiment/structure/C: p-value below q, no correction | q=0.05/mean FDP of the non-empty lists': 0.48837, 'experiment/structure/C: p-value below q, no correction | q=0.05/power (mean)': 0.727258, 'experiment/structure/C: p-value below q, no correction | q=0.05/selected (mean)': 503.75, 'experiment/structure/C: p-value below q, no correction | q=0.05/splits with an empty list': 0.0, 'experiment/structure/C: p-value below q, no correction | q=0.10/FDP, 90th percentile': 0.648387, 'experiment/structure/C: p-value below q, no correction | q=0.10/FDR (mean FDP)': 0.620886, 'experiment/structure/C: p-value below q, no correction | q=0.10/mean FDP of the non-empty lists': 0.620886, 'experiment/structure/C: p-value below q, no correction | q=0.10/power (mean)': 0.847397, 'experiment/structure/C: p-value below q, no correction | q=0.10/selected (mean)': 790.963333, 'experiment/structure/C: p-value below q, no correction | q=0.10/splits with an empty list': 0.0, 'experiment/structure/C: p-value below q, no correction | q=0.20/FDP, 90th percentile': 0.770034, 'experiment/structure/C: p-value below q, no correction | q=0.20/FDR (mean FDP)': 0.75225, 'experiment/structure/C: p-value below q, no correction | q=0.20/mean FDP of the non-empty lists': 0.75225, 'experiment/structure/C: p-value below q, no correction | q=0.20/power (mean)': 0.919259, 'experiment/structure/C: p-value below q, no correction | q=0.20/selected (mean)': 1311.986667, 'experiment/structure/C: p-value below q, no correction | q=0.20/splits with an empty list': 0.0, 'experiment/structure/C: p-value below q, no correction | q=0.30/FDP, 90th percentile': 0.825654, 'experiment/structure/C: p-value below q, no correction | q=0.30/FDR (mean FDP)': 0.814478, 'experiment/structure/C: p-value below q, no correction | q=0.30/mean FDP of the non-empty lists': 0.814478, 'experiment/structure/C: p-value below q, no correction | q=0.30/power (mean)': 0.954038, 'experiment/structure/C: p-value below q, no correction | q=0.30/selected (mean)': 1816.86, 'experiment/structure/C: p-value below q, no correction | q=0.30/splits with an empty list': 0.0, 'experiment/structure/conformal selection | q=0.02/FDP, 90th percentile': 0.0, 'experiment/structure/conformal selection | q=0.02/FDR (mean FDP)': 0.0, 'experiment/structure/conformal selection | q=0.02/power (mean)': 0.0, 'experiment/structure/conformal selection | q=0.02/selected (mean)': 0.0, 'experiment/structure/conformal selection | q=0.02/splits with an empty list': 1.0, 'experiment/structure/conformal selection | q=0.05/FDP, 90th percentile': 0.114396, 'experiment/structure/conformal selection | q=0.05/FDR (mean FDP)': 0.019036, 'experiment/structure/conformal selection | q=0.05/mean FDP of the non-empty lists': 0.126904, 'experiment/structure/conformal selection | q=0.05/power (mean)': 0.044021, 'experiment/structure/conformal selection | q=0.05/selected (mean)': 17.866667, 'experiment/structure/conformal selection | q=0.05/splits with an empty list': 0.85, 'experiment/structure/conformal selection | q=0.10/FDP, 90th percentile': 0.17322, 'experiment/structure/conformal selection | q=0.10/FDR (mean FDP)': 0.077135, 'experiment/structure/conformal selection | q=0.10/mean FDP of the non-empty lists': 0.120524, 'experiment/structure/conformal selection | q=0.10/power (mean)': 0.192577, 'experiment/structure/conformal selection | q=0.10/selected (mean)': 79.126667, 'experiment/structure/conformal selection | q=0.10/splits with an empty list': 0.36, 'experiment/structure/conformal selection | q=0.20/FDP, 90th percentile': 0.291403, 'experiment/structure/conformal selection | q=0.20/FDR (mean FDP)': 0.197565, 'experiment/structure/conformal selection | q=0.20/mean FDP of the non-empty lists': 0.198226, 'experiment/structure/conformal selection | q=0.20/power (mean)': 0.458535, 'experiment/structure/conformal selection | q=0.20/selected (mean)': 208.003333, 'experiment/structure/conformal selection | q=0.20/splits with an empty list': 0.003333, 'experiment/structure/conformal selection | q=0.30/FDP, 90th percentile': 0.394324, 'experiment/structure/conformal selection | q=0.30/FDR (mean FDP)': 0.295267, 'experiment/structure/conformal selection | q=0.30/mean FDP of the non-empty lists': 0.295267, 'experiment/structure/conformal selection | q=0.30/power (mean)': 0.5821, 'experiment/structure/conformal selection | q=0.30/selected (mean)': 298.74, 'experiment/structure/conformal selection | q=0.30/splits with an empty list': 0.0, 'eyring_half_life_ms': 2.813669, 'hit_rate_hold': 0.071272, 'n_candidates': 4935, 'resolution/Structure-only model | n=100/FDR': 0.002672, 'resolution/Structure-only model | n=100/power': 0.003759, 'resolution/Structure-only model | n=100/smallest non-empty list observed': 670, 'resolution/Structure-only model | n=100/smallest possible list m/(q(n+1))': 578, 'resolution/Structure-only model | n=100/splits with an empty list': 0.995, 'resolution/Structure-only model | n=1000/FDR': 0.080344, 'resolution/Structure-only model | n=1000/power': 0.198958, 'resolution/Structure-only model | n=1000/smallest non-empty list observed': 58, 'resolution/Structure-only model | n=1000/smallest possible list m/(q(n+1))': 50, 'resolution/Structure-only model | n=1000/splits with an empty list': 0.31, 'resolution/Structure-only model | n=200/FDR': 0.020917, 'resolution/Structure-only model | n=200/power': 0.042639, 'resolution/Structure-only model | n=200/smallest non-empty list observed': 286, 'resolution/Structure-only model | n=200/smallest possible list m/(q(n+1))': 286, 'resolution/Structure-only model | n=200/splits with an empty list': 0.925, 'resolution/Structure-only model | n=2000/FDR': 0.090403, 'resolution/Structure-only model | n=2000/power': 0.235114, 'resolution/Structure-only model | n=2000/smallest non-empty list observed': 20, 'resolution/Structure-only model | n=2000/smallest possible list m/(q(n+1))': 20, 'resolution/Structure-only model | n=2000/splits with an empty list': 0.03, 'resolution/Structure-only model | n=500/FDR': 0.059026, 'resolution/Structure-only model | n=500/power': 0.141929, 'resolution/Structure-only model | n=500/smallest non-empty list observed': 109, 'resolution/Structure-only model | n=500/smallest possible list m/(q(n+1))': 109, 'resolution/Structure-only model | n=500/splits with an empty list': 0.61, 'resolution/Δ-learning model | n=100/FDR': 0.016041, 'resolution/Δ-learning model | n=100/power': 0.044351, 'resolution/Δ-learning model | n=100/smallest non-empty list observed': 592, 'resolution/Δ-learning model | n=100/smallest possible list m/(q(n+1))': 578, 'resolution/Δ-learning model | n=100/splits with an empty list': 0.955, 'resolution/Δ-learning model | n=1000/FDR': 0.094574, 'resolution/Δ-learning model | n=1000/power': 0.96755, 'resolution/Δ-learning model | n=1000/smallest non-empty list observed': 312, 'resolution/Δ-learning model | n=1000/smallest possible list m/(q(n+1))': 50, 'resolution/Δ-learning model | n=1000/splits with an empty list': 0.0, 'resolution/Δ-learning model | n=200/FDR': 0.065424, 'resolution/Δ-learning model | n=200/power': 0.924569, 'resolution/Δ-learning model | n=200/smallest non-empty list observed': 307, 'resolution/Δ-learning model | n=200/smallest possible list m/(q(n+1))': 286, 'resolution/Δ-learning model | n=200/splits with an empty list': 0.0, 'resolution/Δ-learning model | n=2000/FDR': 0.097194, 'resolution/Δ-learning model | n=2000/power': 0.972028, 'resolution/Δ-learning model | n=2000/smallest non-empty list observed': 262, 'resolution/Δ-learning model | n=2000/smallest possible list m/(q(n+1))': 20, 'resolution/Δ-learning model | n=2000/splits with an empty list': 0.0, 'resolution/Δ-learning model | n=500/FDR': 0.089325, 'resolution/Δ-learning model | n=500/power': 0.95628, 'resolution/Δ-learning model | n=500/smallest non-empty list observed': 331, 'resolution/Δ-learning model | n=500/smallest possible list m/(q(n+1))': 109, 'resolution/Δ-learning model | n=500/splits with an empty list': 0.0, 'scores/Structure-only model | q=0.05 | FDR | (mu - c) / sigma': 0.023545, 'scores/Structure-only model | q=0.05 | FDR | mu - c': 0.019036, 'scores/Structure-only model | q=0.05 | power | (mu - c) / sigma': 0.065497, 'scores/Structure-only model | q=0.05 | power | mu - c': 0.044021, 'scores/Structure-only model | q=0.10 | FDR | (mu - c) / sigma': 0.083131, 'scores/Structure-only model | q=0.10 | FDR | mu - c': 0.077135, 'scores/Structure-only model | q=0.10 | power | (mu - c) / sigma': 0.291772, 'scores/Structure-only model | q=0.10 | power | mu - c': 0.192577, 'scores/Structure-only model | q=0.20 | FDR | (mu - c) / sigma': 0.194856, 'scores/Structure-only model | q=0.20 | FDR | mu - c': 0.197565, 'scores/Structure-only model | q=0.20 | power | (mu - c) / sigma': 0.466433, 'scores/Structure-only model | q=0.20 | power | mu - c': 0.458535, 'scores/Δ-learning model | q=0.05 | FDR | (mu - c) / sigma': 0.043934, 'scores/Δ-learning model | q=0.05 | FDR | mu - c': 0.044451, 'scores/Δ-learning model | q=0.05 | power | (mu - c) / sigma': 0.92334, 'scores/Δ-learning model | q=0.05 | power | mu - c': 0.922244, 'scores/Δ-learning model | q=0.10 | FDR | (mu - c) / sigma': 0.095122, 'scores/Δ-learning model | q=0.10 | FDR | mu - c': 0.097613, 'scores/Δ-learning model | q=0.10 | power | (mu - c) / sigma': 0.97213, 'scores/Δ-learning model | q=0.10 | power | mu - c': 0.968289, 'scores/Δ-learning model | q=0.20 | FDR | (mu - c) / sigma': 0.19645, 'scores/Δ-learning model | q=0.20 | FDR | mu - c': 0.196612, 'scores/Δ-learning model | q=0.20 | power | (mu - c) / sigma': 0.987454, 'scores/Δ-learning model | q=0.20 | power | mu - c': 0.982048, 'single_split/Structure-only model | A: predicted barrier below c/FDP': 0.228448, 'single_split/Structure-only model | A: predicted barrier below c/false discoveries': 53, 'single_split/Structure-only model | A: predicted barrier below c/hits found': 179, 'single_split/Structure-only model | A: predicted barrier below c/power': 0.523392, 'single_split/Structure-only model | A: predicted barrier below c/selected': 232, 'single_split/Structure-only model | B: 90% interval entirely below c/FDP': 0.0, 'single_split/Structure-only model | B: 90% interval entirely below c/false discoveries': 0, 'single_split/Structure-only model | B: 90% interval entirely below c/hits found': 0, 'single_split/Structure-only model | B: 90% interval entirely below c/power': 0.0, 'single_split/Structure-only model | B: 90% interval entirely below c/selected': 0, 'single_split/Structure-only model | C: p-value below 0.10 for each reaction/FDP': 0.652071, 'single_split/Structure-only model | C: p-value below 0.10 for each reaction/false discoveries': 551, 'single_split/Structure-only model | C: p-value below 0.10 for each reaction/hits found': 294, 'single_split/Structure-only model | C: p-value below 0.10 for each reaction/power': 0.859649, 'single_split/Structure-only model | C: p-value below 0.10 for each reaction/selected': 845, 'single_split/Δ-learning model | A: predicted barrier below c/FDP': 0.047904, 'single_split/Δ-learning model | A: predicted barrier below c/false discoveries': 16, 'single_split/Δ-learning model | A: predicted barrier below c/hits found': 318, 'single_split/Δ-learning model | A: predicted barrier below c/power': 0.929825, 'single_split/Δ-learning model | A: predicted barrier below c/selected': 334, 'single_split/Δ-learning model | B: 90% interval entirely below c/FDP': 0.0, 'single_split/Δ-learning model | B: 90% interval entirely below c/false discoveries': 0, 'single_split/Δ-learning model | B: 90% interval entirely below c/hits found': 241, 'single_split/Δ-learning model | B: 90% interval entirely below c/power': 0.704678, 'single_split/Δ-learning model | B: 90% interval entirely below c/selected': 241, 'single_split/Δ-learning model | C: p-value below 0.10 for each reaction/FDP': 0.601874, 'single_split/Δ-learning model | C: p-value below 0.10 for each reaction/false discoveries': 514, 'single_split/Δ-learning model | C: p-value below 0.10 for each reaction/hits found': 340, 'single_split/Δ-learning model | C: p-value below 0.10 for each reaction/power': 0.994152, 'single_split/Δ-learning model | C: p-value below 0.10 for each reaction/selected': 854, 'single_split_bh/delta/FDP': 0.119048, 'single_split_bh/delta/false discoveries': 45, 'single_split_bh/delta/hits found': 333, 'single_split_bh/delta/power': 0.973684, 'single_split_bh/delta/selected': 378, 'single_split_bh/structure/FDP': 0.137097, 'single_split_bh/structure/false discoveries': 17, 'single_split_bh/structure/hits found': 107, 'single_split_bh/structure/power': 0.312865, 'single_split_bh/structure/selected': 124, 'threshold': 50.0, 'versus_rule_a/Structure-only model/FDR of conformal selection at q = FDR of rule A': 0.229291, 'versus_rule_a/Structure-only model/FDR of rule A': 0.229084, 'versus_rule_a/Structure-only model/its power': 0.505286, 'versus_rule_a/Structure-only model/power of rule A': 0.525224, 'versus_rule_a/Δ-learning model/FDR of conformal selection at q = FDR of rule A': 0.042549, 'versus_rule_a/Δ-learning model/FDR of rule A': 0.045783, 'versus_rule_a/Δ-learning model/its power': 0.915492, 'versus_rule_a/Δ-learning model/power of rule A': 0.93399}})
# <<< reference results


def compare(name: str, results: dict) -> dict:
    """Compare the numbers of one notebook with the reference run.

    Returns the lists of keys that are 'identical', 'close', 'different' or 'missing'. Close means:
    within 0.02 for numbers between -1 and 1 (coverages, rates, fractions) and within 5% for larger
    ones; counts that do not depend on the fitted models (sizes of data sets, ranks, thresholds)
    must be identical. A number that is absent or not finite counts as missing.
    """
    ref = {k: v for k, v in REFERENCE_RESULTS.get(name, {}).items() if not isinstance(v, (str, bool))}
    new = flatten(_plain(results))
    out = {"identical": [], "close": [], "different": [], "missing": []}
    for key, expected in ref.items():
        value = new.get(key)
        if key not in new or isinstance(value, (str, bool)):
            out["missing"].append(key)
        elif expected is None:                                # infinite in the reference run
            out["identical" if value is None else "different"].append(key)
        elif value is None:                                   # finite in the reference run, not finite here
            out["missing"].append(key)
        elif value == expected:
            out["identical"].append(key)
        elif isinstance(expected, int) and not any(tag in key for tag in MODEL_DEPENDENT_COUNTS):
            out["different"].append(key)
        elif abs(value - expected) <= (0.02 if abs(expected) <= 1 else 0.05 * abs(expected)):
            out["close"].append(key)
        else:
            out["different"].append(key)
    out["reference"], out["new"] = ref, new
    return out


def _what_differs() -> str:
    """One sentence on how this environment differs from that of the reference run."""
    ref_env, env = REFERENCE_RESULTS.get("_environment", {}), environment()
    changed = [f"{lib} {env.get(lib)} (reference run: {ref_env.get(lib)})" for lib in LIBRARIES
               if ref_env.get(lib) is not None and env.get(lib) != ref_env.get(lib)]
    if not ref_env:
        return ""
    if changed:
        return ("   This environment differs from that of the reference run in: " + "; ".join(changed) + ".\n"
                "   Small differences are then expected: different library versions do not fit gradient-boosted trees identically.")
    return ("   The library versions are those of the reference run, so the numbers should be identical. If you have changed\n"
            "   the notebook (for example for an exercise), differences are expected. Otherwise check that primer.py is the\n"
            "   one that belongs to this notebook and that the data were reported as the Zenodo files.")


def check(name: str, results: dict) -> bool:
    """Print how the numbers just computed compare with the reference run; True if none differs or is missing."""
    if not REFERENCE_RESULTS.get(name):
        print("No reference values are stored for this notebook yet.")
        return True
    c = compare(name, results)
    total = sum(len(c[k]) for k in ("identical", "close", "different", "missing"))
    bad = len(c["different"]) + len(c["missing"])
    line = f"reference check: {total} numbers compared with the reference run: {len(c['identical'])} identical"
    if c["close"]:
        line += f", {len(c['close'])} close (within 0.02 for numbers between −1 and 1, within 5% otherwise)"
    if bad:
        line += f", {len(c['different'])} different, {len(c['missing'])} missing or not finite"
    print(line)

    def size(key):                                            # relative size of a difference, for sorting
        expected, value = c["reference"].get(key), c["new"].get(key)
        if expected is None or value is None:
            return np.inf
        return abs(value - expected) / max(abs(expected), 0.4)

    shown = sorted(c["different"], key=size, reverse=True)[:8]
    for key in shown:
        expected, value = c["reference"].get(key), c["new"].get(key)
        print(f"   different: {key}: reference {'infinite' if expected is None else format(expected, '.6g')}, "
              f"here {'not finite' if value is None else format(value, '.6g')}")
    for key in c["missing"][:4]:
        print(f"   missing: {key}")
    rest = bad - len(shown) - min(len(c["missing"]), 4)
    if rest > 0:
        print(f"   ... and {rest} more")
    note = _what_differs()
    if (bad or c["close"]) and note:
        print(note)
    return not bad


def check_all(out_dir: str = "outputs") -> dict:
    """Compare the results of all six notebooks with the reference run; one line per notebook.

    Returns the totals {'identical', 'close', 'different', 'missing', 'not_run', 'data_status'}."""
    totals = {"identical": 0, "close": 0, "different": 0, "missing": 0, "not_run": [], "data_status": {}}
    for name in NOTEBOOKS:
        path = os.path.join(out_dir, f"{name}_results.json")
        if not os.path.exists(path):
            print(f"{name}: no results found (the notebook was not run)")
            totals["not_run"].append(name)
            totals["missing"] += len(REFERENCE_RESULTS.get(name, {}))
            continue
        with open(path, encoding="utf-8") as fh:
            payload = json.load(fh)
        c = compare(name, payload["results"])
        for k in ("identical", "close", "different", "missing"):
            totals[k] += len(c[k])
        totals["data_status"].update(payload.get("data_status", {}))
        print(f"{name}: {len(c['identical'])} identical, {len(c['close'])} close, "
              f"{len(c['different'])} different, {len(c['missing'])} missing")
    n = totals["identical"] + totals["close"] + totals["different"] + totals["missing"]
    print(f"all notebooks: {totals['identical']} of {n} numbers identical to the reference run, {totals['close']} close, "
          f"{totals['different']} different, {totals['missing']} missing")
    note = _what_differs()
    if totals["identical"] < n and note:
        print(note)
    return totals
