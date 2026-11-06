# ConformalPrimer

**Conformal prediction for physical chemists: six hands-on notebooks.**

A machine-learning model that predicts a reaction barrier returns one number. These notebooks show how to
turn that number into a statement with a guarantee: an interval that contains the reference value with a
chosen probability, or a shortlist of reactions with a controlled expected fraction of false entries. The method
is conformal prediction. It works with any model and needs no assumption about the distribution of the
errors. It needs one condition: the cases used for calibration and the cases to be predicted must come
from the same population (they must be *exchangeable*). Notebook 5 shows what happens when this condition
fails, how to notice it, and what can be done about it.

Everything is derived step by step in NumPy on one public data set, the
[RDB7](https://doi.org/10.1038/s41597-022-01529-6) collection of 11,926 gas-phase reaction barriers at
the CCSD(T)-F12a level. All notebooks run on two CPU cores in a few minutes; no GPU and no data upload
are needed.

## The notebooks

| | notebook | what you learn | run time |
|---|---|---|---|
| 1 | `01_models_and_error_bars.ipynb` | the data, two models, and why an MAE or a Gaussian "±σ" is not an error bar | < 1 min |
| 2 | `02_split_conformal.ipynb` | split conformal prediction from scratch; why it works; what the guarantee does not say; the Beta law of the coverage; how large the calibration set must be; CV+ and jackknife+ for small data | < 1 min |
| 3 | `03_adaptive_intervals.ipynb` | intervals whose width follows the difficulty of the prediction: normalized scores and conformalized quantile regression; diagnostics for conditional coverage | < 1 min |
| 4 | `04_mondrian.ipynb` | Mondrian (group-conditional) calibration: a guarantee for every reaction class, and what it costs | < 1 min |
| 5 | `05_exchangeability.ipynb` | when the guarantee breaks: shifted class mix, larger molecules, unseen reaction classes, reactants already seen in fitting, clustered data; weighted conformal prediction, recalibration, warning signs | 4–5 min |
| 6 | `06_selection.ipynb` | from intervals to decisions: conformal p-values and the Benjamini–Hochberg step give shortlists with a guaranteed false discovery rate | < 1 min |

The run times were measured on two CPU cores; the free Colab runtime can take up to twice as long.
`ConformalPrimer_all_in_one.ipynb` runs the six parts in sequence (in October 2026: 8 minutes on Kaggle, 12 minutes
on Colab) and packs every
figure and number into `ConformalPrimer_outputs.zip`.

Each notebook ends with take-aways, exercises ("Try it yourself"), and the key references. The repository
stores no figures and no run outputs: each notebook draws its figures and computes its numbers when it runs.

## Quick start

**Google Colab.** Open a notebook from GitHub in Colab
(`https://colab.research.google.com/github/alikhairbek/ConformalPrimer/blob/main/01_models_and_error_bars.ipynb`;
replace the file name for the other notebooks) and choose *Runtime → Run all*.

**Kaggle.** Create a notebook, choose *File → Import notebook* and give the GitHub address of the
notebook (or upload the `.ipynb` file). Switch **Internet on** in the notebook settings, then *Run all*.
The all-in-one notebook writes `ConformalPrimer_outputs.zip` to the output folder.

**Local.**

```bash
git clone https://github.com/alikhairbek/ConformalPrimer.git
cd ConformalPrimer
pip install -r requirements.txt    # the libraries and JupyterLab
jupyter lab                        # open a notebook and run all cells
```

or, without a browser:

```bash
pip install -r requirements-dev.txt
python run_all.py                  # executes the six notebooks, writes executed/ and outputs/
```

Every notebook is self-contained. Its first (collapsed) cell installs RDKit if necessary and writes the
helper module `primer.py` next to the notebook, so a single `.ipynb` file is all that has to be opened
on Colab or Kaggle. The two data tables (3.8 MB each) are downloaded from Zenodo on first use and
checked against stored fingerprints.

## Data

The notebooks use two tables of RDB7, version 1.0.1:

> Spiekermann, K. A.; Pattanaik, L.; Green, W. H. High Accuracy Barrier Heights, Enthalpies, and Rate
> Coefficients for Chemical Reactions. *Sci. Data* **2022**, *9*, 417.
> Data: https://doi.org/10.5281/zenodo.6618262 (CC BY 4.0).

| file | content |
|---|---|
| `ccsdtf12_dz.csv` | barrier heights and reaction enthalpies at CCSD(T)-F12a/cc-pVDZ-F12//ωB97X-D3/def2-TZVP (the reference values to be predicted) |
| `wb97xd3.csv` | the same quantities at ωB97X-D3/def2-TZVP (the cheaper level used by the Δ-learning model) |

The data are not redistributed here. `primer.load_rdb7()` looks for the files in `data/`, then among the
Kaggle inputs, and otherwise downloads them. It then compares them with the Zenodo files and prints one
of three verdicts: *byte-identical to the Zenodo file*, *same content as the Zenodo file* (same reactions
and energies, different line endings or number formatting), or *DIFFERENT from the Zenodo file*, in which
case it stops. Energies are rounded to 10⁻⁵ kcal/mol on loading, so every copy with the same content
gives the same numbers. Without internet access, download the two files by hand and put them in a folder
`data/` next to the notebook.

## The two models

Both are gradient-boosted regression trees from scikit-learn with fixed settings, fitted to the reactions
of half of the reactants (5991 reactions) and judged on the other half (5935 reactions, none of whose
reactant molecules occurs as a reactant in the fitting half).

| | structure-only model | Δ-learning model |
|---|---|---|
| input | bond-change descriptors and Morgan-fingerprint counts of the reaction | bond-change descriptors plus the ωB97X-D3 barrier and reaction enthalpy |
| predicts | the CCSD(T)-F12a barrier | the correction from the ωB97X-D3 barrier to the CCSD(T)-F12a barrier |
| held-out MAE | 8.6 kcal/mol | 1.6 kcal/mol |
| 90% conformal interval | ±20 kcal/mol | ±3.8 kcal/mol |

The models are deliberately simple. The subject of the notebooks is what conformal prediction says about
a given model, good or bad, not how to build the best one.

## The toolbox in your own project

`conformal_toolbox.py` contains the functions that the notebooks derive: about 100 lines of code that
need only NumPy (and SciPy for one function). Copy the file into your project:

```python
import numpy as np
from conformal_toolbox import split_conformal, conformal_pvalues, benjamini_hochberg

# y_cal, mu_cal: reference values and predictions for a calibration set that the model has not seen
# mu_new: predictions for new cases from the same population
lower, upper, q_hat = split_conformal(y_cal, mu_cal, mu_new, alpha=0.10)     # 90% intervals: mu_new ± q_hat

# shortlist of the new cases whose value is below c, with a false discovery rate of at most 10%
c = 50.0                                       # the threshold that defines a hit
p = conformal_pvalues(mu_cal - c, y_cal >= c, mu_new - c)
shortlist = np.flatnonzero(benjamini_hochberg(p, q=0.10))
```

| function | purpose | notebook |
|---|---|---|
| `conformal_quantile`, `split_conformal` | intervals of constant width with guaranteed coverage | 2 |
| `coverage_law` | Beta distribution of the coverage obtained from one calibration set | 2 |
| `cv_plus` | CV+ and jackknife+ intervals when no calibration set can be set aside | 2 |
| `normalized_conformal`, `cqr`, `interval_score` | adaptive intervals and a score to compare them | 3 |
| `mondrian_quantiles` | one conformal quantile per category | 4 |
| `weighted_conformal_quantile`, `effective_sample_size` | conformal prediction under covariate shift | 5 |
| `conformal_pvalues`, `benjamini_hochberg` | selection with a guaranteed false discovery rate | 6 |

The guarantees hold under the conditions stated in the notebooks, above all that the calibration cases
were not used to fit or tune the model and are exchangeable with the new cases.

## Reproducibility

Every random choice is seeded. Each notebook stores its numbers in `outputs/nbN_results.json` and compares
them with the reference run (`reference_results.json`, 905 numbers, also embedded in the notebooks). The
reference run uses the library versions of the Colab runtime of October 2026. The check reports how many
numbers are identical, how many are close (within 0.02 for numbers between −1 and 1, such as coverages, and
within 5% for the others; counts that do not depend on the fitted models must be identical), and how many differ by
more or are missing.

| environment | Python | NumPy | pandas | SciPy | scikit-learn | Matplotlib | RDKit | identical | close | different |
|---|---|---|---|---|---|---|---|---|---|---|
| reference run (Colab versions) | 3.12 | 2.1.3 | 2.2.3 | 1.16.3 | 1.6.1 | 3.10.0 | 2026.09.1 | 905 | 0 | 0 |
| Kaggle-like, oldest versions tested | 3.11 | 1.26.4 | 2.2.3 | 1.11.4 | 1.2.2 | 3.7.5 | 2023.09.6 | 905 | 0 | 0 |
| newest versions tested | 3.13 | 2.5.3 | 3.0.5 | 1.18.1 | 1.9.1 | 3.11.2 | 2026.09.1 | 413 | 472 | 20 |
| Google Colab, October 2026 | 3.13 | 2.1.3 | 2.2.3 | 1.16.3 | 1.6.1 | 3.10.0 | 2026.09.1 | 905 | 0 | 0 |
| Kaggle, October 2026 | 3.13 | 2.1.3 | 2.3.3 | 1.16.3 | 1.6.1 | 3.10.0 | 2026.09.1 | 905 | 0 | 0 |

Apart from the reference run, each row is a clean run of `ConformalPrimer_all_in_one.ipynb` (the last two on the
platforms themselves, where the data were downloaded from Zenodo and all seven figures came out byte-identical to the
reference); in a fresh environment with the Colab versions, in which the first cell had to install RDKit, the run also
reproduced all 905 numbers. With scikit-learn 1.2.2 and 1.6.1 all numbers are identical. scikit-learn 1.9.1 fits the
trees of the Δ-learning model slightly differently (held-out MAE 1.624 instead of 1.621 kcal/mol; the NumPy version
plays no role), which moves the numbers that involve this model; all numbers that involve only the structure-only model
stay identical. The median change of the numbers that move is 0.4%. The 20 numbers that move by more than the tolerance
come from a single calibration set or from small samples, for example the number of false discoveries in one shortlist
(31 instead of 45) and the conformal quantiles of single reaction classes.

The statements made in the text do not depend on the environment: `tests/test_narrative.py` checks the 285
numbers quoted in the notebooks against a run, each with the range in which its sentence stays true, and
passes in all three environments.

## Repository layout

```
01_… to 06_….ipynb                 the six tutorial notebooks (generated; no outputs stored)
ConformalPrimer_all_in_one.ipynb   all six in one run, with a results archive
primer.py                          data loading, features, the two models, toolbox, plotting, result checks
conformal_toolbox.py               the conformal functions alone (generated from primer.py)
nbsrc/nb01.py … nb06.py            notebook sources in "percent" format (edit these, not the notebooks)
build_notebooks.py                 nbsrc/ + primer.py  ->  notebooks and conformal_toolbox.py
run_all.py                         execute the notebooks; --record stores new reference numbers
run_src.py                         run one notebook source as a plain script (development)
reference_results.json             numbers of the reference run
tests/                             tests of the toolbox, of the data pipeline, and of the numbers quoted in the text
executed/, outputs/                created by run_all.py and the notebooks (not stored in the repository)
```

To change a notebook, edit its source in `nbsrc/`, then run `python build_notebooks.py`. After a change
that alters numbers, run `python run_all.py --record`: it executes the notebooks, stores the new reference
numbers, rebuilds the notebooks with these numbers embedded and executes them once more.

## Tests

```bash
python tests/test_toolbox.py       # 26 tests on simulated data: every guarantee is checked against theory
python tests/test_pipeline.py      # 14 tests of data, split, reaction classes, models, and caches (downloads RDB7 on first use)
python tests/test_narrative.py     # the 285 numbers quoted in the text of the notebooks, checked against outputs/
```

All three files also run under `pytest`.

## How to cite

The notebooks accompany a Tutorial article that is in preparation; this section will be updated with the
reference. If you use the data, cite Spiekermann, Pattanaik, and Green (above). See also `CITATION.cff`.

## License

Code: MIT (see `LICENSE`). The RDB7 data are provided by their authors under CC BY 4.0 and are not part
of this repository.
