# NOTEBOOK: 01_models_and_error_bars
# %% [markdown]
r"""
# Notebook 1 · Two models, and why an MAE is not an error bar

**Conformal Prediction for Physical Chemists: hands-on notebooks (1 of 6)**

This notebook sets the stage. It introduces the data set and the two machine-learning models that are used
in all later notebooks, and it shows what is missing when a model is described only by its mean absolute error (MAE).

**You will**

1. look at the RDB7 data set of reaction barrier heights and at how a reaction is turned into numbers;
2. meet the two point predictors of the tutorial, a *structure-only* model and a *Δ-learning* model (Ramakrishnan
   et al., 2015);
3. see that a mean absolute error says little about how wrong a single prediction can be;
4. see that the usual Gaussian "±σ" error bar does not deliver the *coverage* it promises (coverage: the fraction of
   reactions whose reference value lies inside the error bar).

Runs on a CPU in under a minute. No conformal prediction yet: that starts in Notebook 2.
"""

# %% [setup]

# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

P.set_style()
rdb7 = P.load_problem()      # downloads RDB7 once, builds features, fits both models
y = rdb7.y                   # CCSD(T)-F12a barrier heights in kcal/mol: the quantity we want to predict

# %% [markdown]
r"""
## 1. The data: RDB7

RDB7 (Spiekermann, Pattanaik, and Green, *Sci. Data* **2022**, *9*, 417) contains 11,926 gas-phase reactions of
molecules built from H, C, N, and O with at most seven heavy atoms. Every reaction has one reactant and one to three
products. The reactions were found by automated reaction discovery (Grambow, Pattanaik, and Green, 2020), not collected
from textbooks, which is why unusual, high-energy products are common. For each reaction the data set gives
*atom-mapped* SMILES strings (every atom carries a number that identifies it on both sides of the reaction) and the
barrier height at several levels of theory, of which we use two:

| column here | level of theory | role in this tutorial |
|---|---|---|
| `barrier_cc` | CCSD(T)-F12a/cc-pVDZ-F12//ωB97X-D3/def2-TZVP | the reference value we want to predict |
| `barrier_dft` | ωB97X-D3/def2-TZVP | a cheaper estimate, used as an input by the Δ-learning model |

A barrier height is the energy of the transition state minus that of the reactant, including zero-point energy, in
kcal/mol. The lines printed above confirm that the two tables are the ones deposited on Zenodo
(DOI 10.5281/zenodo.6618262).

**Reference values are not the truth.** Throughout these notebooks the "reference barrier" $y$ is the CCSD(T)-F12a
number. It is a high-level estimate, but it has errors of its own: a double-zeta F12 basis set, a single-reference
treatment of transition states that can have multireference character, and geometries and harmonic zero-point energies
taken from density functional theory (DFT). Every statement about errors, coverage, and intervals that follows is a
statement about this reference level.
"""

# %%
table = rdb7.df[["idx", "barrier_cc", "barrier_dft", "dH_cc", "rmg_family"]].head(5).copy()
table.insert(1, "reaction", [f"{P.readable(r)} >> {P.readable(p)}" for r, p in zip(rdb7.df.rsmi[:5], rdb7.df.psmi[:5])])
table["rmg_family"] = table["rmg_family"].fillna("–")
display(table.round(2))
print(f"barrier heights: mean {y.mean():.1f}, standard deviation {y.std():.1f}, range {y.min():.1f} to {y.max():.1f} kcal/mol")
print(f"reactions per reactant molecule: {rdb7.n / len(set(rdb7.reactant)):.1f} on average")

# %% [markdown]
r"""
(`dH_cc` is the reaction enthalpy at 298 K at the reference level, shown for orientation; `rmg_family` is explained
next. In the SMILES strings, `>>` separates the reactant from the products, a dot separates two product molecules,
lower-case letters are aromatic atoms, square brackets hold atoms with a formal charge, an unusual valence, or a stereo
label (`@`, `@@`), and `/` and `\` give the geometry at a double bond; `[C]`, `[CH]`, and `[CH2]` are carbene carbons
(in a few cases radical centers).)

### Reaction classes

Later notebooks need to know what *kind* of reaction each entry is. The authors of RDB7 assigned a reaction family of
the Reaction Mechanism Generator (RMG; Gao et al., 2016) to about 12% of the reactions; the rest match no template. We
therefore use a coarser classification that can be assigned to every reaction automatically, from the way the products
are written and from the bonds that are broken. The rules are applied in this order, and the first one that applies
decides (see `primer.reaction_class`):

| class | rule (structures as RDB7 writes them) | what it contains |
|---|---|---|
| carbene product | a product has a divalent carbon atom | carbenes left behind by 1,2-shifts, ring openings, and extrusions (in about 2% of the cases a nitrene or two radical centers instead) |
| zwitterionic product | a product is written with formal charges | heteroatom-stabilized carbenes drawn as ylides (three cases in ten); ylides, diazo compounds, and other structures with the charges on neighboring atoms (one third); zwitterions with the charges farther apart (one third) |
| fragmentation | two or three product molecules | loss of H₂, H₂O, CH₂O, C₂H₄, CH₄, or CO in seven cases out of ten; retro-ene reactions and retro-cycloadditions |
| H-transfer isomerization | one product; at least one X–H bond is broken | tautomerizations and hydrogen shifts, in six cases out of ten combined with a change of the heavy-atom skeleton |
| skeletal rearrangement | one product; no X–H bond is broken | ring contractions and expansions, shifts of heavy-atom groups, ring openings, and cyclizations |

These are **bookkeeping classes**, read off the Lewis structures in the data set; they are not mechanistic classes.
Three details matter. Carbon monoxide and isocyanides are conventionally written with formal charges
(`[C-]#[O+]`, `[C-]#[N+]R`); these are not counted as charge separation, so the loss of CO alone is a fragmentation.
Hydrogen isocyanide, which RDB7 writes as the carbene `[C]=N`, counts as a carbene product. And the first two classes
overlap chemically: a heteroatom-stabilized carbene lands in one or the other depending on which resonance structure
the data set happens to use. (The tables below list the classes in a fixed order, which is not the order of the rules.)
"""

# %%
classes = pd.Series(rdb7.rxn_class)
overview = pd.DataFrame({
    "reactions": classes.value_counts().reindex(P.CLASS_ORDER),
    "mean barrier (kcal/mol)": pd.Series(y).groupby(classes.values).mean().reindex(P.CLASS_ORDER).round(1),
})
EXAMPLES = {"H-transfer isomerization": 75, "skeletal rearrangement": 8, "fragmentation": 21,      # row numbers of one
            "carbene product": 7, "zwitterionic product": 0}                                        # clear example each
examples = []
for c in P.CLASS_ORDER:
    row = EXAMPLES[c]
    assert rdb7.rxn_class[row] == c
    examples.append(f"{P.readable(rdb7.df.rsmi[row])} >> {P.readable(rdb7.df.psmi[row])}")
overview["example"] = examples
display(overview)
print(f"reactions with an RMG family label: {rdb7.df.rmg_family.notna().mean():.1%}")

# %% [markdown]
r"""
## 2. From a reaction to numbers

The atom maps tell us which bonds are broken, formed, or change their order. Counting these events by bond type gives
a small set of interpretable descriptors. *Morgan fingerprints* (counts of the circular atom environments that occur in
a molecule, up to two bonds around each atom; Rogers and Hahn, 2010) of the reactant, and the *difference* between the
product and reactant fingerprints, add information about the molecular environment. All of this is computed with RDKit.
Here is the keto–enol tautomerization of butanal and its nonzero descriptors:
"""

# %%
i = EXAMPLES["H-transfer isomerization"]
descriptors, fp_reactant, fp_difference = P.describe_reaction(rdb7.df.rsmi[i], rdb7.df.psmi[i])
print(P.readable(rdb7.df.rsmi[i]), ">>", P.readable(rdb7.df.psmi[i]), f"   (barrier {y[i]:.1f} kcal/mol)\n")
print({name: value for name, value in descriptors.items() if value != 0})
print(f"\nreactant fingerprint: {np.count_nonzero(fp_reactant)} nonzero entries; "
      f"difference fingerprint: {np.count_nonzero(fp_difference)} nonzero entries")

# %% [markdown]
r"""
Read the names as follows: `broken_H-C` is the number of C–H bonds broken, `formed_H-O` the number of O–H bonds formed,
`order_down_C-O` the number of C–O bonds whose order decreases (here C=O becomes C–O), `order_up_C-C` the number of C–C
bonds whose order increases, `n_reacting_atoms` the number of atoms that take part in any of these changes.

## 3. Two point predictors

Both models are gradient-boosted regression trees (Friedman, 2001) from scikit-learn (Pedregosa et al., 2011): 300
trees, learning rate 0.1, no early stopping, all other settings at their defaults. Of the fingerprint entries, the
structure-only model keeps those that are nonzero in at least 5% of the training reactions. The models differ in what
they are given.

| | **Structure-only model** | **Δ-learning model** |
|---|---|---|
| input | descriptors and fingerprints of the reaction | descriptors **plus** the ωB97X-D3 barrier and reaction enthalpy |
| output | the CCSD(T)-F12a barrier | the *correction* from the ωB97X-D3 to the CCSD(T)-F12a barrier |
| what a prediction requires | only the reaction itself (reactant, products, and atom map) | a DFT transition-state search, with optimizations and frequency calculations for reactant, transition state, and products |
| typical use | screening very many candidate reactions | refining barriers that were already computed with DFT |
| name in the code | `rdb7.models["structure"]` | `rdb7.models["delta"]` |

Reference and ωB97X-D3 values share geometries and harmonic frequencies, so the Δ-learning model learns a difference of
electronic energies at fixed geometry; what it saves are the coupled-cluster single-point calculations.

The models are deliberately simple and fast. They are not the state of the art: a graph neural network that was
pretrained on lower-level DFT barriers and then trained on these data (forward and reverse reactions, about 24,000 in
all) reaches a test MAE of 2.6 kcal/mol from the atom-mapped structures of reactant and product alone (Spiekermann,
Pattanaik, and Green, *J. Phys. Chem. A* 2022; their test set is split by the scaffold of the reactant, so the number
is not directly comparable with ours). What we need here is one weak and one strong predictor, because the subject of
the notebooks is what can be said about the errors of a *given* model.

The data are split once into a training half and a held-out half. The split is made **by reactant**: all reactions of a
given reactant molecule go to the same side, so the models are judged on reactant molecules they have never seen
(Notebook 5 comes back to why this matters). The split is by reactant only: in about 70 held-out reactions (about 1% of
the held-out half) the product is a reactant of the training half, and about 20 of these are the exact reverse of a
training reaction (stereoisomers count as different molecules). The models are fitted to the training half only.
**Everything that follows is evaluated on the held-out half.** The function that builds either model is short:
"""

# %%
P.show(P.fit_model)

# %%
def error_summary(residuals):
    a = np.abs(residuals)
    return {"MAE": a.mean(), "RMSE": np.sqrt((a ** 2).mean()), "median": np.median(a),
            "90th pct.": np.quantile(a, 0.90), "95th pct.": np.quantile(a, 0.95),
            "99th pct.": np.quantile(a, 0.99), "largest": a.max()}

residuals = {name: (y - mod.mu)[mod.hold] for name, mod in rdb7.models.items()}
errors = pd.DataFrame({rdb7.models[name].label: error_summary(res) for name, res in residuals.items()}).T
print(f"training half: {len(rdb7.train)} reactions; held-out half: {len(rdb7.hold)} reactions; reactants in common: "
      f"{len(set(rdb7.reactant[rdb7.train]) & set(rdb7.reactant[rdb7.hold]))}")
print("Absolute errors on the held-out reactions (kcal/mol)")
display(errors.round(2))

# %% [markdown]
r"""
Read the table from left to right. The MAE of the Δ-learning model is about 1.6 kcal/mol, not far from the 1 kcal/mol
that is conventionally called chemical accuracy (here relative to the reference level). But one prediction in ten is
off by more than about 3.8 kcal/mol, one in a hundred by more than about 11 kcal/mol, and the worst prediction by more
than 30 kcal/mol. **The MAE describes the typical reaction; a user needs a statement about *this* reaction.**

What such an error means for kinetics follows from $k \propto \exp(-E/RT)$, with $E$ the barrier height: an error
$\Delta E$ in the barrier changes the rate constant by the factor $\exp(\Delta E/RT)$.
"""

# %%
rows = []
for name, res in residuals.items():
    for label, value in (("MAE", np.abs(res).mean()), ("90th percentile", np.quantile(np.abs(res), 0.90))):
        rows.append({"model": rdb7.models[name].label, "error measure": label, "kcal/mol": round(value, 2),
                     "factor in k at 298 K": f"{P.rate_factor(value, 298.15):.1e}",
                     "factor in k at 1000 K": f"{P.rate_factor(value, 1000.0):.2g}"})
display(pd.DataFrame(rows))

# %% [markdown]
r"""
## 4. The Gaussian error bar, and why it fails

The textbook way to attach an uncertainty is to assume that errors are normally distributed with a standard deviation
$\sigma$ equal to the root-mean-square error (RMSE) and to report $\hat\mu(x) \pm z\sigma$, where $\hat\mu(x)$ is the
prediction for reaction $x$: $z = 1$ for 68%, $z = 1.645$ for 90%, $z = 2.576$ for 99%. We can test this promise
directly, because we know the reference barriers of the held-out reactions. (We even use the held-out RMSE, which is
the most favorable choice for the Gaussian recipe.)
"""

# %%
LEVELS = np.array([0.50, 0.68, 0.80, 0.90, 0.95, 0.99])

def gaussian_coverage(res, levels=LEVELS):
    """Fraction of residuals inside +/- z * RMSE, for each nominal level."""
    rmse = np.sqrt(np.mean(res ** 2))
    return np.array([np.mean(np.abs(res) <= stats.norm.ppf(0.5 + level / 2) * rmse) for level in levels])

gauss = pd.DataFrame({rdb7.models[name].label: gaussian_coverage(res) for name, res in residuals.items()},
                     index=[f"{level:.0%}" for level in LEVELS])
gauss.index.name = "promised"
print("Observed coverage of Gaussian intervals")
display(gauss.round(3))
print("excess kurtosis of the residuals (0 for a normal distribution):",
      {rdb7.models[name].label: round(float(stats.kurtosis(res)), 1) for name, res in residuals.items()})

# %% [markdown]
r"""
The promise is broken in both directions. Low and medium levels are over-covered (the "68%" interval of the Δ-learning
model contains more than 80% of the reactions, so it is wider than it needs to be), while the 99% interval misses two
to three times as often as promised. Panel c of the figure below shows this as the ratio of the observed to the promised
miss rate, on a logarithmic axis. An honest error bar would sit at 1 for every level; values above 1 mean more misses
than promised, values below 1 mean that the interval is wider than it needs to be. This is the signature of a
**heavy-tailed error distribution**: most reactions are predicted well, a minority very badly. No choice of a single
$\sigma$ can repair that. (Pernot, 2022, reviews how prediction uncertainties in computational chemistry can be
validated.)

The figure summarizes the notebook.
"""

# %%
fig, axes = P.panels(3)

for ax, letter, name in zip(axes[:2], "ab", ("structure", "delta")):
    mod = rdb7.models[name]
    ax.scatter(y[mod.hold], mod.mu[mod.hold], s=3, alpha=0.25, color=P.MODEL_COLORS[name], linewidths=0, rasterized=True)
    ax.plot([0, 200], [0, 200], color=P.GRAY, lw=0.8)
    ax.set(xlim=(0, 200), ylim=(0, 200), xticks=[0, 50, 100, 150, 200], yticks=[0, 50, 100, 150, 200],
           xlabel="reference barrier (kcal/mol)", ylabel="predicted barrier (kcal/mol)")
    ax.text(0.97, 0.05, f"MAE {errors.loc[mod.label, 'MAE']:.1f}\nRMSE {errors.loc[mod.label, 'RMSE']:.1f}\nkcal/mol",
            transform=ax.transAxes, ha="right", va="bottom", fontsize=7)
    P.panel_label(ax, letter, mod.label)

# panel c: how often does the Gaussian interval miss, relative to what it promises?
ax = axes[2]
positions = np.arange(len(LEVELS))
for name in ("structure", "delta"):
    mod = rdb7.models[name]
    ax.plot(positions, (1 - gauss[mod.label].values) / (1 - LEVELS), marker=P.MODEL_MARKERS[name],
            color=P.MODEL_COLORS[name], label=mod.label)
ax.axhline(1, color=P.GRAY, lw=0.8, ls="--")
ax.set_yscale("log")
ax.set_yticks([0.5, 1, 2, 3], ["0.5", "1", "2", "3"])
ax.minorticks_off()
ax.set_ylim(0.4, 4.2)
ax.set_xticks(positions, [f"{100 * level:.0f}" for level in LEVELS])
ax.set(xlabel="promised coverage (%)", ylabel="observed / promised miss rate")
ax.legend(loc="upper left", borderaxespad=0.2)
ax.text(-0.1, 1.05, "1 = as promised", fontsize=6.5, va="bottom")
P.panel_label(ax, "c", "Gaussian error bars")
for ax in axes:
    ax.set_box_aspect(1)
P.savefig(fig, "nb1_models_and_gaussian_error_bars")
plt.show()

# %% [markdown]
r"""
Panels **a** and **b**: predicted against reference barriers for the held-out reactions. Panel **c**: miss rate of the
Gaussian interval $\hat\mu \pm z\sigma$ relative to the promised one (logarithmic vertical axis; the six levels are
equally spaced on the horizontal axis).

## 5. Errors depend on the chemistry

A last observation prepares Notebooks 3 and 4. The size of the errors is not the same for all reaction classes:
"""

# %%
by_class = {}
for name, mod in rdb7.models.items():
    a = pd.Series(np.abs(y - mod.mu)[mod.hold])
    g = a.groupby(rdb7.rxn_class[mod.hold])
    by_class[(mod.label, "MAE")] = g.mean()
    by_class[(mod.label, "90th pct.")] = g.quantile(0.90)
by_class = pd.DataFrame(by_class).reindex(P.CLASS_ORDER)
by_class.insert(0, "held-out reactions", pd.Series(rdb7.rxn_class[rdb7.hold]).value_counts().reindex(P.CLASS_ORDER))
display(by_class.round(2))

# %% [markdown]
r"""
For the Δ-learning model the 90th percentile of the absolute error differs by a factor of about three between the
class with the smallest errors (carbene products) and the one with the largest (skeletal rearrangements). A single
error bar for all reactions will be too wide for the former and too narrow for the latter.

## Notation used in all six notebooks

| symbol | meaning |
|---|---|
| $x$, $y$ | a reaction (its descriptors) and its reference barrier height in kcal/mol |
| $\hat\mu(x)$ | the prediction of a model for reaction $x$ (`mu` in the code) |
| $C(x)$ | the prediction interval for reaction $x$ |
| coverage | the fraction of cases whose reference value lies inside the interval |
| $\alpha$ | the miscoverage level: the fraction of misses we accept; $1-\alpha$ is the nominal coverage (default 90%) |
| $n$ | number of calibration reactions (Notebook 2) |
| $N$, $K$ | number of labeled reactions when no calibration set is set aside, and number of folds (Notebook 2, CV+) |
| $s_i$ | score of calibration reaction $i$: a number that says how wrong the model was (in Notebook 6: how promising a candidate looks) |
| $k$, $\hat q$ | $k = \lceil (n+1)(1-\alpha)\rceil$, and the conformal quantile $\hat q$: the $k$-th smallest calibration score (in Notebooks 1, 2, and 6, $k$ also stands for a rate constant) |
| half-width | half the length of an interval, the number after the "±" |
| $\hat\sigma(x)$ | a difficulty estimate for reaction $x$ (Notebook 3); not to be confused with the Gaussian $\sigma$ above |
| $w(x)$ | weight of a calibration reaction under covariate shift (Notebook 5) |
| $m$ | number of new cases: test reactions, candidates, or new reference calculations |
| $c$, $p_j$ | the threshold that defines a hit, and the conformal p-value of candidate $j$ (Notebook 6) |
| $q$, $r$ | the target false discovery rate of Notebook 6 (the usual symbol in that field; unrelated to $\hat q$), and the rank of a candidate in the Benjamini–Hochberg step |

## Take-aways

* A point prediction and an MAE do not tell the user how far to trust one particular number.
* The Gaussian "±σ" recipe assumes a shape for the error distribution. Here the shape is wrong (heavy tails): the
  intervals are too wide at low levels and too narrow in the tails, and right only where the two errors happen to
  cancel.
* What we want instead is an interval with a *guaranteed* coverage that does not depend on the shape of the error
  distribution or on the model. Notebook 2 constructs exactly that, in about fifteen lines of NumPy.

## Try it yourself

1. Replace the held-out RMSE in `gaussian_coverage` by the RMSE of the *training* residuals, as one would do without a
   held-out set. What happens to the observed coverages?
2. Restrict the comparison to one reaction class. Is the Gaussian recipe better within a class?

## References

* Spiekermann, K. A.; Pattanaik, L.; Green, W. H. High Accuracy Barrier Heights, Enthalpies, and Rate Coefficients for
  Chemical Reactions. *Sci. Data* **2022**, *9*, 417. https://doi.org/10.1038/s41597-022-01529-6
  (data: https://doi.org/10.5281/zenodo.6618262, CC BY 4.0)
* Grambow, C. A.; Pattanaik, L.; Green, W. H. Reactants, Products, and Transition States of Elementary Chemical
  Reactions Based on Quantum Chemistry. *Sci. Data* **2020**, *7*, 137. https://doi.org/10.1038/s41597-020-0460-4
* Spiekermann, K. A.; Pattanaik, L.; Green, W. H. Fast Predictions of Reaction Barrier Heights: Toward Coupled-Cluster
  Accuracy. *J. Phys. Chem. A* **2022**, *126*, 3976–3986. https://doi.org/10.1021/acs.jpca.2c02614
* Gao, C. W.; Allen, J. W.; Green, W. H.; West, R. H. Reaction Mechanism Generator: Automatic Construction of Chemical
  Kinetic Mechanisms. *Comput. Phys. Commun.* **2016**, *203*, 212–225. https://doi.org/10.1016/j.cpc.2016.02.013
* Ramakrishnan, R.; Dral, P. O.; Rupp, M.; von Lilienfeld, O. A. Big Data Meets Quantum Chemistry Approximations: The
  Δ-Machine Learning Approach. *J. Chem. Theory Comput.* **2015**, *11*, 2087–2096. https://doi.org/10.1021/acs.jctc.5b00099
* Rogers, D.; Hahn, M. Extended-Connectivity Fingerprints. *J. Chem. Inf. Model.* **2010**, *50*, 742–754.
  https://doi.org/10.1021/ci100050t
* Friedman, J. H. Greedy Function Approximation: A Gradient Boosting Machine. *Ann. Stat.* **2001**, *29*, 1189–1232.
  https://doi.org/10.1214/aos/1013203451
* Pedregosa, F.; et al. Scikit-learn: Machine Learning in Python. *J. Mach. Learn. Res.* **2011**, *12* (85), 2825–2830.
  https://jmlr.org/papers/v12/pedregosa11a.html
* RDKit: Open-Source Cheminformatics. https://www.rdkit.org (accessed 2026-10-11). The version used in a run is stored
  with its results.
* Pernot, P. The Long Road to Calibrated Prediction Uncertainty in Computational Chemistry. *J. Chem. Phys.* **2022**,
  *156*, 114109. https://doi.org/10.1063/5.0084302
"""

# %%
results = {"n_reactions": rdb7.n, "n_train": len(rdb7.train), "n_hold": len(rdb7.hold),
           "n_reactants": len(set(rdb7.reactant)), "shared_reactants": len(set(rdb7.reactant[rdb7.train]) & set(rdb7.reactant[rdb7.hold]))}
for name, mod in rdb7.models.items():
    results[name] = {
        "n_features": mod.X.shape[1],
        "errors": {k.replace(" pct.", "").replace("th", ""): v for k, v in errors.loc[mod.label].items()},
        "gaussian_coverage": {f"{level:.2f}": c for level, c in zip(LEVELS, gauss[mod.label].values)},
        "excess_kurtosis": float(stats.kurtosis(residuals[name])),
        "class_q90": {c: by_class.loc[c, (mod.label, "90th pct.")] for c in P.CLASS_ORDER},
        "class_mae": {c: by_class.loc[c, (mod.label, "MAE")] for c in P.CLASS_ORDER},
    }
results["class_counts"] = {c: int((rdb7.rxn_class == c).sum()) for c in P.CLASS_ORDER}
results["barrier_mean"], results["family_share"] = float(y.mean()), float(rdb7.df.rmg_family.notna().mean())
P.save_results("nb1", results, rdb7)
P.check("nb1", results);
