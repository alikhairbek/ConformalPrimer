# NOTEBOOK: 05_exchangeability
# %% [markdown]
r"""
# Notebook 5 · When the guarantee breaks: exchangeability in practice

**Conformal Prediction for Physical Chemists: hands-on notebooks (5 of 6)**

Every guarantee in Notebooks 2 to 4 rests on one assumption: the calibration reactions and the reactions we want to
predict are *exchangeable*. So far we have ensured this by drawing both at random from the same pool. In real work the
calibration set is whatever has been computed already, and tomorrow's reactions are chosen by a chemist with a
different question. This notebook breaks the assumption in controlled ways, measures the damage, and shows what can be
repaired and what cannot.

**You will**

1. see how the coverage of "90%" intervals changes when the deployment set differs from the calibration set:
   in its mix of reaction classes, in molecular size, and by a reaction class the model has never seen;
2. repair the first kind of shift with *weighted* conformal prediction and with Mondrian calibration;
3. see why no reweighting can repair the other two, and recalibrate with a few reference calculations from the new
   domain instead;
4. measure whether hidden dependence in the data (several reactions of the same reactant) matters;
5. learn two warning signs: a classifier that can tell calibration from deployment inputs, and a coverage alarm.

Runs on a CPU in four to five minutes (it refits the models many times).
"""

# %% [setup]

# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict

P.set_style()
rdb7 = P.load_problem()
y = rdb7.y
alpha = 0.10
classes = rdb7.rxn_class
conformal_quantile = P.conformal_quantile

model = rdb7.models["delta"]                       # the Δ-learning model is again the running example
abs_err = {name: np.abs(y - mod.mu) for name, mod in rdb7.models.items()}
scenario_coverage = {}                             # collected for the summary figure: {scenario: {model: coverage}}

# %% [markdown]
r"""
## 1. What exchangeability asks for

Exchangeability is a statement about **how the calibration set relates to the cases you will predict**, not about the
model. In practical terms it asks one question:

> *Could my calibration reactions and my future reactions have been drawn from the same hat?*

Three things can go wrong, and they are worth keeping apart.

* **The inputs change** (covariate shift): tomorrow's reactions are larger, belong to other classes, or come in other
  proportions, while the relation between a reaction and its barrier stays the same.
* **The relation changes**: the reference level of theory, the solvent model, or the experimental protocol is
  different. The same reaction now has a different reference value.
* **The data are not independent draws**: conformers of one molecule, forward and reverse reactions, or many reactions
  of the same reactant sit on both sides of a split.

We start with the mildest case.

## 2. A different mix of reaction classes

Suppose the calibration set is a random sample of RDB7-like chemistry, but the next project is about skeletal
rearrangements: 60% of the reactions to be predicted are of this class, against roughly one in seven in the
calibration set. Nothing about any individual reaction has changed; only the mix has.
"""

# %%
HARD = "skeletal rearrangement"


def deployment_set(pool, rng, n_deploy=600, share=0.60):
    """n_deploy reactions from `pool`, a fraction `share` of them skeletal rearrangements (these come first)."""
    hard, rest = pool[classes[pool] == HARD], pool[classes[pool] != HARD]
    n_hard = int(round(share * n_deploy))
    return np.concatenate([rng.choice(hard, n_hard, replace=False), rng.choice(rest, n_deploy - n_hard, replace=False)])


rng = np.random.default_rng(P.SEED)
shuffled = rng.permutation(model.hold)
cal, pool = shuffled[:1500], shuffled[1500:]
deploy = deployment_set(pool, rng)

err = abs_err["delta"]
q_plain = conformal_quantile(err[cal], alpha)
print(f"share of skeletal rearrangements: calibration {np.mean(classes[cal] == HARD):.0%}, deployment {np.mean(classes[deploy] == HARD):.0%}")
print(f"split conformal, q = {q_plain:.2f} kcal/mol: coverage on the deployment set {np.mean(err[deploy] <= q_plain):.1%}")

# %% [markdown]
r"""
The "90%" interval covers clearly less than 90% of the reactions in the new project, because the class with the
largest errors is now over-represented. Two repairs are available.

### Repair 1: weighted conformal prediction

If we knew how much more (or less) likely each kind of reaction is in the deployment set than in the calibration set,
the **likelihood ratio**

$$ w(x) = \frac{p_\text{deployment}(x)}{p_\text{calibration}(x)} , $$

we could let every calibration score count with weight $w(x_i)$: scores of reactions that are common in the new
project count more. The conformal quantile becomes a *weighted* quantile, and the test point itself contributes its
own weight at $+\infty$ (the weighted counterpart of the "$n+1$" in Notebook 2). Tibshirani, Barber, Candès, and Ramdas
(2019) showed that with the true $w$ this restores the guarantee under covariate shift.
"""

# %%
def weighted_conformal_quantile(scores, w_cal, w_test, alpha=0.10):
    """Weighted conformal quantile, one value per test point."""
    order = np.argsort(scores)
    cum = np.cumsum(w_cal[order])                       # weight at or below each sorted score
    need = (1 - alpha) * (cum[-1] + w_test)             # (1 - alpha) of the total weight, test point included
    idx = np.searchsorted(cum, need * (1 - 1e-9), side="left")      # the factor guards against floating-point round-up
    return np.append(scores[order], np.inf)[idx]        # infinite if the calibration weight does not suffice


def class_ratio_weights(classes_cal, classes_deploy):
    """Weights when only the class proportions differ: share in the deployment sample / share in the calibration sample."""
    share_cal = pd.Series(classes_cal).value_counts(normalize=True)
    share_dep = pd.Series(classes_deploy).value_counts(normalize=True).reindex(share_cal.index).fillna(0.0)
    ratio = (share_dep / share_cal).to_dict()            # 0 for a class that does not occur in the deployment sample
    return np.array([ratio[g] for g in classes_cal]), np.array([ratio[g] for g in classes_deploy]), ratio


# here the shift is entirely in the class proportions, so w(x) is a ratio of class frequencies
w_cal, w_deploy, ratio = class_ratio_weights(classes[cal], classes[deploy])
print("weights by class:", {P.CLASS_SHORT[g]: round(ratio[g], 2) for g in P.CLASS_ORDER})

q_weighted = weighted_conformal_quantile(err[cal], w_cal, w_deploy, alpha)
assert np.array_equal(q_weighted, P.weighted_conformal_quantile(err[cal], w_cal, w_deploy, alpha))
print(f"weighted conformal: coverage {np.mean(err[deploy] <= q_weighted):.1%}, mean half-width {q_weighted.mean():.2f} kcal/mol")

# %% [markdown]
r"""
In real life nobody hands us $w(x)$, and the shift is rarely a clean change of class frequencies. The standard trick is
to **learn $w$ from the inputs alone** (Bickel, Brückner, and Scheffer, 2009): train a classifier to tell calibration
reactions from deployment reactions using their features (no barriers needed), and convert its probability $\pi(x)$ of
"deployment" into

$$ \hat w(x) = \frac{\pi(x)}{1-\pi(x)}\cdot\frac{n_\text{calibration}}{n_\text{deployment}} . $$

With an estimated $\hat w$ the guarantee holds only approximately, to the extent that the estimate is accurate. The
same classifier gives two diagnostics. Its **AUC** (the area under its receiver operating characteristic curve) says
whether the two sets can be told apart at all (0.5: not at all; 1: perfectly). The **effective sample size**
$(\sum w_i)^2/\sum w_i^2$ is a rule of thumb for how many equally weighted calibration reactions the weighted set is
worth.
"""

# %%
def estimate_weights(X_cal, X_deploy, seed=P.SEED, clip=0.01):
    """Likelihood-ratio weights from a classifier 'calibration or deployment?'. Returns (w_cal, w_deploy, AUC)."""
    X = np.vstack([X_cal, X_deploy])
    is_deploy = np.r_[np.zeros(len(X_cal)), np.ones(len(X_deploy))]
    classifier = HistGradientBoostingClassifier(max_iter=100, max_depth=3, random_state=seed)
    folds = StratifiedKFold(5, shuffle=True, random_state=seed)        # shuffled: the rows may come in a systematic order
    prob = cross_val_predict(classifier, X, is_deploy, cv=folds, method="predict_proba")[:, 1]    # out-of-fold probabilities
    auc = roc_auc_score(is_deploy, prob)
    prob = np.clip(prob, clip, 1 - clip)                # keeps the weights finite; see Section 3
    w = prob / (1 - prob) * len(X_cal) / len(X_deploy)
    return w[: len(X_cal)], w[len(X_cal):], auc


w_cal_hat, w_deploy_hat, auc = estimate_weights(model.X[cal], model.X[deploy])
q_learned = weighted_conformal_quantile(err[cal], w_cal_hat, w_deploy_hat, alpha)
print(f"classifier AUC {auc:.2f}; effective calibration sample size {P.effective_sample_size(w_cal_hat):.0f} of {len(cal)}")
print(f"weighted conformal with learned weights: coverage {np.mean(err[deploy] <= q_learned):.1%}")
print(f"(one calibration set and one deployment set of {len(deploy)} reactions: such coverages scatter by several percentage points; "
      "see the standard deviations in the next table)")

# %% [markdown]
r"""
### Repair 2: Mondrian calibration

Notebook 4 already contains the second repair. If every class is calibrated on its own, the class proportions drop
out of the argument altogether: Mondrian intervals stay valid under *any* change of the mix, with no weights to
estimate. The comparison below repeats the whole experiment 200 times with new calibration and deployment sets (the
learned weights, which need a classifier each time, in the first 40 repetitions). The standard deviation says how
much the coverage of one such pair of sets scatters around the mean.
"""

# %%
rows, diagnostics = [], []
rng = np.random.default_rng(31)
for rep in range(200):
    shuffled = rng.permutation(model.hold)
    cal_b, pool_b = shuffled[:1500], shuffled[1500:]
    dep_b = deployment_set(pool_b, rng)
    w_c, w_d, _ = class_ratio_weights(classes[cal_b], classes[dep_b])
    learned = estimate_weights(model.X[cal_b], model.X[dep_b], seed=rep) if rep < 40 else None
    if learned is not None:
        diagnostics.append({"AUC": learned[2], "effective sample size": P.effective_sample_size(learned[0])})
    for name in ("delta", "structure"):
        e = abs_err[name]
        half = {"split conformal (no repair)": np.full(len(dep_b), conformal_quantile(e[cal_b], alpha)),
                "weighted, class-frequency ratios": weighted_conformal_quantile(e[cal_b], w_c, w_d, alpha)}
        if name == "delta" and learned is not None:           # the learned weights use the features of the Δ-learning model
            half["weighted, learned weights"] = weighted_conformal_quantile(e[cal_b], learned[0], learned[1], alpha)
        qm = P.mondrian_quantiles(e[cal_b], classes[cal_b], alpha)
        half["Mondrian by class"] = np.array([qm[g] for g in classes[dep_b]])
        for method, h in half.items():
            rows.append({"model": rdb7.models[name].label, "method": method, "coverage": np.mean(e[dep_b] <= h), "mean half-width": h.mean()})
mix = pd.DataFrame(rows).groupby(["model", "method"], sort=False).agg(**{
    "coverage": ("coverage", "mean"), "s.d. of coverage": ("coverage", "std"), "mean half-width": ("mean half-width", "mean")})
display(mix.round(3))
diagnostics = pd.DataFrame(diagnostics).mean()
print(f"classifier AUC {diagnostics['AUC']:.2f}; effective sample size {diagnostics['effective sample size']:.0f} of 1500")
scenario_coverage["different mix of classes"] = {name: mix.loc[(rdb7.models[name].label, "split conformal (no repair)"), "coverage"] for name in rdb7.models}

# %% [markdown]
r"""
The class-frequency weights and Mondrian calibration bring the coverage back to 90%. So do the learned weights, within
the noise of their 40 repetitions (the standard error of that mean is about 0.3 percentage points), although the
classifier was not given the class label (it can reconstruct the class from the descriptors). For learned weights this
is an observation and not a guarantee: how close the coverage comes depends on how well the classifier estimates $w$.
All repairs pay in the same way: the intervals are wider, because the new project really is harder. Note the effective
sample size: 1500 calibration reactions are worth only a few hundred once they are reweighted, so the coverage obtained
from one calibration set is less certain than the Beta law of Notebook 2 says for $n = 1500$. The table shows this in
the standard deviations: those of the repaired rows are larger than that of the unrepaired row. (All of them also
contain the sampling noise of the 600 deployment reactions, about 1.2 to 1.5 percentage points, which the Beta law
leaves out; the effective sample size is only a rough guide.)

## 3. Outside the calibration domain: larger molecules

Now a shift of a different kind. The models are fitted and calibrated on reactions of molecules with **at most six**
heavy atoms and then used for molecules with **seven** (in a real project: the data set was built on small molecules,
and the interesting chemistry is one size up). The models have to be refitted for this experiment; as always, no
reactant is shared between the fitting set and the calibration set.
"""

# %%
small, large = np.flatnonzero(rdb7.n_heavy <= 6), np.flatnonzero(rdb7.n_heavy == 7)
first, second = P.group_split(rdb7.reactant[small], 2 / 3, seed=P.SEED)
fit_small, cal_small = small[first], small[second]
print(f"fitting set {len(fit_small)}, calibration set {len(cal_small)} (both up to 6 heavy atoms); deployment set {len(large)} (7 heavy atoms)")

size_models, rows = {}, []
for name in rdb7.models:
    size_models[name] = mod = P.fit_model(rdb7, name, fit_small)
    e = np.abs(y - mod.mu)
    q = conformal_quantile(e[cal_small], alpha)
    rows.append({"model": mod.label, "MAE, calibration set": e[cal_small].mean(), "MAE, larger molecules": e[large].mean(),
                 "half-width used": q, "coverage on larger molecules": np.mean(e[large] <= q),
                 "half-width needed for 90%": np.quantile(e[large], 1 - alpha)})
size_table = pd.DataFrame(rows).set_index("model")
display(size_table.round(3))
scenario_coverage["larger molecules"] = {name: size_table.loc[rdb7.models[name].label, "coverage on larger molecules"] for name in rdb7.models}

# %% [markdown]
r"""
The errors on the larger molecules are somewhat bigger, and the coverage falls short of the promise by two to three
percentage points for both models. A modest loss, but a systematic one: it does not go away with a larger calibration
set. Can weighting repair it?
"""

# %%
mod_small = size_models["delta"]
err_small = np.abs(y - mod_small.mu)
subset = np.random.default_rng(P.SEED).choice(large, 2000, replace=False)
w_c, w_d, auc_size = estimate_weights(mod_small.X[cal_small], mod_small.X[subset])
q_w = weighted_conformal_quantile(err_small[cal_small], w_c, w_d, alpha)
print(f"classifier AUC: {auc_size:.3f}")
print(f"weighted conformal intervals that are infinite: {np.mean(np.isinf(q_w)):.0%}")

# %% [markdown]
r"""
No. The classifier separates the two sets perfectly (the number of heavy atoms is one of its features), which means
that **no calibration reaction looks like a deployment reaction**. The estimated likelihood ratio is then enormous
for every new reaction, the calibration scores carry almost no weight in comparison, and weighted conformal
prediction returns the only interval it can still vouch for: an infinite one. That is the correct answer. Reweighting
can redistribute the evidence in a calibration set; it cannot create evidence about a region the calibration set
never visited.

(A technical remark. `estimate_weights` clips the classifier's probabilities to [0.01, 0.99]; this limit decides how
extreme a weight can become. It is a tuning constant: with a much tighter clip the weights would be too tame to signal
the problem, and the intervals would come out finite without having earned it. When the AUC is close to 1, a finite
weighted interval should not be trusted.)

### The repair that needs no overlap: reference calculations in the deployment domain

What is needed is information about the deployment domain: a (small) number $m$ of reference calculations for
reactions drawn at random from the deployment set. Recalibrating on those alone restores exchangeability, and
Notebook 2 tells us what to expect from $m$ calibration points:
"""

# %%
def recalibration_table(errors_new, sizes=(10, 20, 50, 100, 200, 500), n_rep=500, seed=32):
    """Coverage and half-width when m randomly chosen new cases are labeled and used as the calibration set."""
    rng = np.random.default_rng(seed)
    rows = []
    for m in sizes:
        cov, half = [], []
        for _ in range(n_rep):
            perm = rng.permutation(errors_new)
            q = conformal_quantile(perm[:m], alpha)
            half.append(q)
            cov.append(np.mean(perm[m:] <= q))
        rows.append({"reference calculations m": m, "mean coverage": np.mean(cov), "5%": np.quantile(cov, 0.05),
                     "95%": np.quantile(cov, 0.95), "median half-width": np.median(half)})
    return pd.DataFrame(rows).set_index("reference calculations m")


recal = recalibration_table(err_small[large])
before = size_table.loc[model.label]
print(f"Δ-learning model on the larger molecules; before recalibration: half-width {before['half-width used']:.2f} kcal/mol, "
      f"coverage {before['coverage on larger molecules']:.1%}")
display(recal.round(3))

# %% [markdown]
r"""
On average the coverage is back at 90% for any $m \ge 9$. What $m$ buys is *reliability*: with 20 calculations
the coverage of a particular recalibration lies between about 78% and 98% in nine cases out of ten, with 200 between
about 86% and 93%. The price in width is visible too: small calibration sets give inflated half-widths.

Old and new calibration data can also be combined, with the new ones weighted more strongly (Barber et al., 2023;
Exercise 2 tries this); and when enough reference values from the deployment domain accumulate, the model itself
should be refitted, which goes beyond this tutorial.

## 4. A reaction class the model has never seen

The most drastic shift: an entire reaction class is missing from the fitting set **and** from the calibration set and
then turns up at deployment. We leave out each of the five classes in turn.
"""

# %%
rows, class_models = [], {}
for name in rdb7.models:
    for g in P.CLASS_ORDER:
        fit_rows = rdb7.train[classes[rdb7.train] != g]
        cal_rows = rdb7.hold[classes[rdb7.hold] != g]
        new_rows = rdb7.hold[classes[rdb7.hold] == g]
        mod = P.fit_model(rdb7, name, fit_rows)
        e = np.abs(y - mod.mu)
        q = conformal_quantile(e[cal_rows], alpha)
        after = recalibration_table(e[new_rows], sizes=(50,), n_rep=300, seed=33).iloc[0]
        rows.append({"model": mod.label, "unseen class": g, "MAE, known classes": e[cal_rows].mean(), "MAE, unseen class": e[new_rows].mean(),
                     "half-width used": q, "coverage on the unseen class": np.mean(e[new_rows] <= q),
                     "coverage after 50 reference calculations": after["mean coverage"], "half-width after": after["median half-width"]})
        class_models[(name, g)] = (e, cal_rows, new_rows, q)
        scenario_coverage.setdefault(f"unseen class: {P.CLASS_SHORT[g]}", {})[name] = rows[-1]["coverage on the unseen class"]
unseen = pd.DataFrame(rows).set_index(["model", "unseen class"])
display(unseen.round(3))

# %% [markdown]
r"""
Here the damage is of a different order. Depending on the class and the model, the "90%" interval covers between
about one half and 86% of the reactions of the unseen class. The errors on an unseen class are up to almost two and a
half times larger than on the classes the model knows, and the calibration set, which does not contain the class either,
has no way of knowing. Fifty reference calculations from the unseen class restore the coverage, at the price of
intervals that are honest about the model's poor grasp of that chemistry.

## 5. Hidden dependence: several reactions per reactant

RDB7 contains on average ten reactions for each reactant molecule. Throughout these notebooks the split into fitting
and held-out data was therefore made **by reactant**. Was that necessary? The question can be put to the data. In each
of 60 repetitions the reactants are divided at random into "known" and "new" ones. The models are fitted to 80% of the
reactions of the known reactants, and the interval is calibrated on the remaining 20%: reactions the model has not
seen, of reactants it has seen. The interval is then tested on all reactions of the new reactants. Had we calibrated on
reactions of new reactants instead, the coverage would be $k/(n+1)$ on average (Notebook 2); the shortfall from that
value is the price of calibrating on known reactants.
"""

# %%
rows = []
for rep in range(60):
    known, new = P.group_split(rdb7.reactant, 0.5, seed=100 + rep)             # a fresh division of the reactants
    known = np.random.default_rng(200 + rep).permutation(known)
    n_fit = int(0.8 * len(known))
    fit_rows, cal_known = known[:n_fit], known[n_fit:]                         # known reactants: 80% of the reactions to fit, 20% to calibrate
    k = int(np.ceil((len(cal_known) + 1) * (1 - alpha) - 1e-9))
    for name in rdb7.models:
        e = np.abs(y - P.fit_model(rdb7, name, fit_rows).mu)
        q_known = conformal_quantile(e[cal_known], alpha)
        rows.append({"model": rdb7.models[name].label, "coverage": np.mean(e[new] <= q_known), "expected": k / (len(cal_known) + 1),
                     "width_ratio": q_known / np.quantile(e[new], 1 - alpha)})
runs = pd.DataFrame(rows)
runs["loss"] = 100 * (runs["expected"] - runs["coverage"])                     # percentage points of coverage lost
dependence = runs.groupby("model", sort=False).agg(
    coverage=("coverage", "mean"), expected=("expected", "mean"), loss=("loss", "mean"), loss_se=("loss", "sem"),
    width_ratio=("width_ratio", "mean"))
display(dependence.rename(columns={"coverage": "coverage on new reactants", "expected": "expected without dependence",
                                   "loss": "loss (percentage points)", "loss_se": "standard error of the loss",
                                   "width_ratio": "half-width used / half-width needed"}).round(3))
for name in rdb7.models:
    scenario_coverage.setdefault("calibration reactants seen in fitting", {})[name] = dependence.loc[rdb7.models[name].label, "coverage"]

# %% [markdown]
r"""
In RDB7 the loss is small but systematic: calibrating on other reactions of *known* reactants lowers the coverage on
new reactants by about half a percentage point for either model (0.5 to 0.6 points, more than three times the standard
error of the mean over the 60 repetitions), and the half-width is about 2% too small. Unlike the noise of a single
calibration set, such a loss does not average out and does not shrink with more calibration data. (The experiment needs
its 60 repetitions: a single repetition scatters by about one percentage point, twice the size of the effect.)

Do not generalize the small size of the effect. In data sets with near-duplicates (conformers of one molecule, the
same reaction in several solvents, forward and reverse reactions) the same experiment can reveal a large gap, and
Notebook 2 showed the limiting case: calibrating on the fitting set itself gave about three quarters instead of 90%.
The experiment costs some extra fits; run it instead of assuming.

Clusters have a second effect even when the split is done correctly (Dunn, Wasserman, and Ramdas, 2023, treat such
two-layer data in general). A calibration set of about 1200 reactions that come from only 113 reactants carries less
information than 1200 unrelated reactions. In this experiment the coverage is still 90% on average, but it varies more
from one calibration set to the next than for the same number of unrelated reactions:
"""

# %%
rows = []
for name, mod in rdb7.models.items():
    e, hold, groups = abs_err[name], rdb7.hold, rdb7.reactant[rdb7.hold]
    unique = np.unique(groups)
    rng = np.random.default_rng(36)
    by_reactant, by_reaction = [], []
    for _ in range(1000):
        chosen = np.isin(groups, rng.permutation(unique)[: len(unique) // 5])      # one fifth of the reactants ...
        by_reactant.append(np.mean(e[hold[~chosen]] <= conformal_quantile(e[hold[chosen]], alpha)))
        perm = rng.permutation(hold)                                               # ... versus as many single reactions
        by_reaction.append(np.mean(e[perm[chosen.sum():]] <= conformal_quantile(e[perm[: chosen.sum()]], alpha)))
    rows.append({"model": mod.label, "mean coverage, whole reactants": np.mean(by_reactant), "standard deviation, whole reactants": np.std(by_reactant),
                 "mean coverage, single reactions": np.mean(by_reaction), "standard deviation, single reactions": np.std(by_reaction)})
clusters = pd.DataFrame(rows).set_index("model")
display(clusters.round(4))
ratio_sd = (clusters["standard deviation, whole reactants"] / clusters["standard deviation, single reactions"]).mean()
print(f"calibrating on whole reactants: the coverage scatters {ratio_sd - 1:.0%} more than for the same number of unrelated reactions")

# %% [markdown]
r"""
## 6. Two warning signs

In an experiment we know what we broke. In a project we do not, so we need symptoms.

**Before any reference value from the deployment set exists: can a classifier tell the two sets apart?** The AUC of the
calibration-versus-deployment classifier of Section 2 (a "domain classifier") needs only inputs. Collected for the
scenarios above:
"""

# %%
rng = np.random.default_rng(37)
shuffled = rng.permutation(model.hold)
same_population = estimate_weights(model.X[shuffled[:1500]], model.X[shuffled[1500:2100]])[2]
e_cls, cal_cls, new_cls, q_cls = class_models[("delta", HARD)]
auc_class = estimate_weights(model.X[rng.choice(cal_cls, 1500, replace=False)], model.X[rng.choice(new_cls, 600, replace=False)])[2]

# coverage without any shift, measured in the same way as for the other scenarios
scenario_coverage["same population"] = {
    name: float(np.mean([np.mean(abs_err[name][t] <= conformal_quantile(abs_err[name][c], alpha)) for c, t in P.resplits(rdb7.hold, 1500, 200, seed=40)]))
    for name in rdb7.models}

auc_table = pd.DataFrame({"AUC of the domain classifier": {
    "same population (no shift)": same_population, "different mix of classes": diagnostics["AUC"],
    "larger molecules": auc_size, "unseen class (skeletal)": auc_class}})
auc_table["coverage of the Δ-learning intervals"] = [scenario_coverage[s]["delta"] for s in
                                                    ("same population", "different mix of classes", "larger molecules", "unseen class: skeletal")]
display(auc_table.round(3))

# %% [markdown]
r"""
As a rule of thumb: an AUC near 0.5, a little above or below it, means that the classifier found nothing and the inputs
give no reason for concern. An AUC between 0.5 and about 0.9 means that the sets differ but overlap: weighting has
something to work with. An AUC close to 1 means that the deployment reactions are recognizably *not* from the
calibration population, and the intervals should not be trusted until reference values from the deployment set are
available. Two cautions. The AUC measures whether the inputs differ, not whether the difference matters for the errors
(the two shifts with an AUC of 1 above cost very different amounts of coverage). And it is blind to a change in the
relation between inputs and outcome.

**With a few reference values from the deployment set: a coverage alarm.** Decide in advance on a number $m$, draw $m$
deployment reactions **at random**, compute them at the reference level, and count how many fall outside their
intervals. If nothing has changed, the number of misses among $m$ cases follows the Beta-binomial law of Notebook 2,
so we know how many misses are "too many":
"""

# %%
def alarm_threshold(n_cal, m, alpha=0.10, level=0.05):
    """Smallest number t such that t or more misses among m new cases occur with probability <= level if nothing has changed."""
    k = int(np.ceil((n_cal + 1) * (1 - alpha) - 1e-9))
    return int(stats.betabinom(m, n_cal + 1 - k, k).isf(level)) + 1


sizes = (10, 20, 50, 100, 200, 500)
print("misses that trigger the alarm (calibration set of 1000, 90% intervals):",
      {m: alarm_threshold(1000, m) for m in sizes})


def alarm_rate(errors, draw_new, m, n_cal=1000, n_rep=2000, seed=38):
    """How often does the alarm ring? Each repetition draws a calibration set of n_cal from `errors` and m new cases.
    `draw_new(rng, rest, m)` returns the errors of the new cases; `rest` are the positions not used for calibration."""
    rng = np.random.default_rng(seed)
    threshold, rings = alarm_threshold(n_cal, m), 0
    for _ in range(n_rep):
        perm = rng.permutation(len(errors))
        q = conformal_quantile(errors[perm[:n_cal]], alpha)
        rings += np.sum(draw_new(rng, perm[n_cal:], m) > q) >= threshold
    return rings / n_rep


hold_err, hold_is_hard = abs_err["delta"][rdb7.hold], classes[rdb7.hold] == HARD


def new_cases_no_shift(rng, rest, m):                 # m of the held-out reactions that are not in the calibration set
    return hold_err[rest[:m]]


def new_cases_other_mix(rng, rest, m):                # the same, but 60% of them skeletal rearrangements
    n_hard = int(round(0.6 * m))
    return np.r_[hold_err[rng.choice(rest[hold_is_hard[rest]], n_hard, replace=False)],
                 hold_err[rng.choice(rest[~hold_is_hard[rest]], m - n_hard, replace=False)]]


def new_cases_larger(rng, rest, m):                   # molecules with seven heavy atoms (model of Section 3)
    return rng.choice(err_small[large], m, replace=False)


def new_cases_unseen_class(rng, rest, m):             # the class that was left out (model of Section 4)
    return rng.choice(e_cls[new_cls], m, replace=False)


scenarios = {"same population (no shift)": (hold_err, new_cases_no_shift),
             "different mix of classes": (hold_err, new_cases_other_mix),
             "larger molecules": (err_small[cal_small], new_cases_larger),
             "unseen class (skeletal)": (e_cls[cal_cls], new_cases_unseen_class)}
alarm = pd.DataFrame({label: {m: alarm_rate(errors, draw, m) for m in sizes} for label, (errors, draw) in scenarios.items()}).T
alarm.columns.name = "reference calculations m"
# the exact probability of a false alarm, from the Beta-binomial law (k = 901 for n = 1000 and alpha = 0.10)
alarm_exact = {m: float(stats.betabinom(m, 1001 - 901, 901).sf(alarm_threshold(1000, m) - 1)) for m in sizes}
print("Probability that the alarm rings (first row: exact; the others: estimated from 2000 simulated cases each)")
display(pd.concat([pd.DataFrame({"no shift, exact probability": alarm_exact}).T, alarm]).round(3))

# the same alarm when the count is inspected after each of the six values of m instead of once
rng = np.random.default_rng(39)
thresholds = [alarm_threshold(1000, m) for m in sizes]
rings = 0
for _ in range(2000):
    perm = rng.permutation(hold_err)
    misses = perm[1000:1000 + max(sizes)] > conformal_quantile(perm[:1000], alpha)
    rings += any(misses[:m].sum() >= t for m, t in zip(sizes, thresholds))
repeated_looks = rings / 2000
print(f"no shift, but the count is inspected at all six values of m: the alarm rings in {repeated_looks:.0%} of the cases")

# %% [markdown]
r"""
Without a shift the alarm rings in at most 5% of the cases, as designed (first row; the simulated values in the second
row scatter around it). A drastic shift (the unseen class) is caught with 20 to 50 reference calculations. A loss of a
few percentage points (the larger molecules) is still missed almost half of the time with 500: small violations are
hard to detect for the same reason that small calibration sets give noisy coverage. An alarm that stays silent after
20 calculations is therefore weak evidence that all is well.

The 5% level holds for **one look at a number $m$ fixed in advance**. If the count is inspected again and again while the
results arrive, false alarms become more frequent (last line of the output); in that case divide the 5% among the
planned looks, or use a sequential test.
"""

# %%
fig, axes = P.panels(3, height=2.9, gridspec_kw={"width_ratios": [1.25, 1, 1]})

# a: coverage of nominal 90% intervals in every scenario, both models
ax = axes[0]
order = ["same population", "calibration reactants seen in fitting", "different mix of classes", "larger molecules"] + \
        [f"unseen class: {P.CLASS_SHORT[g]}" for g in ("carbene product", "zwitterionic product", "fragmentation", "H-transfer isomerization", "skeletal rearrangement")]
labels = ["same population", "reactants seen in fitting", "different class mix", "larger molecules",
          "unseen: carbene", "unseen: zwitterion", "unseen: fragmentation", "unseen: H transfer", "unseen: skeletal"]
ypos = np.arange(len(order))[::-1]
ax.axvline(1 - alpha, color=P.GRAY, lw=0.8, ls="--")
for name, shift in (("structure", 0.14), ("delta", -0.14)):            # the two markers of a row are drawn slightly apart
    ax.plot([scenario_coverage[s][name] for s in order], ypos + shift, ls="none", marker=P.MODEL_MARKERS[name], color=P.MODEL_COLORS[name],
            label=rdb7.models[name].label.replace(" model", ""))
ax.set_yticks(ypos, labels)
ax.set(xlabel="coverage of 90% intervals", xlim=(0.4, 1.0), ylim=(-0.7, len(order) - 0.3))
ax.set_xticks([0.4, 0.6, 0.8, 1.0])
ax.legend(loc="upper left", handletextpad=0.2, borderaxespad=0.2)
P.panel_label(ax, "a", "What a shift costs")

# b: recalibration with m reference calculations from the deployment set (larger molecules)
ax = axes[1]
ax.fill_between(recal.index, recal["5%"], recal["95%"], color=P.SKY, alpha=0.35, lw=0)
ax.plot(recal.index, recal["mean coverage"], marker="o", color=P.BLUE)
ax.axhline(1 - alpha, color=P.GRAY, lw=0.8, ls="--")
ax.axhline(before["coverage on larger molecules"], color=P.VERMILLION, lw=1.2, ls=":")
ax.text(480, before["coverage on larger molecules"] - 0.01, "before recalibration", fontsize=6.5, va="top", ha="right")
ax.text(13, 0.955, "90% of all\nrecalibrations", fontsize=6.5, va="center")
ax.text(480, 0.912, "mean", fontsize=6.5, va="bottom", ha="right")
ax.set_xscale("log")
ax.set_xticks(list(recal.index), [str(m) for m in recal.index])
ax.minorticks_off()
ax.set(xlabel="reference calculations $m$", ylabel="coverage on larger molecules", ylim=(0.68, 1.0))
P.panel_label(ax, "b", "Recalibration")

# c: how fast does the coverage alarm ring?
ax = axes[2]
alarm_styles = {"unseen class (skeletal)": (P.VERMILLION, "o", "unseen class"), "different mix of classes": (P.ORANGE, "s", "class mix"),
                "larger molecules": (P.GREEN, "^", "larger"), "same population (no shift)": (P.GRAY, "D", "no shift")}
for label, (color, marker, short) in alarm_styles.items():
    ax.plot(sizes, alarm.loc[label], marker=marker, color=color, label=short)
ax.set_xscale("log")
ax.set_xticks(list(sizes), [str(m) for m in sizes])
ax.minorticks_off()
ax.set(xlabel="reference calculations $m$", ylabel="probability that the alarm rings", ylim=(-0.03, 1.38))
ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
ax.legend(loc="upper center", ncol=2, columnspacing=1.0, handletextpad=0.4, borderaxespad=0.2)
P.panel_label(ax, "c", "Coverage alarm")
P.savefig(fig, "nb5_exchangeability")
plt.show()

# %% [markdown]
r"""
Panel **a**: coverage of nominal 90% split conformal intervals in every scenario of this notebook, for both models
(the axis starts at 0.4). Panel **b**: recalibrating on $m$ reference calculations from the deployment set puts the
coverage back at 90% on average; $m$ decides how far a single recalibration can be off (band: the range of 90% of all
recalibrations; the axis starts at 0.68). Panel **c**: the probability that the coverage alarm rings after $m$ reference
calculations.

## 7. What to do

| What changed between calibration and use | How it shows | What to do |
|---|---|---|
| nothing: the calibration set is a random sample of what will be predicted | classifier AUC ≈ 0.5 | use the intervals as they are |
| the proportions of known categories | category frequencies differ | Mondrian calibration, or weights from the frequency ratios |
| the distribution of the inputs, with overlap | AUC between 0.5 and about 0.9; effective sample size still large | weighted conformal prediction with learned weights |
| inputs outside the calibration domain (other sizes, other classes) | AUC ≈ 1; weighted intervals become infinite | do not trust the intervals; compute reference values for a random sample of the deployment cases and recalibrate; refit when possible |
| the relation between input and outcome (another level of theory, other conditions) | invisible in the inputs; only the coverage alarm shows it | recalibrate on data from the deployment setting |
| dependence within the data (clusters, near-duplicates) | known from how the data were generated | split by cluster; expect noisier coverage |

For data that arrive as a stream and drift slowly (a growing database, an active-learning loop), the calibration set
can be updated continuously; "adaptive conformal inference" (Gibbs and Candès, 2021) is the keyword.

## Take-aways

* The guarantee is about the pair "calibration set and deployment set". Ask whether they could have been drawn from
  the same hat.
* A changed mix of known categories is harmless with Mondrian calibration and repairable with weights.
* Reweighting cannot extrapolate. For other sizes and other classes of chemistry the honest interval is infinite until
  reference values from the deployment domain exist; a few dozen of them restore validity.
* Split by cluster when the data have clusters, and measure whether it matters.
* Monitor: a domain classifier before reference values exist, a coverage alarm once they arrive.

## Try it yourself

1. In Section 2, raise the share of skeletal rearrangements to 80%. What happens to the effective sample size and to
   the width of the weighted intervals?
2. In Section 3, calibrate on a *mixture* of the old calibration set and 50 reactions with seven heavy atoms, giving
   each of the 50 the weight of 20 old ones and the test point the same weight as one of the 50 (use
   `weighted_conformal_quantile`). Compare with recalibrating on the 50 alone.
3. Apply the coverage alarm to the "different mix" scenario after Mondrian calibration. Does it still ring?
4. In Section 5, raise the number of repetitions from 60 to 200 (about ten minutes). Does a difference between the two
   models emerge? Which model would you expect to depend more on having seen the reactant, and why?

## References

* Tibshirani, R. J.; Barber, R. F.; Candès, E. J.; Ramdas, A. Conformal Prediction Under Covariate Shift. In *Advances
  in Neural Information Processing Systems 32*; Curran Associates, Inc., 2019; pp 2530–2540.
  https://proceedings.neurips.cc/paper/2019/hash/8fb21ee7a2207526da55a679f0332de2-Abstract.html
* Bickel, S.; Brückner, M.; Scheffer, T. Discriminative Learning Under Covariate Shift. *J. Mach. Learn. Res.* **2009**,
  *10* (75), 2137–2155. https://jmlr.org/papers/v10/bickel09a.html (likelihood ratios from a classifier)
* Barber, R. F.; Candès, E. J.; Ramdas, A.; Tibshirani, R. J. Conformal Prediction Beyond Exchangeability. *Ann. Stat.*
  **2023**, *51*, 816–845. https://doi.org/10.1214/23-AOS2276
* Dunn, R.; Wasserman, L.; Ramdas, A. Distribution-Free Prediction Sets for Two-Layer Hierarchical Models. *J. Am.
  Stat. Assoc.* **2023**, *118*, 2491–2502. https://doi.org/10.1080/01621459.2022.2060112 (clustered data)
* Gibbs, I.; Candès, E. J. Adaptive Conformal Inference Under Distribution Shift. In *Advances in Neural Information
  Processing Systems 34*; Curran Associates, Inc., 2021; pp 1660–1672.
  https://proceedings.neurips.cc/paper/2021/hash/0d441de75945e5acbc865406fc9a2559-Abstract.html
"""

# %%
results = {
    "mix": {f"{model_label} | {method}": dict(row) for (model_label, method), row in mix.to_dict("index").items()},
    "mix_diagnostics": dict(diagnostics),
    "single_example": {"q_plain": q_plain, "coverage_plain": float(np.mean(err[deploy] <= q_plain)),
                       "coverage_weighted": float(np.mean(err[deploy] <= q_weighted)), "coverage_learned": float(np.mean(err[deploy] <= q_learned)),
                       "share_hard_cal": float(np.mean(classes[cal] == HARD))},
    "size": {label: dict(row) for label, row in size_table.to_dict("index").items()},
    "size_auc": auc_size, "size_weighted_infinite": float(np.mean(np.isinf(q_w))),
    "recalibration": {str(m): dict(row) for m, row in recal.to_dict("index").items()},
    "unseen_class": {f"{model_label} | {P.CLASS_SHORT[g]}": dict(row) for (model_label, g), row in unseen.to_dict("index").items()},
    "dependence": {label: dict(row) for label, row in dependence.to_dict("index").items()},
    "clusters": {label: dict(row) for label, row in clusters.to_dict("index").items()},
    "clusters_extra_scatter": float(ratio_sd - 1),
    "same_population": scenario_coverage["same population"],
    "auc": {label: dict(row) for label, row in auc_table.to_dict("index").items()},
    "alarm_thresholds": {str(m): alarm_threshold(1000, m) for m in sizes},
    "alarm": {label: {str(m): v for m, v in row.items()} for label, row in alarm.to_dict("index").items()},
    "alarm_exact": {str(m): v for m, v in alarm_exact.items()},
    "alarm_repeated_looks": repeated_looks,
}
P.save_results("nb5", results, rdb7)
P.check("nb5", results);
