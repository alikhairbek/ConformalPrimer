# NOTEBOOK: 04_mondrian
# %% [markdown]
r"""
# Notebook 4 · Mondrian conformal prediction: a guarantee for every reaction class

**Conformal Prediction for Physical Chemists: hands-on notebooks (4 of 6)**

Notebook 3 ended with a limit: coverage cannot be guaranteed for every individual reaction. It *can* be guaranteed for
every member of a finite list of groups that is fixed in advance. The recipe is very simple: run split conformal
prediction separately within each group. Because the groups tile the data like the rectangles of a Mondrian painting,
this is called Mondrian (or group-conditional) conformal prediction (Vovk, Gammerman, and Shafer, 2005; Vovk, 2013). In
cheminformatics it is in routine use, for example to treat active and inactive compounds separately (Norinder et al.,
2014).

**You will**

1. calibrate one conformal quantile per reaction class and obtain 90% coverage in *each* class;
2. understand the price: every class now has its own, smaller calibration set;
3. see a taxonomy that is too fine for the calibration budget, and how to repair it;
4. combine Mondrian calibration with the normalized score of Notebook 3;
5. learn how to choose the categories.

Runs on a CPU in under a minute.
"""

# %% [setup]

# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

P.set_style()
rdb7 = P.load_problem()
y = rdb7.y
alpha = 0.10

model = rdb7.models["delta"]
mu = model.mu
abs_err = np.abs(y - mu)
classes = rdb7.rxn_class
sigma = P.fit_difficulty(rdb7, "delta")       # the difficulty estimate of Notebook 3

shuffled = np.random.default_rng(P.SEED).permutation(model.hold)
cal, test = shuffled[:1000], shuffled[1000:]  # the same split as in Notebooks 2 and 3

conformal_quantile = P.conformal_quantile

# %% [markdown]
r"""
## 1. One quantile per class

Recall the five reaction classes of Notebook 1 (assigned automatically to each reaction) and how unevenly the
constant-width interval treats them. In Mondrian conformal prediction the calibration scores are sorted into the
classes, and each class gets its own conformal quantile.
"""

# %%
def mondrian_quantiles(scores, groups, alpha=0.10):
    """One conformal quantile per group: {group label: q}."""
    return {g: conformal_quantile(scores[groups == g], alpha) for g in np.unique(groups)}


def mondrian_conformal(y_cal, mu_cal, groups_cal, mu_test, groups_test, alpha=0.10):
    """Absolute-residual intervals with a separate half-width for every group. Returns (lower, upper, {group: q})."""
    q = mondrian_quantiles(np.abs(y_cal - mu_cal), groups_cal, alpha)
    q_test = np.array([q.get(g, np.inf) for g in groups_test])      # a group without calibration data gets no finite interval
    return mu_test - q_test, mu_test + q_test, q


q_global = conformal_quantile(abs_err[cal], alpha)
lower, upper, q_class = mondrian_conformal(y[cal], mu[cal], classes[cal], mu[test], classes[test], alpha)
assert q_class == P.mondrian_quantiles(abs_err[cal], classes[cal], alpha)

inside_global = abs_err[test] <= q_global
inside_mondrian = (y[test] >= lower) & (y[test] <= upper)
table = pd.DataFrame({
    "calibration reactions": pd.Series(classes[cal]).value_counts(),
    "half-width, one q for all": q_global,
    "half-width, Mondrian": pd.Series(q_class),
    "coverage, one q for all": pd.Series(inside_global).groupby(classes[test]).mean(),
    "coverage, Mondrian": pd.Series(inside_mondrian).groupby(classes[test]).mean(),
}).reindex(P.CLASS_ORDER)
display(table.round(3))
print(f"overall coverage: one q for all {inside_global.mean():.1%}, Mondrian {inside_mondrian.mean():.1%}")

# %% [markdown]
r"""
The class-wise half-widths differ by a factor of two to three: reactions with a carbene product need roughly
±2 kcal/mol, skeletal rearrangements roughly ±6 to ±7 kcal/mol. With these half-widths every class is covered at close to
90%.

**The guarantee.** Within each class the calibration scores and the score of a new reaction *of that class* are
exchangeable, so the rank argument of Notebook 2 applies class by class:

$$\Pr\{\,Y \in C(X) \mid \text{class of } X = g\,\} \;\ge\; 1-\alpha \qquad \text{for every class } g .$$

This is a strictly stronger statement than the marginal guarantee, and it implies it.

The table above is for one calibration set. Averaged over 200 random calibration/test splits:
"""

# %%
def class_coverage(inside, groups):
    return {g: inside[groups == g].mean() for g in P.CLASS_ORDER}


rows, per_split = [], []
for cal_b, test_b in P.resplits(model.hold, 1000, 200, seed=21):
    q_b = conformal_quantile(abs_err[cal_b], alpha)
    lo, hi, qc = mondrian_conformal(y[cal_b], mu[cal_b], classes[cal_b], mu[test_b], classes[test_b], alpha)
    inside = {"one q for all": abs_err[test_b] <= q_b, "Mondrian": (y[test_b] >= lo) & (y[test_b] <= hi)}
    for method, ins in inside.items():
        rows.append({"method": method, "all": ins.mean(), "mean half-width": q_b if method == "one q for all" else np.mean(hi - lo) / 2,
                     **class_coverage(ins, classes[test_b])})
    per_split.append({"n_cal": pd.Series(classes[cal_b]).value_counts(), "coverage": pd.Series(class_coverage(inside["Mondrian"], classes[test_b])),
                      "q": pd.Series(qc)})
average = pd.DataFrame(rows).groupby("method", sort=False).mean()
print("Coverage (mean over 200 splits)")
display(average[["all"] + P.CLASS_ORDER].rename(columns=P.CLASS_SHORT).round(3))
print("mean half-width over all test reactions (kcal/mol):", average["mean half-width"].round(2).to_dict())

# %% [markdown]
r"""
## 2. The price: smaller calibration sets

Nothing is free. A class with $n_g$ calibration reactions has the coverage distribution of a calibration set of size
$n_g$ (the Beta law of Notebook 2), not of size $n$. With 1000 calibration reactions in total, the smallest class has
only about 140. Its coverage is right on average but varies noticeably from one calibration set to the next:
"""

# %%
n_cal_class = pd.DataFrame([s["n_cal"] for s in per_split]).mean().reindex(P.CLASS_ORDER)
cov_class = pd.DataFrame([s["coverage"] for s in per_split]).reindex(columns=P.CLASS_ORDER)
spread = pd.DataFrame({
    "mean n_g": n_cal_class.round(0).astype(int),
    "mean coverage": cov_class.mean(),
    "2.5%": cov_class.quantile(0.025),
    "97.5%": cov_class.quantile(0.975),
    "Beta law 2.5%": [P.coverage_law(int(round(n)), alpha).ppf(0.025) for n in n_cal_class],
    "Beta law 97.5%": [P.coverage_law(int(round(n)), alpha).ppf(0.975) for n in n_cal_class],
})
print("Class-wise coverage of Mondrian intervals over 200 calibration sets of 1000 reactions")
display(spread.round(3))

# %% [markdown]
r"""
(For some classes the observed range is a little wider than the Beta law predicts, because each class also has a finite
*test* set here and the number of its calibration reactions varies from split to split. With 200 splits, the 2.5% and
97.5% points are themselves uncertain by a few thousandths.)

The effect becomes dramatic when the total calibration budget is small. Below, the same experiment for calibration
sets of 50 to 2000 reactions: how wide are the class-wise intervals on average, and how often does at least one class
end up with fewer than the 9 calibration reactions that a 90% interval needs (Notebook 2) and therefore with an
infinite interval?
"""

# %%
SMALLEST = "skeletal rearrangement"                       # the class with the fewest reactions
rows = []
for n_cal in (50, 100, 200, 500, 1000, 2000):
    half_widths, infinite, class_cov = [], [], []
    for cal_b, test_b in P.resplits(model.hold, n_cal, 300, seed=22):
        qc = mondrian_quantiles(abs_err[cal_b], classes[cal_b], alpha)
        qs = np.array([qc.get(g, np.inf) for g in P.CLASS_ORDER])
        infinite.append(np.isinf(qs).any())
        half_widths.extend(qs[np.isfinite(qs)])
        for g, q in zip(P.CLASS_ORDER, qs):
            if np.isfinite(q):
                class_cov.append(np.mean(abs_err[test_b][classes[test_b] == g] <= q))
    rows.append({"calibration reactions": n_cal, "smallest class (mean n_g)": n_cal * np.mean(classes[model.hold] == SMALLEST),
                 "splits with an infinite interval": np.mean(infinite), "mean finite half-width": np.mean(half_widths),
                 "class coverage 2.5%": np.quantile(class_cov, 0.025), "class coverage 97.5%": np.quantile(class_cov, 0.975)})
budget = pd.DataFrame(rows).set_index("calibration reactions")
display(budget.round(3))

# %% [markdown]
r"""
With 50 calibration reactions, one class or another gets an infinite interval in almost every split, and with 100 this
still happens now and then. From 200 on the intervals are finite, but the coverage of an individual class (last two
columns: the range that contains 95% of all class-wise coverages with a finite interval) can be far from 90%, and the
half-widths are inflated. As a rule of thumb, each category should hold a few hundred calibration points if its
coverage is to be within a few percentage points of the target: the planning table of Notebook 2 applies to every
category separately.

## 3. A taxonomy that is too fine

The authors of RDB7 labeled about 12% of the reactions with a reaction family of the Reaction Mechanism Generator (RMG;
Gao et al., 2016). Chemically this is the more natural classification. (The largest family, `ketoenol`, is RMG's
template for the 1,3-shift of an atom or group between an oxygen atom and a carbon or nitrogen atom, as in a keto–enol
tautomerization; in RDB7 about half of its 687 reactions shift a hydrogen atom and the other half a heavy-atom group.)
Statistically it is a problem:
"""

# %%
family = rdb7.df["rmg_family"].fillna("(no family)").to_numpy()
q_family = mondrian_quantiles(abs_err[cal], family[cal], alpha)
fam_table = pd.DataFrame({
    "calibration reactions": pd.Series(family[cal]).value_counts(),
    "test reactions": pd.Series(family[test]).value_counts(),
    "half-width (kcal/mol)": pd.Series(q_family),
})
fam_table["calibration reactions"] = fam_table["calibration reactions"].fillna(0).astype(int)
fam_table["test reactions"] = fam_table["test reactions"].fillna(0).astype(int)
fam_table["half-width (kcal/mol)"] = fam_table["half-width (kcal/mol)"].fillna(np.inf)      # no calibration data: no finite interval
fam_table = fam_table.sort_index().sort_values("calibration reactions", ascending=False, kind="stable")
display(fam_table.round(2))
unseen = sorted(set(family[test]) - set(family[cal]))
print(f"families in the test set that do not occur in the calibration set at all: {unseen}")

# %% [markdown]
r"""
Only a few families have enough calibration reactions for a finite 90% interval, and even those rest on a few dozen
scores at most; some families of the test set never occur in the calibration set. The honest answer for these
reactions is "no guarantee at this level with these data", which the infinite half-width expresses.

The repair is to **merge small categories before calibrating**. The merging rule must not look at the calibration
*answers*; the cleanest way is to decide it on the training set, as below: a family keeps its own category only if
the training set contains at least 150 reactions of it.
"""

# %%
train_counts = pd.Series(family[model.train]).value_counts()
keep = set(train_counts[train_counts >= 150].index)
merged = np.array([f if f in keep else "(other families)" for f in family])
categories = sorted(set(merged))

rows = []
for cal_b, test_b in P.resplits(model.hold, 1000, 200, seed=23):
    qm = mondrian_quantiles(abs_err[cal_b], merged[cal_b], alpha)
    for g in categories:
        rows.append({"category": g, "calibration reactions": np.sum(merged[cal_b] == g), "half-width (kcal/mol)": qm[g],
                     "coverage": np.mean(abs_err[test_b][merged[test_b] == g] <= qm[g])})
per_category = pd.DataFrame(rows).groupby("category")
merged_table = pd.DataFrame({
    "training reactions": pd.Series(merged[model.train]).value_counts(),
    "calibration reactions (mean)": per_category["calibration reactions"].mean(),
    "half-width (mean)": per_category["half-width (kcal/mol)"].mean(),
    "coverage (mean)": per_category["coverage"].mean(),
    "coverage 2.5%": per_category["coverage"].quantile(0.025),
    "coverage 97.5%": per_category["coverage"].quantile(0.975),
}).sort_values("training reactions", ascending=False, kind="stable")
print("Merged taxonomy, 200 calibration sets of 1000 reactions")
display(merged_table.round(3))

# %% [markdown]
r"""
Three categories survive. Each is covered at 90% on average, and the two small ones show the scatter that
Section 2 predicts for a few dozen calibration reactions. The merged taxonomy also tells us something about the
chemistry: reactions that match a known template get clearly narrower intervals (about ±2 to ±3 kcal/mol) than the
large remainder of unusual reactions that no template describes (about ±4 kcal/mol).

## 4. Mondrian and normalized scores together

Mondrian calibration equalizes coverage *between* classes; inside a class every reaction still gets the same width.
The normalized score of Notebook 3 adapts the width to the individual reaction but guarantees nothing per class.
The two ideas are independent and can be combined: compute the normalized score $|y-\hat\mu|/\hat\sigma$ and take its
conformal quantile class by class.
"""

# %%
def fifths(values):
    return np.digitize(values, np.quantile(values, [0.2, 0.4, 0.6, 0.8]))


rows = []
for cal_b, test_b in P.resplits(model.hold, 1000, 200, seed=21):
    candidates = {}
    q = conformal_quantile(abs_err[cal_b], alpha)
    candidates["split (one q)"] = np.full(len(test_b), q)
    qc = mondrian_quantiles(abs_err[cal_b], classes[cal_b], alpha)
    candidates["Mondrian"] = np.array([qc[g] for g in classes[test_b]])
    q = conformal_quantile(abs_err[cal_b] / sigma[cal_b], alpha)
    candidates["normalized"] = q * sigma[test_b]
    qc = mondrian_quantiles(abs_err[cal_b] / sigma[cal_b], classes[cal_b], alpha)
    candidates["Mondrian + normalized"] = np.array([qc[g] for g in classes[test_b]]) * sigma[test_b]
    difficulty = fifths(sigma[test_b])
    for method, half in candidates.items():
        inside = abs_err[test_b] <= half
        rows.append({"method": method, "coverage": inside.mean(), "median half-width": np.median(half), "mean half-width": half.mean(),
                     **{g: inside[classes[test_b] == g].mean() for g in P.CLASS_ORDER},
                     "easiest fifth": inside[difficulty == 0].mean(), "hardest fifth": inside[difficulty == 4].mean()})
means = pd.DataFrame(rows).groupby("method", sort=False).mean()
combined = means[["coverage", "median half-width", "mean half-width"]].copy()
combined["worst-covered class"] = means[P.CLASS_ORDER].min(axis=1)
combined["best-covered class"] = means[P.CLASS_ORDER].max(axis=1)
combined[["easiest fifth", "hardest fifth"]] = means[["easiest fifth", "hardest fifth"]]
display(combined.round(3))

# %% [markdown]
r"""
Read the columns as three questions. *Is the overall coverage right?* For all four methods. *Are the classes treated
equally?* Only with Mondrian calibration is this guaranteed (columns "worst-covered class" and "best-covered class");
the normalized score comes close without a guarantee. *Are easy and hard reactions treated equally?* Much more nearly
with the normalized score than without it. The combination does well on all three counts, and it keeps the small
median half-width of the normalized score.
"""

# %%
fig, axes = P.panels(3)
short = [P.CLASS_SHORT[c] for c in P.CLASS_ORDER]
ypos = np.arange(len(P.CLASS_ORDER))[::-1]

# a: class-wise coverage, one quantile versus Mondrian
ax = axes[0]
ax.axvline(1 - alpha, color=P.GRAY, lw=0.8, ls="--")
ax.plot(average.loc["one q for all", P.CLASS_ORDER], ypos, ls="none", marker="s", color=P.VERMILLION, label=r"one $\hat{q}$ for all")
ax.plot(average.loc["Mondrian", P.CLASS_ORDER], ypos, ls="none", marker="o", color=P.BLUE, label="Mondrian")
ax.set_yticks(ypos, short)
ax.set(xlabel="coverage", xlim=(0.75, 1.0), ylim=(-0.6, 4.8))
ax.legend(loc="lower left", handletextpad=0.2, borderaxespad=0.3)
P.panel_label(ax, "a", "Coverage by class")

# b: class-wise half-widths
ax = axes[1]
q_mean = pd.DataFrame([s["q"] for s in per_split]).mean().reindex(P.CLASS_ORDER)
q_all = average.loc["one q for all", "mean half-width"]
ax.axvline(q_all, color=P.VERMILLION, lw=1.2, ls="--")
ax.hlines(ypos, 0, q_mean, color=P.BLUE, lw=5)
for yy, value in zip(ypos, q_mean):
    ax.text(8.6, yy, f"{value:.1f}", va="center", ha="right", fontsize=7)
ax.text(q_all + 0.15, 4.5, r"one $\hat{q}$ for all", fontsize=6.5, va="center")
ax.set_yticks(ypos, short)
ax.set(xlabel="half-width of 90% interval (kcal/mol)", xlim=(0, 8.7), ylim=(-0.6, 4.8))
ax.set_xticks([0, 2, 4, 6, 8])
P.panel_label(ax, "b", "Mondrian half-widths")

# c: how much does the coverage of one class scatter?
ax = axes[2]
grid_n = np.unique(np.round(np.logspace(1, np.log10(3000), 150)).astype(int))
laws = [P.coverage_law(int(n), alpha) for n in grid_n]
ax.fill_between(grid_n, [l.ppf(0.025) for l in laws], [l.ppf(0.975) for l in laws], color=P.SKY, alpha=0.35, lw=0)
ax.plot(spread["mean n_g"], spread["2.5%"], ls="none", marker="v", color=P.VERMILLION)
ax.plot(spread["mean n_g"], spread["97.5%"], ls="none", marker="^", color=P.VERMILLION)
ax.axhline(1 - alpha, color=P.GRAY, lw=0.8, ls="--")
ax.text(0.96, 0.17, "band: 95% range (Beta law)", transform=ax.transAxes, ha="right", va="bottom", fontsize=6.5)
ax.text(0.96, 0.05, "triangles: the five classes,\n200 calibration sets", transform=ax.transAxes, ha="right", va="bottom", fontsize=6.5)
ax.set_xscale("log")
ax.set(xlabel="calibration reactions per class, $n_g$", ylabel="coverage of the class", xlim=(10, 3000), ylim=(0.6, 1.0))
ax.set_xticks([10, 100, 1000], ["10", "100", "1000"])
P.panel_label(ax, "c", "The price: smaller $n$")
for ax in axes:
    ax.set_box_aspect(1)
P.savefig(fig, "nb4_mondrian")
plt.show()

# %% [markdown]
r"""
Panel **a**: class-wise coverage with a single quantile (squares) and with Mondrian calibration (circles), averaged over
200 splits. Panel **b**: the class-wise half-widths behind it (bars) and the single half-width (dashed line). Panel
**c**: each class pays with the coverage variability of its own, smaller calibration set; the triangles are the observed
2.5% and 97.5% points of the five class-wise coverages. The coverage axes start at 0.75 (a) and 0.6 (c).

## 5. Choosing the categories

Any rule may be used that assigns a category to a reaction **from $x$ alone and was fixed before calibration**:

* chemistry: reaction class, functional group, element content, charge, or spin state;
* provenance: which data source, which level of theory, which laboratory;
* size: number of heavy atoms, in bins;
* the model itself: bins of the predicted value, or bins of a difficulty estimate (with bin edges fixed beforehand,
  for example on the training set).

A category may **not** be defined through the unknown answer $y$ in regression ("reactions with a barrier below
40 kcal/mol" is not a valid category; "reactions with a *predicted* barrier below 40 kcal/mol" is). In classification
there is a second, very useful variant in which the categories are the true labels (label-conditional conformal
prediction); it guarantees coverage for every class of the outcome and is the standard remedy for imbalanced data.
Guarantees for groups that overlap need more than this recipe (Gibbs, Cherian, and Candès, 2025).

Three practical rules: keep the number of categories small; give each one a calibration set large enough for the
precision you need; and when the natural taxonomy is too fine, merge on the training set. A further benefit comes
for free: because every category is calibrated on its own, the guarantee survives when the *proportions* of the
categories change between calibration and use. Notebook 5 uses exactly that.

## Take-aways

* Mondrian conformal prediction = split conformal prediction within each category. It guarantees the coverage of
  every category, which the marginal guarantee does not.
* Each category pays with the Beta law of its own calibration set; categories with fewer than $(1-\alpha)/\alpha$
  calibration points get infinite intervals.
* Merge small categories using the training set, never the calibration answers.
* Mondrian calibration and normalized scores address different problems and combine without difficulty.

## Try it yourself

1. Use the number of heavy atoms of the reactant (`rdb7.n_heavy`, merged into "up to 5", "6", and "7") as the taxonomy.
   Are the size classes treated unequally by the single quantile?
2. Use fifths of the difficulty estimate `sigma` as categories, with the four cut points computed once from the
   training reactions (`np.quantile(sigma[model.train], [0.2, 0.4, 0.6, 0.8])`). How does this "binned" adaptivity
   compare with the normalized score?
3. Lower the merging threshold of Section 3 from 150 to 30 training reactions and look at the coverage of the small
   families over many splits.

## References

* Vovk, V.; Gammerman, A.; Shafer, G. *Algorithmic Learning in a Random World*; Springer: New York, 2005
  (https://doi.org/10.1007/b106715); 2nd ed., Springer: Cham, 2022 (https://doi.org/10.1007/978-3-031-06649-8).
  Mondrian conformal predictors are introduced in this book.
* Vovk, V. Conditional Validity of Inductive Conformal Predictors. *Mach. Learn.* **2013**, *92*, 349–376.
  https://doi.org/10.1007/s10994-013-5355-6
* Norinder, U.; Carlsson, L.; Boyer, S.; Eklund, M. Introducing Conformal Prediction in Predictive Modeling. A
  Transparent and Flexible Alternative to Applicability Domain Determination. *J. Chem. Inf. Model.* **2014**, *54*,
  1596–1603. https://doi.org/10.1021/ci5001168
* Gibbs, I.; Cherian, J. J.; Candès, E. J. Conformal Prediction with Conditional Guarantees. *J. R. Stat. Soc. Ser. B:
  Stat. Methodol.* **2025**, *87*, 1100–1126. https://doi.org/10.1093/jrsssb/qkaf008 (guarantees for overlapping groups
  and beyond)
* Gao, C. W.; Allen, J. W.; Green, W. H.; West, R. H. Reaction Mechanism Generator: Automatic Construction of Chemical
  Kinetic Mechanisms. *Comput. Phys. Commun.* **2016**, *203*, 212–225. https://doi.org/10.1016/j.cpc.2016.02.013
"""

# %%
results = {
    "single_split": {"q_global": q_global, "q_class": {P.CLASS_SHORT[g]: q_class[g] for g in P.CLASS_ORDER},
                     "coverage_global": {P.CLASS_SHORT[g]: table.loc[g, "coverage, one q for all"] for g in P.CLASS_ORDER},
                     "coverage_mondrian": {P.CLASS_SHORT[g]: table.loc[g, "coverage, Mondrian"] for g in P.CLASS_ORDER}},
    "average": {method: {P.CLASS_SHORT.get(k, k): v for k, v in row.items()} for method, row in average.to_dict("index").items()},
    "class_spread": {P.CLASS_SHORT[g]: dict(row) for g, row in spread.to_dict("index").items()},
    "budget": {str(n): dict(row) for n, row in budget.to_dict("index").items()},
    "families": {"n_finite": int(np.isfinite(fam_table["half-width (kcal/mol)"]).sum()), "n_families_cal": int((fam_table["calibration reactions"] > 0).sum()),
                 "unseen_in_cal": len(unseen), "kept_after_merging": sorted(keep)},
    "merged": {g: dict(row) for g, row in merged_table.to_dict("index").items()},
    "combined": {method: dict(row) for method, row in combined.to_dict("index").items()},
}
P.save_results("nb4", results, rdb7)
P.check("nb4", results);
