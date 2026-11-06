# NOTEBOOK: 03_adaptive_intervals
# %% [markdown]
r"""
# Notebook 3 · Adaptive intervals: one width does not fit all

**Conformal Prediction for Physical Chemists: hands-on notebooks (3 of 6)**

The interval of Notebook 2 has the same half-width for every reaction. Its coverage is right *on average*, but
Notebook 1 showed that the errors of our models depend on the chemistry. A constant width is then too generous for easy
reactions and too tight for hard ones. This notebook builds intervals whose width follows the difficulty of the
individual prediction, and shows how to check whether the adaptation worked.

**You will**

1. see the problem: the right coverage overall, the wrong coverage for subgroups;
2. build a difficulty estimate $\hat\sigma(x)$ and *normalized* conformal intervals $\hat\mu(x) \pm \hat q\,\hat\sigma(x)$;
3. build *conformalized quantile regression* (CQR) intervals;
4. compare the three kinds of interval with diagnostics for conditional coverage;
5. learn what adaptive intervals can and cannot promise.

Runs on a CPU in under a minute.
"""

# %% [setup]

# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

P.set_style()
rdb7 = P.load_problem()
y = rdb7.y
alpha = 0.10

model = rdb7.models["delta"]        # the Δ-learning model, as in Notebook 2
mu = model.mu

# the same calibration/test split as in Notebook 2
shuffled = np.random.default_rng(P.SEED).permutation(model.hold)
cal, test = shuffled[:1000], shuffled[1000:]

conformal_quantile = P.conformal_quantile     # derived in Notebook 2

# %% [markdown]
r"""
## 1. The problem: right on average, wrong for subgroups

We start from the split conformal interval of Notebook 2 and ask a sharper question than how many test barriers are
covered, namely, how many are covered *in each reaction class*.
"""

# %%
lower, upper, q_split = P.split_conformal(y[cal], mu[cal], mu[test], alpha)
covered = (y[test] >= lower) & (y[test] <= upper)

by_class = pd.DataFrame({
    "test reactions": pd.Series(rdb7.rxn_class[test]).value_counts(),
    "coverage": pd.Series(covered).groupby(rdb7.rxn_class[test]).mean(),
    "MAE (kcal/mol)": pd.Series(np.abs(y - mu)[test]).groupby(rdb7.rxn_class[test]).mean(),
}).reindex(P.CLASS_ORDER)
print(f"split conformal: every interval is mu ± {q_split:.2f} kcal/mol; overall coverage {covered.mean():.1%}")
display(by_class.round(3))

# %% [markdown]
r"""
Overall the promise is kept, but the classes are treated very differently: nearly all reactions with a carbene product
are covered (the interval is wider than they need), while roughly one skeletal rearrangement in five falls outside its
"90%" interval. This is not a failure of the guarantee, which only ever spoke about the average over all reactions.
It is a reminder that the average can hide a great deal.

The remedy is to let the width depend on the reaction. Two standard constructions follow. Both keep the logic of
Notebook 2 (score, conformal quantile, interval) and change only the *score*.

## 2. A difficulty estimate $\hat\sigma(x)$

The first construction needs a number $\hat\sigma(x) > 0$ that is large when the model is likely to be wrong for
reaction $x$. Anything that can be computed without knowing the answer is allowed: the spread of an ensemble, the
distance to the training set, a predicted variance. Here we train a second model to predict the size of the first
model's error. Three steps, all on the **training set only**:

1. **Honest residuals.** The residuals of the training reactions themselves are useless for this purpose: the model
   has partly memorized them. We therefore use 5-fold cross-validation and record, for every training reaction, the
   error of a model that was fitted *without* it (an out-of-fold residual). The folds keep all reactions of one
   reactant together, in line with the split of Notebook 1.
2. **Learn the error size.** A second, smaller gradient-boosting model is fitted to predict $|\text{out-of-fold residual}|$
   from the same features.
3. **Keep it positive.** The prediction is bounded from below (here by half the median out-of-fold residual), because
   we are going to divide by it.
"""

# %%
# 1. out-of-fold residuals of the training reactions
train = model.train
fold = P.grouped_folds(rdb7.reactant[train], 5, P.SEED)          # fold number of every training reaction
oof = np.empty(len(train))
for k in range(5):
    fold_model = P.make_model().fit(model.X[train[fold != k]], model.target[train[fold != k]])
    oof[fold == k] = fold_model.predict(model.X[train[fold == k]])
abs_res_oof = np.abs(model.target[train] - oof)
abs_res_in = np.abs(model.target[train] - model.estimator.predict(model.X[train]))
print(f"mean |residual| of the training reactions: in-sample {abs_res_in.mean():.2f}, out-of-fold {abs_res_oof.mean():.2f} kcal/mol"
      f"   (held-out MAE: {np.abs(y - mu)[model.hold].mean():.2f})")

# 2. a second model predicts the size of the error;  3. bounded from below
sigma_model = P.make_model(max_iter=P.DIFFICULTY_TREES).fit(model.X[train], abs_res_oof)
floor = P.DIFFICULTY_FLOOR * np.median(abs_res_oof)
sigma = np.maximum(sigma_model.predict(model.X), floor)

assert np.allclose(sigma, P.fit_difficulty(rdb7, model.name, verbose=False))     # the same recipe is stored in primer.py
hold = model.hold
print(f"sigma on the held-out reactions: median {np.median(sigma[hold]):.2f}, "
      f"range {sigma[hold].min():.2f} to {sigma[hold].max():.2f} kcal/mol; {np.mean(sigma[hold] == floor):.0%} sit at the lower bound")

# %% [markdown]
r"""
The out-of-fold residuals have the same size as the errors on the held-out reactions; the in-sample residuals are
about half as large. Before using $\hat\sigma$ we should check that it has anything to say about the actual errors of
reactions that neither model has seen:
"""

# %%
abs_err = np.abs(y - mu)
rank = np.empty(len(hold), dtype=int)
rank[np.argsort(sigma[hold], kind="stable")] = np.arange(len(hold))
tenth = rank * 10 // len(hold)                                    # 0 = the tenth with the smallest sigma, ..., 9 = the largest
reliability = pd.DataFrame({"mean sigma": pd.Series(sigma[hold]).groupby(tenth).mean(),
                            "mean |error|": pd.Series(abs_err[hold]).groupby(tenth).mean()})
reliability.index = [f"tenth {i + 1}" for i in reliability.index]
rho = stats.spearmanr(sigma[hold], abs_err[hold])[0]
display(reliability.T.round(2))
print(f"Spearman rank correlation between sigma and |error| on the held-out reactions: {rho:.2f}")

# %% [markdown]
r"""
From the easiest to the hardest tenth of the held-out reactions the mean error grows about ninefold, and
$\hat\sigma$ follows it closely. The rank correlation for individual reactions is moderate (an error is a random
quantity; $\hat\sigma$ can only predict its typical size), but that is all the next step needs.

## 3. Normalized conformal intervals

Replace the score $|y - \hat\mu(x)|$ by the **normalized** score

$$ s(x, y) = \frac{|y - \hat\mu(x)|}{\hat\sigma(x)} , $$

take the conformal quantile $\hat q$ of the calibration scores exactly as before, and solve $s(x, y) \le \hat q$ for $y$:

$$ C(x) = \big[\hat\mu(x) - \hat q\,\hat\sigma(x),\ \hat\mu(x) + \hat q\,\hat\sigma(x)\big] . $$

$\hat q$ is now a dimensionless multiplier, and the half-width $\hat q\,\hat\sigma(x)$ differs from reaction to
reaction. The guarantee is untouched: the rank argument of Notebook 2 holds for *any* score, as long as the score
function (here $\hat\mu$ and $\hat\sigma$) was fixed before the calibration data were looked at. (Normalized scores go
back to Papadopoulos et al., 2002 and 2008; Lei et al., 2018, call the result "locally weighted" conformal prediction.)
"""

# %%
def normalized_conformal(y_cal, mu_cal, sigma_cal, mu_test, sigma_test, alpha=0.10):
    """Interval mu ± q * sigma(x) from the score |y - mu| / sigma. Returns (lower, upper, q)."""
    q = conformal_quantile(np.abs(y_cal - mu_cal) / sigma_cal, alpha)
    return mu_test - q * sigma_test, mu_test + q * sigma_test, q


lo_norm, hi_norm, q_norm = normalized_conformal(y[cal], mu[cal], sigma[cal], mu[test], sigma[test], alpha)
half_width = (hi_norm - lo_norm) / 2
print(f"q_hat = {q_norm:.2f} (dimensionless); coverage {np.mean((y[test] >= lo_norm) & (y[test] <= hi_norm)):.1%}")
print(f"half-widths: median {np.median(half_width):.2f}, 5th to 95th percentile {np.quantile(half_width, 0.05):.2f} "
      f"to {np.quantile(half_width, 0.95):.2f} kcal/mol   (split conformal: {q_split:.2f} for every reaction)")
print(f"{np.mean(half_width < q_split):.0%} of the test reactions now have a narrower interval than with split conformal")

# %% [markdown]
r"""
## 4. Conformalized quantile regression (CQR)

The second construction (Romano, Patterson, and Candès, 2019) does not go through a point prediction at all. Gradient
boosting can be trained to predict a *quantile* of the barrier instead of its mean, by minimizing the "pinball" loss
$\rho_\tau(u) = \max\{\tau u,\ (\tau - 1)\,u\}$ of the residual $u$ for the quantile level $\tau$. We fit one model for
the 5% quantile and one for the 95% quantile; the band between them is a first, uncalibrated guess of a 90% interval.
"""

# %%
q_lo_model = P.make_model(loss="quantile", quantile=alpha / 2).fit(model.X[train], model.target[train])
q_hi_model = P.make_model(loss="quantile", quantile=1 - alpha / 2).fit(model.X[train], model.target[train])
band_lo = q_lo_model.predict(model.X) + model.baseline
band_hi = q_hi_model.predict(model.X) + model.baseline

raw_coverage = np.mean((y[hold] >= band_lo[hold]) & (y[hold] <= band_hi[hold]))
print(f"raw 5%–95% quantile band on the held-out reactions: coverage {raw_coverage:.1%} (nominal 90%)")

# %% [markdown]
r"""
The raw band is too narrow: quantile models overfit like any other model. CQR repairs this with one more conformal
step. The score measures how far $y$ lies *outside* the band (it is negative when $y$ is inside),

$$ s(x, y) = \max\{\hat\ell(x) - y,\ \ y - \hat u(x)\} , $$

where $\hat\ell(x)$ and $\hat u(x)$ are the predictions of the lower and the upper quantile model. The interval is the
band, widened (or, if $\hat q < 0$, shrunk) by the conformal quantile $\hat q$ of the calibration scores:
$C(x) = [\hat\ell(x) - \hat q,\ \hat u(x) + \hat q]$.
"""

# %%
def cqr(y_cal, lo_cal, hi_cal, lo_test, hi_test, alpha=0.10):
    """Conformalized quantile regression. Returns (lower, upper, q)."""
    q = conformal_quantile(np.maximum(lo_cal - y_cal, y_cal - hi_cal), alpha)
    return lo_test - q, hi_test + q, q


lo_cqr, hi_cqr, q_cqr = cqr(y[cal], band_lo[cal], band_hi[cal], band_lo[test], band_hi[test], alpha)
print(f"q_hat = {q_cqr:.2f} kcal/mol is added on both sides; coverage {np.mean((y[test] >= lo_cqr) & (y[test] <= hi_cqr)):.1%}")

for name, mine, stored in (("normalized", (lo_norm, hi_norm), P.normalized_conformal(y[cal], mu[cal], sigma[cal], mu[test], sigma[test], alpha)),
                           ("CQR", (lo_cqr, hi_cqr), P.cqr(y[cal], band_lo[cal], band_hi[cal], band_lo[test], band_hi[test], alpha))):
    assert np.array_equal(mine[0], stored[0]) and np.array_equal(mine[1], stored[1])     # same functions as in primer.py

# %% [markdown]
r"""
## 5. Did the intervals adapt? Diagnostics for conditional coverage

All three methods have 90% coverage overall. To compare them we need two more things.

**Efficiency.** Narrower is better, *given* the same coverage. We report the mean and the median half-width, and the
*interval score* (Gneiting and Raftery, 2007), which is the full width of the interval plus a penalty of $2/\alpha$
times the distance by which the interval misses the reference value; it rewards narrow intervals but charges for
misses in proportion to their size.

**Conditional coverage.** We would like 90% coverage not only overall but in every recognizable subgroup. Three
checks:

* *by a feature*: coverage within each reaction class;
* *by difficulty*: coverage within each fifth of the test reactions ordered by $\hat\sigma(x)$, from the easiest fifth
  to the hardest. This check uses the very quantity the normalized interval is built from, so it favors that method;
* *by the method's own interval width*: are the wide intervals covered as often as the narrow ones? This check can
  be made for any adaptive method and favors none.

Everything is averaged over 200 random calibration/test splits of the held-out reactions.
"""

# %%
def fifths(values):
    """0 for the fifth of the cases with the smallest values, ..., 4 for the fifth with the largest."""
    return np.digitize(values, np.quantile(values, [0.2, 0.4, 0.6, 0.8]))


def evaluate(lower, upper, test_rows, difficulty):
    """Coverage, half-width, and conditional-coverage diagnostics of one set of intervals."""
    inside = (y[test_rows] >= lower) & (y[test_rows] <= upper)
    half = (upper - lower) / 2
    out = {"coverage": inside.mean(), "mean half-width": half.mean(), "median half-width": np.median(half),
           "interval score": P.interval_score(y[test_rows], lower, upper, alpha)}
    for c in P.CLASS_ORDER:
        out[f"class: {P.CLASS_SHORT[c]}"] = inside[rdb7.rxn_class[test_rows] == c].mean()
    fifth = fifths(difficulty)
    for j in range(5):
        out[f"sigma fifth {j + 1}"] = inside[fifth == j].mean()
    if np.ptp(half) > 1e-9:                                         # only meaningful if the widths differ
        fifth = fifths(half)
        for j in range(5):
            out[f"own-width fifth {j + 1}"] = inside[fifth == j].mean()
    return out


METHODS = ["split", "normalized", "CQR"]
rows = []
for cal_b, test_b in P.resplits(model.hold, 1000, 200, seed=11):
    intervals = {
        "split": P.split_conformal(y[cal_b], mu[cal_b], mu[test_b], alpha),
        "normalized": normalized_conformal(y[cal_b], mu[cal_b], sigma[cal_b], mu[test_b], sigma[test_b], alpha),
        "CQR": cqr(y[cal_b], band_lo[cal_b], band_hi[cal_b], band_lo[test_b], band_hi[test_b], alpha),
    }
    for method, (lo, hi, _) in intervals.items():
        rows.append({"method": method, **evaluate(lo, hi, test_b, sigma[test_b])})
summary = pd.DataFrame(rows).groupby("method").mean().reindex(METHODS)

print("Efficiency (half-widths and interval score in kcal/mol)")
display(summary[["coverage", "mean half-width", "median half-width", "interval score"]].round(3))
print("Coverage by reaction class")
display(summary[[c for c in summary.columns if c.startswith("class")]].round(3))
print("Coverage by difficulty: fifths of the test reactions ordered by sigma (1 = easiest)")
display(summary[[c for c in summary.columns if c.startswith("sigma fifth")]].round(3))
print("Coverage by the method's own interval width (1 = narrowest fifth)")
display(summary.loc[["normalized", "CQR"], [c for c in summary.columns if c.startswith("own-width")]].round(3))

# %% [markdown]
r"""
How to read the four tables:

* **Efficiency.** Both adaptive methods have a smaller mean half-width than the constant interval, a much smaller
  *median* half-width (most reactions are easy and now get a tighter interval), and a clearly better interval score.
* **Classes.** The constant interval covers only about 80% of the skeletal rearrangements and about 97% of the reactions
  with a carbene product. The normalized interval brings every class to within about one and a half percentage points
  of 90%, although it was not given the class label (the descriptors from which the classes are derived are among its
  inputs).
* **Difficulty.** The constant interval covers practically all of the easiest fifth and only about two thirds of the
  hardest fifth: its misses are concentrated exactly where a user most needs a warning. The normalized interval stays
  between about 88% and 93%. CQR improves on the constant interval but still over-covers easy and under-covers hard
  reactions here.
* **Own width.** For CQR the wide intervals are covered less often than the narrow ones: its widest intervals are still
  too narrow. The reason is visible in the recipe: CQR adds the *same* $\hat q$ to a narrow band and to a wide band.

Which adaptive method wins depends on the data and on the auxiliary models. On other problems CQR can be the better
choice, for example when the error distribution is skewed, because its two ends are fitted separately (Romano et al.,
2019; Sesia and Candès, 2020). The point of this section is the habit: **compute the conditional diagnostics before
trusting an "adaptive" interval.**
"""

# %%
fig, axes = P.panels(3)

# a: does sigma track the error?
ax = axes[0]
lim = float(np.ceil(1.1 * reliability.to_numpy().max()))            # 6 kcal/mol for the Δ-learning model
ax.plot([0, lim], [0, lim], color=P.GRAY, lw=0.8, ls="--")
ax.plot(reliability["mean sigma"], reliability["mean |error|"], marker="o", color=P.BLUE, lw=1.0)
ax.set(xlabel=r"predicted error size $\hat{\sigma}$ (kcal/mol)", ylabel="mean |error| (kcal/mol)", xlim=(0, lim), ylim=(0, lim))
if lim <= 6:
    ax.set_xticks(np.arange(lim + 1))
    ax.set_yticks(np.arange(lim + 1))
ax.text(0.96, 0.06, "tenths of the held-out\nreactions, ordered by " + r"$\hat{\sigma}$", transform=ax.transAxes, ha="right", va="bottom", fontsize=6.5)
P.panel_label(ax, "a", r"$\hat{\sigma}$ tracks the error")

# b: coverage by difficulty
ax = axes[1]
styles = {"split": (P.VERMILLION, "s", "-", "constant width"), "normalized": (P.BLUE, "o", "-", "normalized"), "CQR": (P.GREEN, "^", "--", "CQR")}
positions = np.arange(1, 6)
for method, (color, marker, line, label) in styles.items():
    ax.plot(positions, summary.loc[method, [f"sigma fifth {j}" for j in positions]].values, marker=marker, ls=line, color=color, label=label)
ax.axhline(1 - alpha, color=P.GRAY, lw=0.8, ls="--")
lowest = summary[[f"sigma fifth {j}" for j in positions]].to_numpy().min()
ax.set(xlabel=r"fifths of the test reactions by $\hat{\sigma}$", ylabel="coverage", ylim=(min(0.6, lowest - 0.05), 1.02), xlim=(0.7, 5.3))
ax.set_xticks(positions, ["easiest", "2", "3", "4", "hardest"])
ax.legend(loc="lower left", borderaxespad=0.3)
P.panel_label(ax, "b", "Coverage by difficulty")

# c: distribution of half-widths
ax = axes[2]
w_max = 12.0 if q_split < 5 else 4.0 * np.ceil(2.4 * q_split / 4)   # 12 kcal/mol for the Δ-learning model
grid = np.linspace(0, w_max, 400)
for method, half in (("normalized", (hi_norm - lo_norm) / 2), ("CQR", (hi_cqr - lo_cqr) / 2)):
    color, marker, line, label = styles[method]
    ax.plot(grid, [np.mean(half <= g) for g in grid], color=color, ls=line, label=label)
ax.axvline(q_split, color=P.VERMILLION, lw=1.2, ls=":")
ax.text(q_split + 0.02 * w_max, 0.42, f"constant width:\n±{q_split:.1f} for all", fontsize=6.5, va="center")
ax.set(xlabel="interval half-width (kcal/mol)", ylabel="cumulative fraction of test reactions", xlim=(0, w_max), ylim=(0, 1.02))
ax.set_xticks(np.linspace(0, w_max, 5))
ax.legend(loc="lower right", borderaxespad=0.3)
P.panel_label(ax, "c", "Most intervals shrink")
for ax in axes:
    ax.set_box_aspect(1)
P.savefig(fig, "nb3_adaptive_intervals")
plt.show()

# %% [markdown]
r"""
Panel **a**: the predicted error size follows the observed error over a factor of about nine. Panel **b**: coverage in
each fifth of the test reactions ordered by difficulty (the vertical axis starts at 0.6); a perfectly adaptive method
would be a horizontal line at 0.90. Panel **c**: cumulative distribution of the half-widths. With the normalized score
about two thirds of the reactions get a narrower interval than before, and the remaining third a wider one. The
vertical step at the left end of the normalized curve is the lower bound of $\hat\sigma$: the easiest reactions all
receive the same, smallest half-width.

## 6. A wide interval is information

An adaptive interval ranks the predictions by how much they should be trusted. The widest ones are worth a look:
"""

# %%
order = np.argsort(half_width, kind="stable")
listing = pd.DataFrame({
    "reaction": [f"{P.readable(rdb7.df.rsmi[i])} >> {P.readable(rdb7.df.psmi[i])}" for i in test[order]],
    "class": rdb7.rxn_class[test[order]],
    "predicted": mu[test[order]].round(1),
    "± half-width": half_width[order].round(1),
    "reference": y[test[order]].round(1),
})
print("five of the narrowest and the five widest normalized intervals in the test set (kcal/mol)")
with pd.option_context("display.max_colwidth", None):              # show the SMILES strings in full
    display(pd.concat([listing.head(5), listing.tail(5)]))
widest = order[-len(order) // 10:]
print(f"widest 10% of the intervals: mean |error| {abs_err[test[widest]].mean():.2f} kcal/mol;   "
      f"all others: {abs_err[test[order[:-len(order) // 10]]].mean():.2f} kcal/mol")

# %% [markdown]
r"""
In a screening campaign the reactions with the widest intervals are the natural candidates for a higher-level
calculation, and the narrow ones can be taken as they are.

## 7. A pitfall, and a limit

**Pitfall: a difficulty model trained on in-sample residuals.** Step 1 of Section 2 looks like a technicality. Skip
it (fit the difficulty model to the residuals of the training reactions themselves) and the result is still a valid
90% interval, because validity never depends on $\hat\sigma$. But the adaptation is damaged:
"""

# %%
sigma_in_model = P.make_model(max_iter=P.DIFFICULTY_TREES).fit(model.X[train], abs_res_in)
sigma_in = np.maximum(sigma_in_model.predict(model.X), P.DIFFICULTY_FLOOR * np.median(abs_res_in))

rows = []
for cal_b, test_b in P.resplits(model.hold, 1000, 200, seed=11):
    for label, s in (("out-of-fold residuals (Section 2)", sigma), ("in-sample residuals", sigma_in)):
        lo, hi, _ = normalized_conformal(y[cal_b], mu[cal_b], s[cal_b], mu[test_b], s[test_b], alpha)
        inside = (y[test_b] >= lo) & (y[test_b] <= hi)
        fifth = fifths(sigma[test_b])                                  # difficulty as estimated in Section 2, for both rows
        rows.append({"difficulty model trained on": label, "coverage": inside.mean(), "easiest fifth": inside[fifth == 0].mean(),
                     "hardest fifth": inside[fifth == 4].mean(),
                     "skeletal rearrangements": inside[rdb7.rxn_class[test_b] == "skeletal rearrangement"].mean()})
pitfall = pd.DataFrame(rows).groupby("difficulty model trained on", sort=False).mean()
display(pitfall.round(3))

# %% [markdown]
r"""
With in-sample residuals the easy reactions are over-covered and the hard ones under-covered again (about 95% and
85% for the easiest and the hardest fifth; the fifths are those of the out-of-fold $\hat\sigma$ in both rows), and the
skeletal rearrangements fall back to about 87%: the difficulty estimate has lost part of its meaning.

**Limit: conditional coverage cannot be guaranteed.** Normalized scores and CQR make the coverage *more* uniform, and
the diagnostics above let us see how much. They do not turn the marginal guarantee into a conditional one. It is a
theorem (Vovk, 2013; Lei and Wasserman, 2014; Foygel Barber et al., 2021) that no method can guarantee
$\Pr\{Y \in C(X) \mid X = x\} \ge 1-\alpha$ for every individual $x$ and every distribution, unless its intervals are
infinitely wide (precisely: at every $x$ that does not recur exactly in the population, the expected width must be
infinite). What *can* be guaranteed is coverage within a finite number of groups chosen in advance. That is the subject
of Notebook 4.

## Take-aways

* A constant-width interval can have the right overall coverage and still fail badly for the hardest cases.
* Changing the score changes the shape of the interval, not the guarantee: divide the residual by a difficulty
  estimate (normalized score), or start from a quantile band (CQR).
* A difficulty estimate can be any quantity computed from $x$ alone. If it is learned from residuals, use
  out-of-fold residuals.
* Always check conditional coverage: by chemical class, by difficulty, and by the method's own interval width.
* Adaptive intervals usually give better conditional coverage in practice, never a guarantee for individual reactions.

## Try it yourself

1. Repeat the comparison for the structure-only model (set `model = rdb7.models["structure"]` in the cell that loads
   the data, where `model` is defined). Is the gain from normalization larger or smaller, and why?
2. Replace $\hat\sigma(x)$ by something much simpler, for example the absolute size of the correction that the
   Δ-learning model predicts, `np.abs(mu - model.baseline)` plus a small constant (`model.baseline` is the ωB97X-D3
   barrier). How far does that get?
3. Set `P.DIFFICULTY_FLOOR = 0.05` in the cell where `model` is defined and run the notebook again. How well is the
   narrowest fifth of the normalized intervals covered now, and what does the lower bound protect against?

## References

* Papadopoulos, H.; Proedrou, K.; Vovk, V.; Gammerman, A. Inductive Confidence Machines for Regression. In *Machine
  Learning: ECML 2002*; Springer: Berlin, Heidelberg, 2002; pp 345–356. https://doi.org/10.1007/3-540-36755-1_29
* Papadopoulos, H.; Gammerman, A.; Vovk, V. Normalized Nonconformity Measures for Regression Conformal Prediction. In
  *Proceedings of the IASTED International Conference on Artificial Intelligence and Applications (AIA 2008)*;
  ACTA Press, 2008; pp 64–69.
* Lei, J.; G'Sell, M.; Rinaldo, A.; Tibshirani, R. J.; Wasserman, L. Distribution-Free Predictive Inference for
  Regression. *J. Am. Stat. Assoc.* **2018**, *113*, 1094–1111. https://doi.org/10.1080/01621459.2017.1307116
* Romano, Y.; Patterson, E.; Candès, E. J. Conformalized Quantile Regression. In *Advances in Neural Information
  Processing Systems 32*; Curran Associates, Inc., 2019; pp 3543–3553.
  https://proceedings.neurips.cc/paper/2019/hash/5103c3584b063c431bd1268e9b5e76fb-Abstract.html
* Sesia, M.; Candès, E. J. A Comparison of Some Conformal Quantile Regression Methods. *Stat* **2020**, *9*, e261.
  https://doi.org/10.1002/sta4.261
* Gneiting, T.; Raftery, A. E. Strictly Proper Scoring Rules, Prediction, and Estimation. *J. Am. Stat. Assoc.* **2007**,
  *102*, 359–378. https://doi.org/10.1198/016214506000001437
* Vovk, V. Conditional Validity of Inductive Conformal Predictors. *Mach. Learn.* **2013**, *92*, 349–376.
  https://doi.org/10.1007/s10994-013-5355-6 (conference version 2012)
* Lei, J.; Wasserman, L. Distribution-Free Prediction Bands for Non-Parametric Regression. *J. R. Stat. Soc. Ser. B:
  Stat. Methodol.* **2014**, *76*, 71–96. https://doi.org/10.1111/rssb.12021
* Foygel Barber, R.; Candès, E. J.; Ramdas, A.; Tibshirani, R. J. The Limits of Distribution-Free Conditional
  Predictive Inference. *Inf. Inference* **2021**, *10*, 455–482. https://doi.org/10.1093/imaiai/iaaa017
"""

# %%
results = {
    "q": {"split": q_split, "normalized": q_norm, "cqr": q_cqr},
    "sigma": {"oof_mae": abs_res_oof.mean(), "in_sample_mae": abs_res_in.mean(), "spearman": rho, "floor": floor,
              "share_at_floor": float(np.mean(sigma[hold] == floor)),
              "tenth_sigma": reliability["mean sigma"].values, "tenth_error": reliability["mean |error|"].values},
    "raw_quantile_band_coverage": raw_coverage,
    "class_coverage_single_split": {P.CLASS_SHORT[c]: by_class.loc[c, "coverage"] for c in P.CLASS_ORDER},
    "summary": {method: {k: v for k, v in row.items() if np.isfinite(v)} for method, row in summary.to_dict("index").items()},
    "pitfall_in_sample": {k: dict(v) for k, v in pitfall.to_dict("index").items()},
    "fraction_narrower_than_split": float(np.mean(half_width < q_split)),
    "widest_tenth_error": {"widest": float(abs_err[test[widest]].mean()), "others": float(abs_err[test[order[:-len(order) // 10]]].mean())},
}
P.save_results("nb3", results, rdb7)
P.check("nb3", results);
