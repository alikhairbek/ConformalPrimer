# NOTEBOOK: 02_split_conformal
# %% [markdown]
r"""
# Notebook 2 · Split conformal prediction from scratch

**Conformal Prediction for Physical Chemists: hands-on notebooks (2 of 6)**

Notebook 1 ended with a wish: an interval around every prediction whose coverage is *guaranteed*, whatever the model
and whatever the shape of its error distribution. Split conformal prediction (also called inductive conformal
prediction; Papadopoulos et al., 2002; Lei et al., 2018) delivers this with a held-out calibration set and one sorted
array. The general theory is laid out in the monograph of Vovk, Gammerman, and Shafer (2022), and Angelopoulos and Bates
(2023) give a broader introduction for machine-learning practitioners. In chemistry the method has mostly been used in
cheminformatics (Norinder et al., 2014; Svensson et al., 2018) and, more recently, for machine-learned interatomic
potentials (Hu et al., 2022).

**You will**

1. build 90% prediction intervals for reaction barriers in about fifteen lines of NumPy;
2. understand *why* the recipe works (a statement about ranks, not about distributions);
3. learn what the guarantee does **not** promise;
4. see that the coverage obtained from one calibration set is itself a random number, and how its spread depends
   on the size of the calibration set;
5. reproduce the two most common ways of getting it wrong;
6. see what to do when there are too few reference values to set a calibration set aside (CV+).

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

model = rdb7.models["delta"]        # the Δ-learning model is the running example of this notebook
mu = model.mu                       # its point predictions, for all reactions

# %% [markdown]
r"""
## 1. The recipe

We have a fitted model $\hat\mu$ and a set of reactions that the model has **not** seen during fitting, with known
reference barriers. Split this held-out set into a **calibration set** of size $n$ and a **test set** that plays the
role of future predictions. Choose the *miscoverage level* $\alpha$, the fraction of misses you are willing to accept:
$\alpha = 0.10$ for 90% intervals.

1. For each calibration reaction compute a *score* that says how wrong the model was (the literature calls it a
   conformity or nonconformity score). The simplest choice is the absolute residual, $s_i = |y_i - \hat\mu(x_i)|$.
2. Sort the scores and take the $\lceil (n+1)(1-\alpha)\rceil$-th smallest one. Call it $\hat q$.
3. For any new reaction $x$ report the interval $C(x) = [\hat\mu(x) - \hat q,\ \hat\mu(x) + \hat q]$.

That is all. The claim is

$$\Pr\{\,Y \in C(X)\,\} \;\ge\; 1 - \alpha ,$$

for **any** model and **any** error distribution, provided the calibration reactions and the new reaction are
*exchangeable* (roughly: drawn from the same population, in no particular order). The model was fitted on separate
data, so it can be good or bad; the guarantee holds either way.
"""

# %%
# a calibration set of 1000 reactions; the rest of the held-out reactions act as "future" test cases
rng = np.random.default_rng(P.SEED)
shuffled = rng.permutation(model.hold)
cal, test = shuffled[:1000], shuffled[1000:]

alpha = 0.10                                         # we want 90% intervals

# 1. scores on the calibration set
scores = np.abs(y[cal] - mu[cal])

# 2. the conformal quantile: the ceil((n + 1)(1 - alpha))-th smallest score
n = len(scores)
k = int(np.ceil((n + 1) * (1 - alpha)))
q_hat = np.sort(scores)[k - 1]

# 3. intervals for the test reactions
lower, upper = mu[test] - q_hat, mu[test] + q_hat

# 4. check them against the reference values (possible here because we kept the answers)
covered = (y[test] >= lower) & (y[test] <= upper)
print(f"n = {n}, k = {k}:  q_hat = {q_hat:.2f} kcal/mol")
print(f"every prediction becomes  mu ± {q_hat:.2f} kcal/mol;  {covered.mean():.1%} of {len(test)} test barriers lie inside")

# %% [markdown]
r"""
The same steps, written as two functions. They are also stored in `primer.py`, so that the later notebooks can simply
import them.
"""

# %%
def conformal_quantile(scores, alpha=0.10):
    """The ceil((n+1)(1-alpha))-th smallest calibration score; infinity if n is too small."""
    scores = np.asarray(scores, dtype=float)
    n = scores.size
    k = int(np.ceil((n + 1) * (1 - alpha) - 1e-9))     # the tiny offset guards against floating-point round-up
    if k > n:
        return np.inf
    return float(np.partition(scores, k - 1)[k - 1])


def split_conformal(y_cal, mu_cal, mu_test, alpha=0.10):
    """Absolute-residual split conformal interval: mu ± q."""
    q = conformal_quantile(np.abs(y_cal - mu_cal), alpha)
    return mu_test - q, mu_test + q, q


assert conformal_quantile(scores, alpha) == q_hat == P.conformal_quantile(scores, alpha)

# %% [markdown]
r"""
## 2. Why it works

Nothing in the recipe refers to a normal distribution, to the quality of the model, or to chemistry. The argument is
about **ranks**. Take the $n$ calibration scores and the score of one new reaction. Because $\hat q$ is the $k$-th
smallest calibration score, the new score is at most $\hat q$ exactly when its rank among all $n+1$ scores is at most
$k$. If the $n+1$ scores are exchangeable and no two are equal, the new score is equally likely to be the smallest,
the second smallest, ..., or the largest of them: each rank has probability $1/(n+1)$. The new score is therefore at
most $\hat q$ with probability $k/(n+1)$, and "the new score is at most $\hat q$" is the same event as "the reference
barrier lies in the interval". Choosing $k = \lceil (n+1)(1-\alpha) \rceil$ makes that probability at least $1-\alpha$.
(If some scores are equal, the probability can only be larger.)

The cell below checks the rank statement with numbers that have nothing to do with chemistry: squared Cauchy variates,
a distribution so heavy-tailed that it has no mean.
"""

# %%
rng = np.random.default_rng(1)
n_demo = 19
draws = rng.standard_cauchy(size=(200_000, n_demo + 1)) ** 2            # 19 "calibration" values + 1 "new" value per row
rank_of_new = 1 + (draws[:, :-1] < draws[:, -1:]).sum(axis=1)           # rank of the new value among all 20
frequencies = np.bincount(rank_of_new, minlength=n_demo + 2)[1:] / len(draws)
print("frequency of each rank (should all be 1/20 = 0.05):")
print(np.round(frequencies, 3))
print(f"new value <= largest of the 19 others: {np.mean(rank_of_new <= n_demo):.4f}   (theory 19/20 = 0.95)")

# %% [markdown]
r"""
With $n = 19$ and $\alpha = 0.05$ the recipe gives $k = 19$: the interval half-width is simply the largest of 19
calibration errors, and it covers 95% of new cases. This also shows the smallest calibration set that can support a
given level: $k \le n$ requires $n \ge (1-\alpha)/\alpha$, i.e., 9 reactions for 90%, 19 for 95%, and 99 for 99%
intervals. With fewer, the only interval with a guarantee is the infinite one.

## 3. The result
"""

# %%
LEVELS = np.array([0.50, 0.68, 0.80, 0.90, 0.95, 0.99])
abs_res = np.abs(y - mu)

# conformal coverage at several levels, averaged over 1000 random calibration/test splits
n_splits = 1000
cov_splits = np.zeros((n_splits, len(LEVELS)))
for i, (cal_b, test_b) in enumerate(P.resplits(model.hold, 1000, n_splits, seed=1)):
    for j, level in enumerate(LEVELS):
        cov_splits[i, j] = np.mean(abs_res[test_b] <= conformal_quantile(abs_res[cal_b], 1 - level))
conformal_cov = cov_splits.mean(axis=0)
conformal_se = cov_splits.std(axis=0, ddof=1) / np.sqrt(n_splits)       # Monte Carlo error of these averages

# Gaussian intervals from Notebook 1 for comparison
res_hold = (y - mu)[model.hold]
rmse = np.sqrt(np.mean(res_hold ** 2))
gauss_cov = np.array([np.mean(np.abs(res_hold) <= stats.norm.ppf(0.5 + level / 2) * rmse) for level in LEVELS])

comparison = pd.DataFrame({"conformal": conformal_cov, "Gaussian ± zσ": gauss_cov}, index=[f"{l:.0%}" for l in LEVELS])
comparison.index.name = "promised"
display(comparison.round(3))
print(f"The conformal column averages {n_splits} random splits; the standard error of these averages is "
      f"{conformal_se.min():.4f} to {conformal_se.max():.4f}.")

# %%
fig, axes = P.panels(3)

# a: calibration scores and the conformal quantile
ax = axes[0]
x_max = 12.0 if q_hat < 8 else 4.0 * np.ceil(1.5 * q_hat / 4)        # 12 kcal/mol unless the quantile lies farther out
ax.hist(np.clip(scores, 0, x_max - 0.01), bins=np.linspace(0, x_max, 25), color=P.BLUE, rwidth=0.82)
ax.axvline(q_hat, color=P.VERMILLION, lw=1.2, ls="--")
ax.text(q_hat + 0.025 * x_max, 0.92, f"$\\hat{{q}}$ = {q_hat:.2f}", transform=ax.get_xaxis_transform(), fontsize=7, va="top")
ax.text(q_hat + 0.025 * x_max, 0.80, f"{1 - alpha:.0%} of the scores\nlie to the left", transform=ax.get_xaxis_transform(),
        fontsize=6.5, va="top")
ax.set(xlabel=r"calibration score $|y - \hat{\mu}|$ (kcal/mol)", ylabel="calibration reactions", xlim=(0, x_max))
ticks = np.linspace(0, x_max, 5)
ax.set_xticks(ticks, [f"{t:g}" for t in ticks[:-1]] + [f"≥{x_max:g}"])
P.panel_label(ax, "a", f"{len(scores)} calibration scores")

# b: the interval as a band of constant half-width around the prediction
ax = axes[1]
res_test = y[test] - mu[test]
inside = np.abs(res_test) <= q_hat
ax.axhspan(-q_hat, q_hat, color=P.SKY, alpha=0.30, lw=0)
ax.scatter(mu[test][inside], res_test[inside], s=3, alpha=0.25, color=P.BLUE, linewidths=0, rasterized=True)
ax.scatter(mu[test][~inside], np.clip(res_test[~inside], -19.5, 19.5), s=7, alpha=0.8, color=P.VERMILLION,
           marker="^", linewidths=0, rasterized=True)
ax.text(0.97, 0.97, f"inside (dots): {inside.mean():.1%}\noutside (triangles): {(~inside).mean():.1%}",
        transform=ax.transAxes, ha="right", va="top", fontsize=6.5)
ax.set(xlabel=r"predicted barrier $\hat{\mu}$ (kcal/mol)", ylabel="reference − predicted (kcal/mol)", ylim=(-20, 20), xlim=(0, 200))
ax.set_xticks([0, 50, 100, 150, 200])
P.panel_label(ax, "b", f"$\\hat{{\\mu}}$ ± {q_hat:.1f} kcal/mol")

# c: observed / promised miss rate at every level
ax = axes[2]
positions = np.arange(len(LEVELS))
ax.plot(positions, (1 - gauss_cov) / (1 - LEVELS), marker="D", color=P.ORANGE, label="Gaussian ± zσ")
ax.plot(positions, (1 - conformal_cov) / (1 - LEVELS), marker="o", color=P.BLUE, label="split conformal")
ax.axhline(1, color=P.GRAY, lw=0.8, ls="--")
ax.set_yscale("log")
ax.set_yticks([0.5, 1, 2, 3], ["0.5", "1", "2", "3"])
ax.minorticks_off()
ax.set_ylim(0.4, 4.2)
ax.set_xticks(positions, [f"{100 * level:.0f}" for level in LEVELS])
ax.set(xlabel="promised coverage (%)", ylabel="observed / promised miss rate")
ax.legend(loc="upper left", borderaxespad=0.2)
P.panel_label(ax, "c", "Promise kept")
for ax in axes:
    ax.set_box_aspect(1)
P.savefig(fig, "nb2_split_conformal")
plt.show()

# %% [markdown]
r"""
Panel **a** shows the 1000 calibration scores and $\hat q$. Panel **b** shows the resulting band of constant half-width
around the predictions, with the test reactions that fall outside (errors beyond ±20 kcal/mol are drawn at the edge).
Panel **c** repeats the test of Notebook 1 for all levels, on the same logarithmic axis. The conformal intervals miss
as often as promised, from 50% to 99%, without any assumption about the error distribution. (In the table, the
conformal coverages differ from the promised levels by less than 0.001. Two small effects add up: the rounding in $k$
puts the expected coverage up to $1/(n+1) \approx 0.001$ above $1-\alpha$, and the average over 1000 random splits has
a standard error of a few ten-thousandths, printed below the table.)

## 4. What the guarantee does *not* say

The statement $\Pr\{Y \in C(X)\} \ge 1-\alpha$ is easy to overinterpret. Four limits matter in practice.

* **It is marginal, not conditional.** The probability is an average over all reactions of the population. It does
  not say that 90% of the *skeletal rearrangements* are covered, or 90% of the reactions with a predicted barrier
  below 40 kcal/mol. Notebook 1 showed that the errors of this model depend on the reaction class, so a single
  $\hat q$ will over-cover some classes and under-cover others (Notebooks 3 and 4).
* **It is an average over calibration sets.** A particular calibration set gives a particular $\hat q$, and the
  coverage *of that* $\hat q$ can be above or below $1-\alpha$. Section 5 quantifies this.
* **It needs exchangeability.** If tomorrow's reactions differ systematically from the calibration reactions, the
  guarantee is void (Notebook 5).
* **It refers to the reference level.** The interval covers the CCSD(T)-F12a value with probability $1-\alpha$. The
  error of the reference method itself is outside the guarantee.

## 5. Coverage is a random variable

Suppose a colleague repeats our work with a different random calibration set. Will they find the same coverage? No:
they will get a slightly different $\hat q$ and a different test coverage. For split conformal prediction the
distribution of this coverage is known exactly (Vovk, 2013). For independent draws from one population (and scores
without ties), the coverage that a calibration set of size $n$ achieves on an infinitely large test set follows a Beta
distribution,

$$\text{coverage} \sim \mathrm{Beta}(k,\; n + 1 - k), \qquad k = \lceil (n+1)(1-\alpha) \rceil ,$$

which again does not depend on the model or the data. Its mean is $k/(n+1)$, just above $1-\alpha$. On a finite test
set of $m$ reactions the number of covered reactions follows the corresponding Beta-binomial distribution (a binomial
distribution whose success probability is itself Beta-distributed). The next cell repeats the calibration 2000 times
for several $n$ and compares with this law.
"""

# %%
def coverage_experiment(score_pool, n_cal, alpha=0.10, n_rep=2000, seed=0):
    """Test coverage of split conformal intervals for `n_rep` random calibration sets of size `n_cal`."""
    rng = np.random.default_rng(seed)
    out = np.empty(n_rep)
    for b in range(n_rep):
        perm = rng.permutation(score_pool)
        out[b] = np.mean(perm[n_cal:] <= conformal_quantile(perm[:n_cal], alpha))
    return out


def finite_test_law(n_cal, n_test, alpha=0.10):
    """Exact distribution of the number of covered test points (Beta-binomial)."""
    k = int(np.ceil((n_cal + 1) * (1 - alpha) - 1e-9))
    return stats.betabinom(n_test, k, n_cal + 1 - k)


pool = abs_res[model.hold]
sizes = [20, 50, 100, 200, 500, 1000, 2000]
experiments = {n_cal: coverage_experiment(pool, n_cal) for n_cal in sizes}

rows = []
for n_cal, cov in experiments.items():
    n_test = len(pool) - n_cal
    law = finite_test_law(n_cal, n_test)
    rows.append({"n": n_cal, "mean": cov.mean(), "2.5%": np.quantile(cov, 0.025), "97.5%": np.quantile(cov, 0.975),
                 "theory mean": law.mean() / n_test, "theory 2.5%": law.ppf(0.025) / n_test, "theory 97.5%": law.ppf(0.975) / n_test})
coverage_table = pd.DataFrame(rows).set_index("n")
print("Test coverage of nominal 90% intervals over 2000 random calibration sets")
display(coverage_table.round(3))

# %% [markdown]
r"""
The agreement is as close as 2000 repetitions allow, and the message is practical: **with 50 calibration reactions, a
"90%" interval has an actual coverage between about 81% and 97% for 95% of all calibration sets; with 1000 it is pinned
down to about ±2 percentage points.** A coverage of 87% obtained with a small calibration set is therefore not evidence
that conformal prediction has failed, and 93% is not evidence that it works unusually well. (A small *test* set adds
ordinary binomial noise on top.)

The same law answers a planning question: how many reference calculations should be set aside for calibration? (The
law assumes calibration reactions that are unrelated to one another. Notebook 5 shows that reactions which come in
clusters, such as several reactions of one reactant, are worth fewer.)
"""

# %%
def calibration_size_needed(tolerance, alpha=0.10, confidence=0.95, n_max=20000):
    """Smallest n such that, for this and every larger calibration set (up to n_max), the coverage lies
    within ±tolerance of 1 - alpha with the given probability."""
    sizes = np.arange(10, n_max + 1)
    k = np.ceil((sizes + 1) * (1 - alpha) - 1e-9)                    # the rank k for every n at once
    finite = k <= sizes                                              # otherwise the interval is infinite
    inside = np.zeros(len(sizes))                                    # probability that the coverage is within the tolerance
    inside[finite] = (stats.beta.cdf(1 - alpha + tolerance, k[finite], (sizes + 1 - k)[finite])
                      - stats.beta.cdf(1 - alpha - tolerance, k[finite], (sizes + 1 - k)[finite]))
    enough = inside >= confidence
    too_small = np.flatnonzero(~enough)      # the rounding in k makes the probability wiggle, so take the last failure
    if too_small.size and too_small[-1] == len(sizes) - 1:
        raise ValueError(f"the tolerance is not reached with calibration sets of up to n_max = {n_max} reactions")
    return int(sizes[too_small[-1] + 1]) if too_small.size else int(sizes[0])

needed = pd.DataFrame({"calibration reactions needed": {f"±{100 * t:.0f}": calibration_size_needed(t) for t in (0.05, 0.03, 0.02, 0.01)}})
needed.index.name = "coverage within (percentage points, 95% of the time)"
display(needed)
print("smallest calibration set with a finite interval:",
      {f"{level:.0%}": int(np.ceil(level / (1 - level) - 1e-9)) for level in (0.90, 0.95, 0.99)})

# %%
fig, axes = P.panels(2, height=2.6, gridspec_kw={"width_ratios": [1, 1.15]})

# a: two calibration-set sizes, experiment versus theory
ax = axes[0]
for n_cal, color, label_x in ((50, P.ORANGE, 0.80), (1000, P.BLUE, 0.928)):
    cov = experiments[n_cal]
    n_test = len(pool) - n_cal
    ax.hist(cov, bins=np.linspace(0.74, 1.0, 53), density=True, color=color, alpha=0.35, lw=0)
    law = finite_test_law(n_cal, n_test)
    grid = np.arange(int(0.74 * n_test), n_test + 1)
    ax.plot(grid / n_test, law.pmf(grid) * n_test, color=color, lw=1.4)
    ax.text(label_x, 0.62 if n_cal == 1000 else 0.2, f"$n$ = {n_cal}", transform=ax.get_xaxis_transform(), fontsize=7, ha="left")
ax.axvline(0.90, color=P.GRAY, lw=0.8, ls="--")
ax.set(xlabel="coverage on the test set", ylabel="probability density", xlim=(0.74, 1.0))
P.panel_label(ax, "a", "2000 calibration sets (bars) and theory (lines)")

# b: the 95% range of the coverage as a function of n
ax = axes[1]
grid_n = np.unique(np.round(np.logspace(1, np.log10(5000), 200)).astype(int))
laws = [P.coverage_law(n_cal) for n_cal in grid_n]
ax.fill_between(grid_n, [l.ppf(0.025) for l in laws], [l.ppf(0.975) for l in laws], color=P.SKY, alpha=0.35, lw=0,
                label="95% range (Beta law)")
ax.plot(grid_n, [l.mean() for l in laws], color=P.BLUE, lw=1.2, label="mean (Beta law)")
ax.plot(sizes, coverage_table["2.5%"], ls="none", marker="v", color=P.VERMILLION, label="2.5% and 97.5% (experiment)")
ax.plot(sizes, coverage_table["97.5%"], ls="none", marker="^", color=P.VERMILLION)
ax.axhline(0.90, color=P.GRAY, lw=0.8, ls="--")
ax.set_xscale("log")
ax.set(xlabel="calibration reactions $n$", ylabel="coverage of a nominal 90% interval", xlim=(10, 5000), ylim=(0.68, 1.0))
ax.set_xticks([10, 100, 1000], ["10", "100", "1000"])
ax.legend(loc="lower right")
P.panel_label(ax, "b", "How much can one calibration set be off?")
P.savefig(fig, "nb2_coverage_distribution")
plt.show()

# %% [markdown]
r"""
Panel **a**: the coverage found in 2000 repetitions (bars) and the exact law (lines) for two calibration-set sizes.
Panel **b**: the range that contains the coverage of 95% of all calibration sets, as a function of $n$. The band is the
Beta law, which describes an infinitely large test set; its small teeth come from the rounding in $k$. The triangles
are the experiment, whose finite test sets add a little scatter of their own (visible at $n = 2000$). Note that the
vertical axis starts at 0.68.

## 6. Two ways of getting it wrong

**Pitfall 1: calibrating on the training set.** It is tempting to save reference calculations by computing the scores
on the reactions the model was fitted to. But a flexible model has partly memorized its training set, so those
residuals are too small, and so is $\hat q$.
"""

# %%
q_from_training = conformal_quantile(abs_res[model.train], alpha)
q_from_holdout = conformal_quantile(abs_res[cal], alpha)
pitfall_training = np.mean(abs_res[test] <= q_from_training)
print(f"q from {len(model.train)} TRAINING residuals : {q_from_training:.2f} kcal/mol -> test coverage {pitfall_training:.1%}")
print(f"q from {len(cal)} held-out residuals : {q_from_holdout:.2f} kcal/mol -> test coverage {np.mean(abs_res[test] <= q_from_holdout):.1%}")

# %% [markdown]
r"""
The "90%" interval now covers only about three reactions in four. The same happens, less visibly, whenever the
calibration set is not independent of the fitted model: when it was used for early stopping or for choosing
hyperparameters, or when it contains near-duplicates of training points (conformers of a training molecule, the reverse
of a training reaction).

**Pitfall 2: forgetting the finite-sample correction.** The conformal quantile is the $\lceil (n+1)(1-\alpha)\rceil$-th
smallest score, not the empirical 90th percentile. For large $n$ the difference is negligible; for small $n$ it is
the difference between a guarantee and a shortfall.
"""

# %%
def coverage_with(quantile_rule, n_cal, n_rep=4000, seed=3):
    rng = np.random.default_rng(seed)
    out = np.empty(n_rep)
    for b in range(n_rep):
        perm = rng.permutation(pool)
        out[b] = np.mean(perm[n_cal:] <= quantile_rule(perm[:n_cal]))
    return out.mean()

rows = []
for n_cal in (20, 50, 200, 1000):
    rows.append({"n": n_cal,
                 "np.quantile(scores, 0.90)": coverage_with(lambda s: np.quantile(s, 1 - alpha), n_cal),
                 "conformal quantile": coverage_with(lambda s: conformal_quantile(s, alpha), n_cal)})
pitfall_quantile = pd.DataFrame(rows).set_index("n")
print("Mean test coverage of nominal 90% intervals")
display(pitfall_quantile.round(3))

# %% [markdown]
r"""
## 7. Validity does not depend on the model; sharpness does

An interval method is called *valid* if it keeps its coverage promise, and *sharp* (or efficient) if its intervals are
narrow. The recipe did not ask how good the predictor is, or even what it is. To see what that means, apply it to five
predictors of very different quality: a constant (the mean barrier of the training set, i.e., no model at all), the
structure-only model, the ωB97X-D3 barrier taken as it is (a quantum-chemical method, no machine learning), the same
barrier after subtracting its mean error on the training set, and the Δ-learning model.
"""

# %%
barrier_dft = rdb7.df["barrier_dft"].to_numpy()
dft_offset = (barrier_dft - y)[rdb7.train].mean()            # mean error of the DFT barriers, from the training half
predictors = {"constant (training mean)": np.full(rdb7.n, y[rdb7.train].mean()),
              rdb7.models["structure"].label: rdb7.models["structure"].mu,
              "ωB97X-D3 barrier, uncorrected": barrier_dft,
              "ωB97X-D3 barrier, mean error removed": barrier_dft - dft_offset,
              rdb7.models["delta"].label: rdb7.models["delta"].mu}
rows = []
for label, prediction in predictors.items():
    a = np.abs(y - prediction)
    qs, covs = [], []
    for cal_b, test_b in P.resplits(rdb7.hold, 1000, 500, seed=2):
        q = conformal_quantile(a[cal_b], alpha)
        qs.append(q)
        covs.append(np.mean(a[test_b] <= q))
    rows.append({"predictor": label, "MAE": a[rdb7.hold].mean(), "half-width (kcal/mol)": np.mean(qs), "coverage": np.mean(covs),
                 "factor in k at 1000 K": float(P.rate_factor(np.mean(qs), 1000.0))})
sharpness = pd.DataFrame(rows).set_index("predictor")
shown = sharpness.round({"MAE": 2, "half-width (kcal/mol)": 1, "coverage": 3})
shown["factor in k at 1000 K"] = [f"{value:.3g}" for value in sharpness["factor in k at 1000 K"]]
display(shown)
print(f"ωB97X-D3 minus CCSD(T)-F12a barrier on the training half: mean {dft_offset:+.2f} kcal/mol, "
      f"positive for {np.mean((barrier_dft - y)[rdb7.train] > 0):.0%} of the reactions")

# %% [markdown]
r"""
All five rows have 90% coverage. Even "predict 80 kcal/mol for every reaction" comes with a perfectly valid interval;
it is just uselessly wide. The third row needs no machine learning: for reactions like those in RDB7, an ωB97X-D3
barrier lies within the stated half-width of the CCSD(T)-F12a barrier in 90% of the cases. The fourth row shows that a
symmetric interval around a biased predictor is wasteful: the ωB97X-D3 barrier lies above the CCSD(T)-F12a barrier for
almost nine reactions in ten, and removing the mean difference, the simplest possible correction, already shrinks the
interval by a third. **Conformal prediction does not make a predictor better. It reports how large the errors of the
given predictor are**, in the unit you care about: here, how many kcal/mol (or how many orders of magnitude in the rate
constant) you should allow for.

## 8. When reference data are scarce: CV+

Split conformal prediction spends part of the reference data on calibration. With thousands of reactions that is
painless. With a hundred it hurts twice: the model is fitted on fewer reactions, and the small calibration set gives
an unstable $\hat q$ (Section 5). Calibrating on the training residuals instead is not a way out (Pitfall 1).

**CV+** (Barber, Candès, Ramdas, and Tibshirani, 2021) uses every labeled reaction for both purposes, by
cross-validation. Let $N$ be the number of labeled reactions.

1. Split the $N$ reactions into $K$ folds and fit $K$ models, each *without* one fold.
2. For every labeled reaction $i$, record the residual $R_i$ of the model that did not see it (an *out-of-fold*
   residual).
3. For a new reaction $x$, every labeled reaction "votes" for an interval: with $\hat\mu_{-i}$ the model fitted
   without the fold of reaction $i$, the votes are $\hat\mu_{-i}(x) - R_i$ for the lower end and
   $\hat\mu_{-i}(x) + R_i$ for the upper end.
4. The interval runs from the $\lfloor (N+1)\alpha\rfloor$-th smallest lower vote to the
   $\lceil (N+1)(1-\alpha)\rceil$-th smallest upper vote.

With $K = N$ (leave-one-out) this is the **jackknife+**. The price of using the data twice is a weaker worst-case
guarantee. For the jackknife+ the coverage is at least $1-2\alpha$ instead of $1-\alpha$; for CV+ with $K$ folds of
equal size it is at least

$$1 - 2\alpha - \min\left\{\frac{2\,(1-1/K)}{N/K+1},\ \frac{1-K/N}{K+1}\right\} ,$$

a little lower still (Barber et al., 2021, Theorems 1 and 4; the fitting procedure must not depend on the order of the
data). These bounds are worst cases, approached only when the fitted model changes violently from one fold to the
next. In practice the coverage is close to $1-\alpha$, as the experiment below shows.
"""

# %%
def cv_plus(y, oof_pred, fold, test_pred, alpha=0.10):
    """CV+ interval. `test_pred[k]` holds the test predictions of the model fitted without fold k."""
    n = len(y)
    residual = np.abs(y - oof_pred)                     # R_i: out-of-fold residuals
    k_hi = int(np.ceil((n + 1) * (1 - alpha) - 1e-9))
    k_lo = int(np.floor((n + 1) * alpha + 1e-9))
    if k_hi > n or k_lo < 1:                            # too few labeled points for this level
        return np.full(test_pred.shape[1], -np.inf), np.full(test_pred.shape[1], np.inf)
    low = test_pred[fold] - residual[:, None]           # one row of "votes" per labeled reaction
    high = test_pred[fold] + residual[:, None]
    return np.partition(low, k_lo - 1, axis=0)[k_lo - 1], np.partition(high, k_hi - 1, axis=0)[k_hi - 1]


# the same function is stored in primer.py; check on arbitrary numbers that the two agree
_rng = np.random.default_rng(0)
_args = (_rng.normal(size=40), _rng.normal(size=40), _rng.integers(0, 4, size=40), _rng.normal(size=(4, 7)))
assert all(np.array_equal(a, b) for a, b in zip(cv_plus(*_args), P.cv_plus(*_args)))

# %% [markdown]
r"""
For the experiment we pretend that only $N$ reference barriers exist. In every repetition, $N$ labeled reactions and
2000 test reactions are drawn at random, without overlap, from the held-out half of the data, so the labeled and the
test reactions are exchangeable by construction. Boosted trees are a poor choice for a few dozen data points, so this
section uses a model that suits the situation: ridge regression for the Δ-learning correction. (Conformal prediction
does not care; that is the point of "any model".) Three ways of using the $N$ reactions are compared:

* **split 50/50**: fit on one half, calibrate on the other (Section 1);
* **training residuals**: fit on all $N$ and calibrate on the same $N$ (Pitfall 1);
* **CV+** with $K = 10$ (and the jackknife+ for the smallest $N$).
"""

# %%
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

X, target, baseline = model.X.astype(float), model.target, model.baseline      # features and target of the Δ-learning model


def small_model():
    """Ridge regression on standardized features. (The argument `alpha` of Ridge is the strength of its penalty and
    has nothing to do with our miscoverage level; it would normally be chosen by cross-validation.)"""
    return make_pipeline(StandardScaler(), Ridge(alpha=30.0))


def predict_barrier(fitted, rows):
    return fitted.predict(X[rows]) + baseline[rows]


def cv_plus_intervals(labeled, test_rows, n_folds=10, seed=0):
    """Fit the fold models and return the CV+ interval for every test reaction."""
    fold, oof = np.empty(len(labeled), dtype=int), np.empty(len(labeled))
    test_pred = np.empty((n_folds, len(test_rows)))
    for k_fold, (fit_idx, out_idx) in enumerate(KFold(n_folds, shuffle=True, random_state=seed).split(labeled)):
        fold_model = small_model().fit(X[labeled[fit_idx]], target[labeled[fit_idx]])
        oof[out_idx] = predict_barrier(fold_model, labeled[out_idx])
        fold[out_idx] = k_fold
        test_pred[k_fold] = predict_barrier(fold_model, test_rows)
    return cv_plus(y[labeled], oof, fold, test_pred, alpha)


rows = []
for N in (50, 100, 300, 1000):
    for rep in range(100 if N <= 300 else 40):
        shuffled_pool = np.random.default_rng(1000 * N + rep).permutation(rdb7.hold)
        labeled, test_small = shuffled_pool[:N], shuffled_pool[N:N + 2000]      # N labeled reactions, 2000 others to predict
        intervals = {}
        # split 50/50
        fit_half, cal_half = labeled[: N // 2], labeled[N // 2:]
        half_model = small_model().fit(X[fit_half], target[fit_half])
        lo, hi, _ = split_conformal(y[cal_half], predict_barrier(half_model, cal_half), predict_barrier(half_model, test_small), alpha)
        intervals["split 50/50"] = (lo, hi)
        # training residuals (wrong)
        full_model = small_model().fit(X[labeled], target[labeled])
        lo, hi, _ = split_conformal(y[labeled], predict_barrier(full_model, labeled), predict_barrier(full_model, test_small), alpha)
        intervals["training residuals (wrong)"] = (lo, hi)
        # CV+ and, for the smallest N, the jackknife+
        intervals["CV+ (K = 10)"] = cv_plus_intervals(labeled, test_small, 10, seed=rep)
        if N == 50 and rep < 40:
            intervals["jackknife+ (K = N)"] = cv_plus_intervals(labeled, test_small, N, seed=rep)
        for method, (lo, hi) in intervals.items():
            rows.append({"N": N, "method": method, "coverage": np.mean((y[test_small] >= lo) & (y[test_small] <= hi)),
                         "half_width": np.mean(hi - lo) / 2})
small_data = pd.DataFrame(rows).groupby(["N", "method"], sort=False).agg(
    coverage=("coverage", "mean"), coverage_sd=("coverage", "std"), half_width=("half_width", "mean"), half_width_sd=("half_width", "std"))
print("Mean and standard deviation over the repetitions (different labeled sets); half-widths in kcal/mol")
display(small_data.rename(columns={"coverage": "mean coverage", "coverage_sd": "sd of coverage",
                                   "half_width": "mean half-width", "half_width_sd": "sd of half-width"}).round(3))

# %% [markdown]
r"""
Three things to read off. Calibrating on the training residuals under-covers, and the more so the fewer reactions
there are. Split conformal prediction and CV+ both keep the promise on average. (The split intervals lie above 90% for
the smallest $N$ because of the rounding in $k$: with 25 calibration reactions, $k = 24$ and the expected coverage is
$24/26 \approx 92\%$.) But with 50 or 100 reactions CV+ gives **narrower intervals that vary much less from one data
set to the next**: its model is fitted on 90% of the data instead of 50%, and all $N$ residuals are used instead of
$N/2$. From about a thousand reactions on, the two are practically equivalent and the simpler split recipe is the
natural choice. The cost of CV+ is $K$ model fits, and all $K$ models have to be kept for prediction.

Note also what the half-widths say about the value of data: with 100 reference barriers the ridge model gives 90%
intervals of about ±6 kcal/mol, with 1000 about ±5, and the boosted trees fitted to 6000 reactions in Section 1
reached ±3.8.

## Take-aways

* Split conformal prediction = held-out residuals + one order statistic (the $k$-th smallest score). No distributional
  assumption, no retraining, any predictor.
* The guarantee is marginal, holds on average over calibration sets, requires exchangeability, and refers to the
  reference level.
* The coverage obtained from one calibration set follows a known Beta law. A few hundred calibration reactions pin it
  down to a few percentage points; fewer than $(1-\alpha)/\alpha$ cannot support the level at all.
* Never calibrate on data the model has seen, and use the $(n+1)$ rule.
* The width of the interval is the honest price of the predictor's errors.
* With little data, use CV+ or the jackknife+ instead of setting a calibration set aside.

## Try it yourself

1. Change `alpha` to 0.01. How large is $\hat q$ for the Δ-learning model, and how does the Beta law for $n = 1000$
   change (compare `P.coverage_law(1000, 0.01)` with `P.coverage_law(1000, 0.10)`)?
2. Repeat Section 5 with the structure-only model. Does the coverage distribution change? Why not?
3. Call `conformal_quantile` with only 9 calibration scores at 90%: what is $k$, and what does the interval look like?
   What happens with 8?
4. In Section 8, replace the ridge model by `P.make_model(max_iter=100, min_samples_leaf=5)` for $N = 100$. What happens
   to the coverage of the "training residuals" intervals, and why?

## References

* Vovk, V.; Gammerman, A.; Shafer, G. *Algorithmic Learning in a Random World*, 2nd ed.; Springer: Cham, 2022.
  https://doi.org/10.1007/978-3-031-06649-8
* Papadopoulos, H.; Proedrou, K.; Vovk, V.; Gammerman, A. Inductive Confidence Machines for Regression. In *Machine
  Learning: ECML 2002*; Springer: Berlin, Heidelberg, 2002; pp 345–356. https://doi.org/10.1007/3-540-36755-1_29
* Lei, J.; G'Sell, M.; Rinaldo, A.; Tibshirani, R. J.; Wasserman, L. Distribution-Free Predictive Inference for
  Regression. *J. Am. Stat. Assoc.* **2018**, *113*, 1094–1111. https://doi.org/10.1080/01621459.2017.1307116
* Vovk, V. Conditional Validity of Inductive Conformal Predictors. *Mach. Learn.* **2013**, *92*, 349–376.
  https://doi.org/10.1007/s10994-013-5355-6 (the distribution of the coverage obtained from one calibration set)
* Angelopoulos, A. N.; Bates, S. Conformal Prediction: A Gentle Introduction. *Found. Trends Mach. Learn.* **2023**,
  *16*, 494–591. https://doi.org/10.1561/2200000101
* Barber, R. F.; Candès, E. J.; Ramdas, A.; Tibshirani, R. J. Predictive Inference with the Jackknife+. *Ann. Stat.*
  **2021**, *49*, 486–507. https://doi.org/10.1214/20-AOS1965
* Norinder, U.; Carlsson, L.; Boyer, S.; Eklund, M. Introducing Conformal Prediction in Predictive Modeling. A
  Transparent and Flexible Alternative to Applicability Domain Determination. *J. Chem. Inf. Model.* **2014**, *54*,
  1596–1603. https://doi.org/10.1021/ci5001168
* Svensson, F.; Aniceto, N.; Norinder, U.; Cortés-Ciriano, I.; Spjuth, O.; Carlsson, L.; Bender, A. Conformal Regression
  for Quantitative Structure–Activity Relationship Modeling—Quantifying Prediction Uncertainty. *J. Chem. Inf. Model.*
  **2018**, *58*, 1132–1140. https://doi.org/10.1021/acs.jcim.8b00054
* Hu, Y.; Musielewicz, J.; Ulissi, Z. W.; Medford, A. J. Robust and Scalable Uncertainty Estimation with Conformal
  Prediction for Machine-Learned Interatomic Potentials. *Mach. Learn.: Sci. Technol.* **2022**, *3*, 045028.
  https://doi.org/10.1088/2632-2153/aca7b1
"""

# %%
results = {
    "single_split": {"n_cal": int(n), "k": int(k), "q_hat": q_hat, "coverage": float(covered.mean()), "n_test": len(test)},
    "levels": {f"{level:.2f}": {"conformal": c, "conformal_se": se, "gaussian": g}
               for level, c, se, g in zip(LEVELS, conformal_cov, conformal_se, gauss_cov)},
    "coverage_vs_n": {str(n_cal): {"mean": r["mean"], "lo": r["2.5%"], "hi": r["97.5%"], "theory_mean": r["theory mean"],
                                   "theory_lo": r["theory 2.5%"], "theory_hi": r["theory 97.5%"]}
                      for n_cal, r in coverage_table.iterrows()},
    "n_needed": {k_.replace("±", "pm"): int(v) for k_, v in needed.iloc[:, 0].items()},
    "pitfall_training": {"q": q_from_training, "coverage": float(pitfall_training)},
    "pitfall_quantile": {str(n_cal): {"plain": r.iloc[0], "conformal": r.iloc[1]} for n_cal, r in pitfall_quantile.iterrows()},
    "sharpness": {name: {"mae": r["MAE"], "q": r["half-width (kcal/mol)"], "coverage": r["coverage"]}
                  for name, r in zip(("constant", "structure", "dft", "dft_shifted", "delta"), sharpness.to_dict("records"))},
    "dft_offset": {"mean": float(dft_offset), "share_positive": float(np.mean((barrier_dft - y)[rdb7.train] > 0))},
    "small_data": {f"N={N} | {method}": dict(r) for (N, method), r in small_data.iterrows()},
}
P.save_results("nb2", results, rdb7)
P.check("nb2", results);
