# NOTEBOOK: 06_selection
# %% [markdown]
r"""
# Notebook 6 · From intervals to decisions: selecting reactions with a guaranteed error rate

**Conformal Prediction for Physical Chemists: hands-on notebooks (6 of 6)**

A prediction interval answers the question of where the barrier of *this* reaction lies. Screening asks a different
question: *which* of these several thousand reactions have a barrier below some threshold? The output is a
shortlist, and the natural quality measure of a shortlist is the fraction of its entries that should not be on it.
This notebook shows how the calibration set of Notebook 2 can be used to produce shortlists whose expected fraction
of false entries is at most a level $q$ chosen by the user.

**You will**

1. see three intuitive selection rules and why none of them controls the fraction of false discoveries;
2. turn a calibration set into *conformal p-values*, one for each candidate reaction;
3. apply the Benjamini–Hochberg procedure and obtain a shortlist with a false discovery rate of at most $q$;
4. learn what determines how many true hits are found, and how large the calibration set has to be.

A word on notation: $q$ is the usual symbol for the target false discovery rate. It has nothing to do with the
conformal quantile $\hat q$ of the earlier notebooks. The letter $k$ stands for a rate constant in Section 1; the rank of
a candidate in the Benjamini–Hochberg step is called $r$.

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
conformal_quantile = P.conformal_quantile
sigma_of = {name: P.fit_difficulty(rdb7, name) for name in rdb7.models}      # difficulty estimates of Notebook 3

# %% [markdown]
r"""
## 1. The screening task

We call a reaction a **hit** if its reference barrier is below a threshold $c$, and take $c = 50$ kcal/mol as an
illustration. With the Eyring estimate $k \approx (k_\mathrm{B}T/h)\,e^{-E/RT}$ (with the barrier height $E$ in place
of the Gibbs energy of activation, an approximation that is good to within one or two orders of magnitude for the rate
constant), a unimolecular reaction with a 50 kcal/mol barrier has a half-life of a few milliseconds at 1000 K, a
typical shock-tube time scale. The threshold is a choice: at higher temperatures or longer residence times a higher one
is appropriate (Exercise 1). In RDB7, reactions below 50 kcal/mol are rare.
"""

# %%
c = 50.0                                   # kcal/mol: a "hit" has a reference barrier below c

rate = P.KB_OVER_H * 1000 * np.exp(-c / (P.R_KCAL * 1000))
print(f"Eyring estimate for a {c:.0f} kcal/mol barrier at 1000 K: k = {rate:.3g} s⁻¹, half-life {1000 * np.log(2) / rate:.1f} ms")
print(f"hits among the held-out reactions: {np.mean(y[rdb7.hold] < c):.1%}")

# one calibration/test split, as in the earlier notebooks; the test reactions play the role of the candidates
shuffled = np.random.default_rng(P.SEED).permutation(rdb7.hold)
cal, test = shuffled[:1000], shuffled[1000:]
is_hit = y[test] < c
print(f"candidates: {len(test)}, of which {is_hit.sum()} are hits (unknown to the person who screens)")

# %% [markdown]
r"""
Two numbers describe a shortlist:

* the **false discovery proportion** (FDP): the fraction of selected reactions that are not hits;
* the **power**: the fraction of all hits that were selected.

## 2. Three tempting rules

**Rule A: trust the point prediction.** Select every reaction whose *predicted* barrier is below $c$.

**Rule B: use the interval.** Select a reaction if its whole 90% conformal interval lies below $c$.

**Rule C: test every reaction at the 10% level.** For each candidate, ask how surprising its prediction would be if it
were not a hit, and select it if the answer is "would happen in fewer than 10% of such cases". (The p-value used for
this is constructed in Section 3.)
"""

# %%
def report(selected, is_hit):
    """Size of a shortlist, its false discovery proportion and its power."""
    n_sel = int(selected.sum())
    false = int((selected & ~is_hit).sum())
    found = int((selected & is_hit).sum())
    return {"selected": n_sel, "false discoveries": false, "FDP": false / max(n_sel, 1),
            "hits found": found, "power": found / max(int(is_hit.sum()), 1)}


def in_words(r):
    return (f"{r['selected']} selected, {r['false discoveries']} of them false (FDP {r['FDP']:.1%}); "
            f"{r['hits found']} hits found (power {r['power']:.1%})")


rows = []
for name, mod in rdb7.models.items():
    mu = mod.mu
    q_hat = conformal_quantile(np.abs(y - mu)[cal], 0.10)                               # half-width of the 90% interval
    p_values = P.conformal_pvalues((mu - c)[cal], y[cal] >= c, (mu - c)[test])          # explained in Section 3
    rules = {"A: predicted barrier below c": mu[test] < c,
             "B: 90% interval entirely below c": mu[test] + q_hat < c,
             "C: p-value below 0.10 for each reaction": p_values <= 0.10}
    for rule, selected in rules.items():
        rows.append({"model": mod.label, "rule": rule, **report(selected, is_hit)})
tempting = pd.DataFrame(rows).set_index(["model", "rule"])
display(tempting.round(3))

# %% [markdown]
r"""
* **Rule A** gives no control. With the accurate Δ-learning model only a few percent of the selected reactions are false;
  with the structure-only model more than one in five is. The user of rule A cannot know which situation applies
  without the reference values that the screening was meant to avoid.
* **Rule B** makes hardly any false selection here, but that is a property of these data, not of the rule. The interval
  guarantee is a statement about single reactions, averaged over all candidates: a candidate is a non-hit *and* has its
  whole interval below $c$ with a probability of at most 10% (its barrier then lies above the interval). The guarantee
  therefore limits the number of false selections only to about one tenth of all candidates, a bound that does not
  shrink with the length of the list: a short list may consist entirely of non-hits. When hits are rare and errors
  heavy-tailed, most of a rule-B list can be false. The rule also answers the wrong question: it loses about 30% of the
  hits with the Δ-learning model and practically all of them with the structure-only model, whose intervals are so wide
  that hardly any reaction passes.
* **Rule C** looks statistically respectable and performs worst. A test at the 10% level lets about one non-hit in ten
  pass, and more than 90% of the candidates are non-hits: several hundred false discoveries swamp the true ones. This
  is the multiple-testing problem, well known from genomics and from virtual screening.

What we need is a rule that controls the error *of the list*: the **false discovery rate**

$$ \mathrm{FDR} = \mathbb{E}\!\left[\frac{\text{number of selected non-hits}}{\max(\text{number selected},\,1)}\right] \le q . $$

## 3. Conformal p-values

First, a measure of evidence for each candidate. Take any **score** computed from the candidate alone, such that a
smaller score looks more promising; the obvious choice is $\hat\mu(x) - c$, the predicted barrier measured from the
threshold (negative when the prediction lies below it). On its own the score has no scale: is "predicted 4 kcal/mol
below the threshold" convincing? The calibration set provides the scale. Among the calibration reactions, look only at
the **non-hits** (reference barrier at or above $c$), and count how many of them look at least as promising as
candidate $j$:

$$ p_j = \frac{1 + \#\{\,i \in \text{calibration}:\ y_i \ge c \ \text{ and } \ s_i \le s_j \,\}}{n + 1} . $$

Here $n$ is the size of the whole calibration set, hits included. A small $p_j$ means: hardly any reaction that we
*know* to be a non-hit ever looked this good.

Why is this a p-value? By the rank argument of Notebook 2. Put candidate $j$ together with the $n$ calibration
reactions: $n+1$ exchangeable reactions, any of which could equally well have been the candidate. Line up the non-hits
of this group by score. Candidate $j$ is a non-hit *and* receives $p_j \le t$ only if it is among the
$\lfloor t\,(n+1) \rfloor$ most promising-looking non-hits of the group, and at most $\lfloor t\,(n+1) \rfloor$ of the
$n+1$ reactions are in that position. Hence

$$ \Pr\{\, p_j \le t \ \text{ and candidate } j \text{ is a non-hit} \,\} \;\le\; t \qquad \text{for every } t : $$

a false alarm at level $t$ happens with probability at most $t$. (The construction is the "conformal selection" of Jin
and Candès, 2023, with their "clipped" score, written here for a threshold that is the same for all candidates.
Conformal p-values are also used to test for outliers; see Bates et al., 2023.)
"""

# %%
def conformal_pvalues(score_cal, null_cal, score_test):
    """p_j = (1 + number of calibration non-hits that look at least as promising as j) / (n + 1)."""
    null_scores = np.sort(score_cal[null_cal])
    return (1 + np.searchsorted(null_scores, score_test, side="right")) / (len(score_cal) + 1)


model = rdb7.models["delta"]
mu = model.mu
score = mu - c                                   # smaller = more promising
non_hit_cal = y[cal] >= c
p = conformal_pvalues(score[cal], non_hit_cal, score[test])
assert np.array_equal(p, P.conformal_pvalues(score[cal], non_hit_cal, score[test]))

print(f"calibration set: {non_hit_cal.sum()} non-hits, {(~non_hit_cal).sum()} hits; smallest possible p-value 1/(n+1) = {1 / (len(cal) + 1):.4f}")
print(f"median p-value of the candidates that are hits: {np.median(p[is_hit]):.4f};  of the non-hits: {np.median(p[~is_hit]):.3f}")

# %% [markdown]
r"""
## 4. The Benjamini–Hochberg step

Selecting every candidate with $p_j \le 0.10$ was rule C, and it failed. The Benjamini–Hochberg (BH) procedure
(Benjamini and Hochberg, 1995) replaces the fixed cutoff by one that adapts to the number of candidates $m$:

1. sort the p-values, $p_{(1)} \le p_{(2)} \le \dots \le p_{(m)}$;
2. find the largest rank $r$ with $p_{(r)} \le q\,r/m$;
3. select the $r$ candidates with the smallest p-values (none if no such $r$ exists).

BH was designed for independent p-values, and ours are not independent: they all use the same calibration set. Jin and
Candès (2023) proved that the procedure nevertheless gives $\mathrm{FDR} \le q$ with these conformal p-values when the
calibration and candidate reactions are independent draws from the same population. Nothing is assumed about the
model or about the distribution of the barriers. (Their theorem breaks ties between equal scores at random. We count a
tie against the candidate instead, which is the least favorable outcome of that random draw for the candidate. Ties
are rare here, and the tests of the toolbox check that the false discovery rate stays below $q$ for heavily tied
scores as well.)
"""

# %%
def benjamini_hochberg(pvalues, q):
    """Boolean mask of the candidates selected by the Benjamini–Hochberg procedure at level q."""
    m = len(pvalues)
    order = np.argsort(pvalues, kind="stable")
    line = q * np.arange(1, m + 1) / m * (1 + 1e-12)     # the tiny factor makes "p equal to the line" count as below it
    passed = np.flatnonzero(pvalues[order] <= line)
    selected = np.zeros(m, dtype=bool)
    if passed.size:
        selected[order[: passed[-1] + 1]] = True
    return selected


q = 0.10
selected = benjamini_hochberg(p, q)
assert np.array_equal(selected, P.benjamini_hochberg(p, q))
print(f"Δ-learning model, target FDR {q:.0%}:     {in_words(report(selected, is_hit))}")

model_s = rdb7.models["structure"]
p_s = conformal_pvalues((model_s.mu - c)[cal], non_hit_cal, (model_s.mu - c)[test])
selected_s = benjamini_hochberg(p_s, q)
print(f"Structure-only model, target FDR {q:.0%}: {in_words(report(selected_s, is_hit))}")

# %% [markdown]
r"""
One split proves little, because the guarantee is about an expectation. For the structure-only model in particular
the outcome of a single split varies a great deal at this level: the list is short, its false discovery proportion can
lie above or below $q$, and for some calibration sets the list is empty (Section 5 explains why). The experiment below
repeats the selection for 300 random calibration/candidate splits, for both models, at several target levels $q$, and
adds the three tempting rules for comparison.
"""

# %%
LEVELS = (0.02, 0.05, 0.10, 0.20, 0.30)


def selection_experiment(name, c=c, n_cal=1000, n_rep=300, levels=LEVELS, seed=41, normalized=False):
    """FDP and power of conformal selection (and of rules A-C) over random calibration/candidate splits."""
    mu = rdb7.models[name].mu
    abs_err = np.abs(y - mu)
    score = (mu - c) / sigma_of[name] if normalized else mu - c
    rows = []
    for cal_b, test_b in P.resplits(rdb7.hold, n_cal, n_rep, seed=seed):
        hit_b = y[test_b] < c
        p_b = conformal_pvalues(score[cal_b], y[cal_b] >= c, score[test_b])
        rows.append({"rule": "A: predicted barrier below c", "q": np.nan, **report(mu[test_b] < c, hit_b)})
        rows.append({"rule": "B: 90% interval below c", "q": np.nan, **report(mu[test_b] + conformal_quantile(abs_err[cal_b], 0.10) < c, hit_b)})
        for level in levels:
            rows.append({"rule": "C: p-value below q, no correction", "q": level, **report(p_b <= level, hit_b)})
            rows.append({"rule": "conformal selection", "q": level, **report(benjamini_hochberg(p_b, level), hit_b)})
    return pd.DataFrame(rows)


def summarize(runs):
    """One row per rule and level: averages over the splits."""
    runs = runs.assign(fdp_if_not_empty=runs["FDP"].where(runs["selected"] > 0))      # undefined for an empty list
    g = runs.groupby(["rule", "q"], sort=False, dropna=False)
    return pd.DataFrame({"selected (mean)": g["selected"].mean(), "FDR (mean FDP)": g["FDP"].mean(),
                         "FDP, 90th percentile": g["FDP"].quantile(0.90), "power (mean)": g["power"].mean(),
                         "splits with an empty list": g["selected"].apply(lambda s: np.mean(s == 0)),
                         "mean FDP of the non-empty lists": g["fdp_if_not_empty"].mean()})


def show_summary(table):
    """Display a summary table; a dash stands for 'does not apply'."""
    shown = table.round(3).astype(object).where(table.notna(), "–")
    shown.index = pd.MultiIndex.from_tuples([(rule, "–" if np.isnan(level) else f"{level:.2f}") for rule, level in table.index],
                                            names=["rule", "q"])
    display(shown)


runs = {name: selection_experiment(name) for name in rdb7.models}
for name, mod in rdb7.models.items():
    print(mod.label)
    show_summary(summarize(runs[name]))

# rule A against conformal selection at the false discovery rate that rule A turned out to have
rows = []
for name, mod in rdb7.models.items():
    rule_a = summarize(runs[name]).loc["A: predicted barrier below c"].iloc[0]
    same_level = selection_experiment(name, levels=(float(rule_a["FDR (mean FDP)"]),))
    same_level = same_level[same_level["rule"] == "conformal selection"]
    rows.append({"model": mod.label, "FDR of rule A": rule_a["FDR (mean FDP)"], "power of rule A": rule_a["power (mean)"],
                 "FDR of conformal selection at q = FDR of rule A": same_level["FDP"].mean(), "its power": same_level["power"].mean()})
versus_rule_a = pd.DataFrame(rows).set_index("model")
display(versus_rule_a.round(3))

# %% [markdown]
r"""
Read the rows "conformal selection" first. For both models and every level, the mean false discovery proportion is at
or below the target $q$: **the promise is kept by a model with a mean absolute error (MAE) of 1.6 kcal/mol and equally
by one with an MAE of 8.6 kcal/mol.** What the model quality decides is the *power*. The Δ-learning model finds more
than 90% of the hits already at $q = 5\%$; the structure-only model finds less than a quarter of the hits at $q = 10\%$
and needs $q = 30\%$ to find more than half of them. As with the intervals of Notebook 2: validity does not depend on
the model, power does.

Compare with the tempting rules. Rule A happens to be fine for the Δ-learning model and delivers more than 20% false
discoveries for the structure-only model. The last table puts the two approaches side by side at the same error rate:
there rule A finds slightly more hits (the stars in panel c below lie a little above the curves). That is the price of
the guarantee: rule A gives a slightly better list whose error rate is unknown until the reference values exist;
conformal selection gives up a few percent of the hits and states the error rate in advance. Rule C exceeds every
target by a wide margin.

Two further observations. The FDR is an *average*: the column "FDP, 90th percentile" shows that an individual list can
contain a larger fraction of false entries than $q$. And at strict levels the list is sometimes empty. An empty list
enters the average as a list without errors, so part of the promise can be kept by selecting nothing: when empty lists
are frequent, the lists that are *not* empty may contain more false entries than $q$ on average, as they do here for
the structure-only model at the strict levels (last column). Section 5 explains when the list is empty, and Section 7
comes back to what the promise means.
"""

# %%
fig, axes = P.panels(3)

# a: the Benjamini–Hochberg picture for one split (Δ-learning model)
ax = axes[0]
m_cand = len(p)
n_selected = int(selected.sum())
r_max = min(m_cand, max(800, 200 * int(np.ceil(1.5 * n_selected / 200))))   # 800 candidates unless the list is longer
ranks = np.arange(1, r_max + 1)
p_sorted = np.sort(p)[:r_max]
p_top = max(0.04, 1.6 * q * r_max / m_cand)
ax.axvspan(0, n_selected, color=P.SKY, alpha=0.30, lw=0)
ax.plot(ranks, q * ranks / m_cand, color=P.VERMILLION, lw=1.2, ls="--")
ax.plot(ranks, p_sorted, color=P.BLUE)
ax.text(0.55 * n_selected, 0.775 * p_top, f"selected:\n{n_selected} reactions", ha="center", va="center", fontsize=6.5)
r_label = int(0.81 * r_max)
ax.text(r_label, q * r_label / m_cand + 0.1 * p_top, "BH line $q\\,r/m$", ha="center", va="center", fontsize=6.5)
r_label = max(1, min(int(1.3 * n_selected), r_max - 1))
right = r_label > 0.8 * r_max                                         # near the right edge: label to the left of the curve
ax.annotate("sorted p-values", (r_label, min(p_sorted[r_label - 1], 0.95 * p_top)), xytext=(-4 if right else 4, 0),
            textcoords="offset points", ha="right" if right else "left", va="center", fontsize=6.5)
ax.set(xlabel="candidates ordered by p-value, $r$", ylabel="p-value", xlim=(0, r_max), ylim=(0, p_top))
if r_max < m_cand:                                                    # a multiple of 200: five round tick marks
    ax.set_xticks(np.linspace(0, r_max, 5))
P.panel_label(ax, "a", "Benjamini–Hochberg step")

# b: realized FDR against the target
ax = axes[1]
ax.plot([0, 0.32], [0, 0.32], color=P.GRAY, lw=0.8, ls="--", label="FDR = $q$")
for name in ("structure", "delta"):
    stats_q = summarize(runs[name]).loc["conformal selection"]
    ax.plot(stats_q.index, stats_q["FDR (mean FDP)"], marker=P.MODEL_MARKERS[name], color=P.MODEL_COLORS[name],
            label=rdb7.models[name].label.replace(" model", ""))
ax.set(xlabel="target level $q$", ylabel="false discovery rate", xlim=(0, 0.32), ylim=(0, 0.32))
ax.set_xticks([0, 0.1, 0.2, 0.3])
ax.set_yticks([0, 0.1, 0.2, 0.3])
ax.legend(loc="upper left", borderaxespad=0.3)
P.panel_label(ax, "b", "FDR at or below $q$")

# c: power against the target
ax = axes[2]
for name in ("structure", "delta"):
    stats_q = summarize(runs[name]).loc["conformal selection"]
    ax.plot(stats_q.index, stats_q["power (mean)"], marker=P.MODEL_MARKERS[name], color=P.MODEL_COLORS[name],
            label=rdb7.models[name].label.replace(" model", ""))
    rule_a = versus_rule_a.loc[rdb7.models[name].label]
    ax.plot([rule_a["FDR of rule A"]], [rule_a["power of rule A"]], ls="none", marker="*", markersize=9, color=P.MODEL_COLORS[name],
            markeredgecolor="white", markeredgewidth=0.6, label="rule A" if name == "delta" else None)
ax.set(xlabel="target level $q$", ylabel="fraction of hits found (power)", xlim=(0, 0.32), ylim=(0, 1.08))
ax.set_xticks([0, 0.1, 0.2, 0.3])
ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
ax.legend(loc="center left", bbox_to_anchor=(0.02, 0.57), borderaxespad=0)
P.panel_label(ax, "c", "Power is earned")
for ax in axes:
    ax.set_box_aspect(1)
P.savefig(fig, "nb6_selection")
plt.show()

# %% [markdown]
r"""
Panel **a**: the BH step for one split and the Δ-learning model; only the first 800 of the 4935 candidates are shown.
BH selects all candidates up to the last rank $r$ at which the sorted p-values (solid) are at or below the line $q\,r/m$
(dashed). For the first 49 candidates the curve lies above the line, which is why a list cannot be shorter than 50
here (Section 5). Panel **b**: the realized false discovery rate, averaged over 300 splits, against the target. Panel
**c**: the power. The stars show rule A (select if the predicted barrier is below $c$) at the false discovery rate it
turned out to have.

## 5. How large must the calibration set be?

A conformal p-value cannot be smaller than $1/(n+1)$. BH selects $r$ candidates only if the $r$-th smallest p-value is
at most $q\,r/m$. Together:

$$ \frac{1}{n+1} \le \frac{q\,r}{m} \quad\Longrightarrow\quad r \;\ge\; \frac{m}{q\,(n+1)} . $$

A BH shortlist is therefore either empty or contains at least $m/(q(n+1))$ candidates. Put differently, a shortlist
can only be as short as a fraction $f$ of the candidates if **the calibration set has at least about $1/(q f)$
reactions.** If 5% of the candidates are hits, a list of that length at $q = 10\%$ needs $n \ge 200$; in a
needle-in-a-haystack screen with 0.1% hits it needs $n \ge 10{,}000$. The experiment confirms the threshold:
"""

# %%
rows = []
for name in rdb7.models:
    for n_cal in (100, 200, 500, 1000, 2000):
        r = selection_experiment(name, n_cal=n_cal, n_rep=200, levels=(0.10,), seed=42)
        r = r[r["rule"] == "conformal selection"]
        m_cand = len(rdb7.hold) - n_cal
        nonempty = r.loc[r["selected"] > 0, "selected"]
        rows.append({"model": rdb7.models[name].label, "calibration reactions n": n_cal,
                     "smallest possible list m/(q(n+1))": int(np.ceil(m_cand / (0.10 * (n_cal + 1)))),
                     "smallest non-empty list observed": int(nonempty.min()) if len(nonempty) else np.nan,
                     "splits with an empty list": np.mean(r["selected"] == 0), "FDR": r["FDP"].mean(), "power": r["power"].mean()})
resolution = pd.DataFrame(rows).set_index(["model", "calibration reactions n"])
display(resolution.round(3))

# %% [markdown]
r"""
With 100 calibration reactions a list needs at least 578 entries, more than there are hits. However good the model,
the list is then empty in the great majority of the splits (and when it is not, it has to contain many non-hits). From
200 on the Δ-learning model finds most hits. For the structure-only model a larger calibration set keeps helping,
because many of its p-values lie close to the BH line.

## 6. A better score gives more power

The score $\hat\mu(x) - c$ treats all predictions as equally reliable. Dividing by the difficulty estimate of Notebook 3
gives $(\hat\mu(x) - c)/\hat\sigma(x)$: a candidate predicted 5 kcal/mol below the threshold counts for more when the
model is usually accurate for reactions of its kind. Any function of $x$ is a legitimate score, so the guarantee is
unchanged; only the power can differ.
"""

# %%
rows = []
for name in rdb7.models:
    for label, flag in (("mu - c", False), ("(mu - c) / sigma", True)):
        r = selection_experiment(name, normalized=flag, levels=(0.05, 0.10, 0.20))
        r = r[r["rule"] == "conformal selection"].groupby("q").mean(numeric_only=True)
        for level, row in r.iterrows():
            rows.append({"model": rdb7.models[name].label, "score": label, "q": level, "FDR": row["FDP"], "power": row["power"]})
scores = pd.DataFrame(rows).pivot_table(index=["model", "q"], columns="score", values=["FDR", "power"])
display(scores.round(3))

# %% [markdown]
r"""
For the structure-only model the normalized score finds clearly more hits at the strict levels; for the Δ-learning
model, which already finds nearly everything, there is little left to gain.

## 7. Reading the result, and its limits

* **What the list means.** "FDR at most 10%" says: averaged over random calibration and candidate sets, at most one in
  ten of the selected reactions is a non-hit, *where an empty list counts as a list without errors*. It is a statement
  about the list as a whole; it is not a 90% probability for each entry. When empty lists are frequent, the lists that
  are not empty can contain more false entries than $q$ on average (for them alone, the bound is $q$ divided by the
  probability that the list is not empty): for the structure-only model at $q = 10\%$ the list is empty in about a
  third of the splits and the other lists contain 12% non-hits on average (at $q = 5\%$: empty in 85% of the splits,
  13% non-hits in the others). For the Δ-learning model, whose list is practically never empty here, there is no such
  difference.
* **The threshold must be fixed in advance.** Tuning $c$ or $q$ after looking at the list invalidates the guarantee.
  (The threshold may differ from candidate to candidate if it is a known function of $x$ that is defined in the same
  way for the calibration reactions.)
* **Exchangeability, again.** The calibration set *as a whole* must be exchangeable with the candidates: hits and
  non-hits in the proportions of the candidate population, and its non-hits a fair sample of the candidates' non-hits.
  A calibration set enriched in hits (reference calculations that were run because the reactions looked promising)
  breaks the guarantee. Everything in Notebook 5 applies; weighted conformal p-values exist for covariate shift (Jin
  and Candès, 2026).
* **Do not attach ordinary intervals to the selected reactions.** Selection changes the population: the reactions on
  the list are those whose predictions were low, and predictions that are too low are over-represented among them.
  Intervals that remain valid after selection need a separate construction.

## Take-aways

* Screening needs error control for a *list*, not for single predictions. The false discovery rate is the natural
  quantity.
* Conformal p-values compare each candidate with the calibration reactions that are known non-hits; the
  Benjamini–Hochberg step turns them into a shortlist with FDR at most $q$, for any model.
* A poor model does not break the guarantee. It shortens the list.
* A shortlist that is a fraction $f$ of the candidates requires a calibration set of about $1/(q f)$ reactions.

## Try it yourself

1. Change the threshold to `c = 60.0` (a half-life of about 0.4 s at 1000 K). How do the fraction of hits among the
   candidates and the power of the structure-only model change?
2. Select reactions with a *high* barrier instead (say, above 110 kcal/mol). Three definitions have to change in
   Sections 1 to 4: the score, which calibration reactions count as non-hits, and which candidates count as hits.
3. Run rule C at the level `q / m` (the Bonferroni correction). It controls an even stricter error rate. What happens
   to the power?

**This is the last notebook.** Together the six notebooks cover the path from a point prediction to an interval with a
guarantee (2), to intervals that adapt (3) and are valid class by class (4), to the limits of the guarantee (5), and
finally to decisions (6).

## References

* Jin, Y.; Candès, E. J. Selection by Prediction with Conformal p-Values. *J. Mach. Learn. Res.* **2023**, *24* (244),
  1–41. https://jmlr.org/papers/v24/22-1176.html
* Benjamini, Y.; Hochberg, Y. Controlling the False Discovery Rate: A Practical and Powerful Approach to Multiple
  Testing. *J. R. Stat. Soc. Ser. B: Methodol.* **1995**, *57*, 289–300. https://doi.org/10.1111/j.2517-6161.1995.tb02031.x
* Bates, S.; Candès, E.; Lei, L.; Romano, Y.; Sesia, M. Testing for Outliers with Conformal p-Values. *Ann. Stat.*
  **2023**, *51*, 149–178. https://doi.org/10.1214/22-AOS2244
* Jin, Y.; Candès, E. J. Model-Free Selective Inference Under Covariate Shift via Weighted Conformal p-Values.
  *Biometrika* **2026**, *113*, asaf066. https://doi.org/10.1093/biomet/asaf066
"""

# %%
def table_dict(frame):
    out = {}
    for (rule, level), row in frame.iterrows():
        key = rule if not np.isfinite(level) else f"{rule} | q={level:.2f}"
        out[key] = {name: value for name, value in row.items() if np.isfinite(value)}     # undefined entries are left out
    return out


results = {
    "threshold": c, "hit_rate_hold": float(np.mean(y[rdb7.hold] < c)), "eyring_half_life_ms": float(1000 * np.log(2) / rate),
    "n_candidates": len(test),
    "single_split": {f"{model_label} | {rule}": dict(row) for (model_label, rule), row in tempting.to_dict("index").items()},
    "single_split_bh": {"delta": report(selected, is_hit), "structure": report(selected_s, is_hit)},
    "experiment": {name: table_dict(summarize(runs[name])) for name in rdb7.models},
    "versus_rule_a": {label: dict(row) for label, row in versus_rule_a.to_dict("index").items()},
    "resolution": {f"{model_label} | n={n}": dict(row) for (model_label, n), row in resolution.to_dict("index").items()},
    "scores": {f"{model_label} | q={level:.2f} | {stat} | {score}": value
               for (model_label, level), row in scores.iterrows() for (stat, score), value in row.items()},
}
P.save_results("nb6", results, rdb7)
P.check("nb6", results);
