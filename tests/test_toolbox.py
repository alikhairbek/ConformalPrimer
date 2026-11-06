"""
Tests of the conformal toolbox against theory, on simulated data (no chemistry, no downloads).

    python -m pytest tests            (or simply:  python tests/test_toolbox.py)

Every test states the theoretical value it checks. Monte Carlo tolerances are four standard errors.
"""
import math
import os
import sys

import numpy as np
from scipy import stats

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import conformal_toolbox as T  # noqa: E402


def heavy_tailed(rng, size):
    """Scores with very heavy tails: validity must not depend on the distribution."""
    return np.abs(rng.standard_cauchy(size)) ** 1.5


# ----------------------------------------------------------------------------- conformal quantile
def test_conformal_quantile_is_the_right_order_statistic():
    rng = np.random.default_rng(0)
    for n in (1, 2, 8, 9, 10, 19, 20, 50, 99, 100, 1000):
        scores = rng.normal(size=n)
        for alpha in (0.01, 0.05, 0.1, 0.2, 0.32, 0.5):
            k = math.ceil(round((n + 1) * (1 - alpha), 9))
            expected = np.inf if k > n else np.sort(scores)[k - 1]
            assert T.conformal_quantile(scores, alpha) == expected, (n, alpha)


def test_smallest_calibration_sets():
    # (1 - alpha)/alpha points are needed for a finite interval: 9 at 90%, 19 at 95%, 99 at 99%
    for alpha, n_min in ((0.10, 9), (0.05, 19), (0.01, 99)):
        scores = np.arange(1.0, n_min + 1)
        assert T.conformal_quantile(scores, alpha) == n_min          # the largest score
        assert np.isinf(T.conformal_quantile(scores[:-1], alpha))


def test_marginal_coverage_equals_k_over_n_plus_1():
    rng = np.random.default_rng(1)
    n_rep = 40000
    for n, alpha in ((9, 0.1), (20, 0.1), (57, 0.1), (200, 0.05), (33, 0.32)):
        draws = heavy_tailed(rng, (n_rep, n + 1))
        q = np.array([T.conformal_quantile(row[:n], alpha) for row in draws])
        covered = draws[:, n] <= q
        k = math.ceil(round((n + 1) * (1 - alpha), 9))
        target = k / (n + 1)                                          # exact for continuous scores
        se = math.sqrt(target * (1 - target) / n_rep)
        assert target >= 1 - alpha
        assert abs(covered.mean() - target) < 4 * se, (n, alpha, covered.mean(), target)


def test_coverage_law_is_beta():
    # with uniform scores the coverage given the calibration set is q itself
    rng = np.random.default_rng(2)
    for n, alpha in ((20, 0.1), (100, 0.1), (500, 0.05)):
        q = np.array([T.conformal_quantile(rng.uniform(size=n), alpha) for _ in range(4000)])
        law = T.coverage_law(n, alpha)
        k = math.ceil(round((n + 1) * (1 - alpha), 9))
        assert abs(law.mean() - k / (n + 1)) < 1e-12
        assert stats.kstest(q, law.cdf).pvalue > 1e-3
    assert T.coverage_law(8, 0.1) is None                              # too small for a finite interval


# ----------------------------------------------------------------------------- intervals
def simulate(rng, n):
    """Heteroscedastic regression problem with an imperfect model and an imperfect difficulty estimate."""
    x = rng.uniform(0, 1, n)
    group = (x > 0.7).astype(int)
    scale = 0.2 + 2.0 * x ** 2
    y = np.sin(3 * x) + scale * rng.standard_t(3, n)
    mu = np.sin(3 * x) + 0.1                                           # biased on purpose
    sigma = 0.3 + 1.5 * x                                              # only roughly right
    return x, y, mu, sigma, group


def test_split_normalized_cqr_cover():
    rng = np.random.default_rng(3)
    alpha, n_cal, n_test, n_rep = 0.1, 200, 400, 300
    hits = {"split": [], "normalized": [], "cqr": []}
    for _ in range(n_rep):
        x, y, mu, sigma, _ = simulate(rng, n_cal + n_test)
        c, t = slice(0, n_cal), slice(n_cal, None)
        lo, hi, _ = T.split_conformal(y[c], mu[c], mu[t], alpha)
        hits["split"].append(np.mean((y[t] >= lo) & (y[t] <= hi)))
        lo, hi, _ = T.normalized_conformal(y[c], mu[c], sigma[c], mu[t], sigma[t], alpha)
        hits["normalized"].append(np.mean((y[t] >= lo) & (y[t] <= hi)))
        band_lo, band_hi = mu - 0.5 * sigma, mu + 0.5 * sigma          # a (too narrow) quantile band
        lo, hi, q = T.cqr(y[c], band_lo[c], band_hi[c], band_lo[t], band_hi[t], alpha)
        hits["cqr"].append(np.mean((y[t] >= lo) & (y[t] <= hi)))
    target = math.ceil((n_cal + 1) * (1 - alpha)) / (n_cal + 1)
    for name, values in hits.items():
        se = np.std(values) / math.sqrt(n_rep)
        assert abs(np.mean(values) - target) < 4 * se, (name, np.mean(values), target)


def test_cqr_can_shrink_a_band_that_is_too_wide():
    rng = np.random.default_rng(4)
    y = rng.normal(size=2000)
    lo, hi, q = T.cqr(y[:1000], np.full(1000, -5.0), np.full(1000, 5.0), np.full(1000, -5.0), np.full(1000, 5.0), 0.1)
    assert q < 0 and abs(hi[0] - 1.645) < 0.2


def test_interval_score():
    y = np.array([0.0, 2.0, -3.0])
    lower, upper = np.array([-1.0, -1.0, -1.0]), np.array([1.0, 1.0, 1.0])
    # widths 2, 2, 2; misses 0, 1, 2 -> penalties (2/alpha) * miss
    assert abs(T.interval_score(y, lower, upper, 0.1) - (2 + (2 / 0.1) * (0 + 1 + 2) / 3)) < 1e-12


def test_mondrian_covers_every_group():
    rng = np.random.default_rng(5)
    alpha, n_rep = 0.1, 300
    cov_global, cov_mondrian = [], []
    for _ in range(n_rep):
        x, y, mu, sigma, group = simulate(rng, 1500)
        score = np.abs(y - mu)
        c, t = slice(0, 500), slice(500, None)
        q_all = T.conformal_quantile(score[c], alpha)
        q_group = T.mondrian_quantiles(score[c], group[c], alpha)
        q_test = np.array([q_group[g] for g in group[t]])
        cov_global.append([np.mean(score[t][group[t] == g] <= q_all) for g in (0, 1)])
        cov_mondrian.append([np.mean(score[t][group[t] == g] <= q_test[group[t] == g]) for g in (0, 1)])
    cov_global, cov_mondrian = np.mean(cov_global, axis=0), np.mean(cov_mondrian, axis=0)
    assert cov_global[1] < 0.80 < 0.95 < cov_global[0]                 # one quantile: hard group under-covered
    assert np.all(np.abs(cov_mondrian - 0.9) < 0.015), cov_mondrian   # Mondrian: both groups at 90%


# ----------------------------------------------------------------------------- CV+
def test_cv_plus_reduces_to_split_conformal_for_identical_fold_models():
    rng = np.random.default_rng(6)
    for n, alpha in ((40, 0.1), (57, 0.1), (200, 0.05)):
        y, oof = rng.normal(size=n), rng.normal(size=n)
        fold = rng.integers(0, 5, size=n)
        prediction = rng.normal(size=11)
        lower, upper = T.cv_plus(y, oof, fold, np.tile(prediction, (5, 1)), alpha)
        q = T.conformal_quantile(np.abs(y - oof), alpha)
        assert np.allclose(lower, prediction - q) and np.allclose(upper, prediction + q)


def test_cv_plus_coverage_with_least_squares():
    rng = np.random.default_rng(7)
    alpha, n, d, n_folds, n_rep = 0.1, 60, 10, 6, 400
    covered = []
    for _ in range(n_rep):
        X = rng.normal(size=(n + 200, d))
        y = X @ np.ones(d) + rng.standard_t(3, size=n + 200)
        fold = rng.permutation(n) % n_folds
        oof, test_pred = np.empty(n), np.empty((n_folds, 200))
        for k in range(n_folds):
            beta = np.linalg.lstsq(X[:n][fold != k], y[:n][fold != k], rcond=None)[0]
            oof[fold == k] = X[:n][fold == k] @ beta
            test_pred[k] = X[n:] @ beta
        lower, upper = T.cv_plus(y[:n], oof, fold, test_pred, alpha)
        covered.append(np.mean((y[n:] >= lower) & (y[n:] <= upper)))
    assert np.mean(covered) > 1 - 2 * alpha                            # the guarantee
    assert abs(np.mean(covered) - 0.9) < 0.03                          # and in practice close to 1 - alpha


# ----------------------------------------------------------------------------- weighted quantile
def test_weighted_quantile_with_equal_weights_is_the_conformal_quantile():
    rng = np.random.default_rng(8)
    for n in (5, 9, 10, 37, 500):
        scores = rng.normal(size=n)
        for alpha in (0.05, 0.1, 0.3):
            w = T.weighted_conformal_quantile(scores, np.full(n, 0.7), np.array([0.7, 0.7]), alpha)
            assert w[0] == w[1] == T.conformal_quantile(scores, alpha), (n, alpha)


def test_weighted_quantile_restores_coverage_under_covariate_shift():
    rng = np.random.default_rng(9)
    alpha, n_cal, n_test, n_rep = 0.1, 500, 300, 300
    plain, weighted = [], []
    for _ in range(n_rep):
        x_cal = rng.uniform(0, 1, n_cal)
        x_test = rng.beta(4, 1, n_test)                                # shifted toward x = 1, density 4 x^3
        score = lambda x: np.abs((0.2 + 2.0 * x ** 2) * rng.standard_normal(len(x)))
        s_cal, s_test = score(x_cal), score(x_test)
        plain.append(np.mean(s_test <= T.conformal_quantile(s_cal, alpha)))
        q = T.weighted_conformal_quantile(s_cal, 4 * x_cal ** 3, 4 * x_test ** 3, alpha)
        weighted.append(np.mean(s_test <= q))
    assert np.mean(plain) < 0.80                                       # the unweighted interval fails
    assert 0.9 <= np.mean(weighted) < 0.925, np.mean(weighted)         # the weighted one is valid


def exact_weighted_quantile(scores, w_cal, w_test, alpha):
    """The definition, in exact rational arithmetic: the smallest score v with
    (weight of the calibration scores <= v) >= (1 - alpha) * (total calibration weight + test weight)."""
    from fractions import Fraction

    total = sum(w_cal, Fraction(0)) + w_test
    if total == 0:
        return math.inf
    below = Fraction(0)
    for score, weight in sorted(zip(scores, w_cal)):
        below += weight
        if below >= (1 - alpha) * total:
            return score
    return math.inf


def test_weighted_quantile_against_exact_arithmetic():
    from fractions import Fraction

    rng = np.random.default_rng(13)
    for _ in range(400):
        n = int(rng.integers(1, 30))
        scores = rng.permutation(n).astype(float)                      # distinct scores
        w_cal = [Fraction(int(v), 7) for v in rng.integers(0, 6, size=n)]
        w_test = Fraction(int(rng.integers(0, 6)), 7)
        alpha = Fraction(int(rng.integers(1, 10)), 20)                 # 0.05, 0.10, ..., 0.45
        expected = exact_weighted_quantile(scores, w_cal, w_test, alpha)
        got = T.weighted_conformal_quantile(scores, [float(w) for w in w_cal], float(w_test), float(alpha))[0]
        assert got == expected, (n, alpha, got, expected)


def test_weighted_quantile_does_not_depend_on_the_scale_of_the_weights():
    rng = np.random.default_rng(14)
    for n in (1, 9, 10, 19, 40, 99, 100, 999):
        scores = rng.normal(size=n)
        w_cal, w_test = rng.uniform(0.1, 3.0, size=n), rng.uniform(0.1, 3.0, size=5)
        for alpha in (0.05, 0.1, 0.2, 0.5):
            reference = T.weighted_conformal_quantile(scores, w_cal, w_test, alpha)
            for scale in (1e-15, 1e-9, 1e-3, 1e3, 1e9, 1e15):          # e.g. unnormalized Boltzmann factors
                assert np.array_equal(T.weighted_conformal_quantile(scores, scale * w_cal, scale * w_test, alpha), reference), (n, alpha, scale)
                equal = T.weighted_conformal_quantile(scores, np.full(n, scale), scale, alpha)[0]
                assert equal == T.conformal_quantile(scores, alpha), (n, alpha, scale)


def test_weighted_quantile_edge_cases():
    scores = np.array([1.0, 2.0, 3.0])
    assert np.isinf(T.weighted_conformal_quantile(scores, np.zeros(3), 0.0, 0.1)[0])       # no weight at all
    assert np.isinf(T.weighted_conformal_quantile(scores, np.ones(3), 100.0, 0.1)[0])      # the test point outweighs everything
    assert np.isinf(T.weighted_conformal_quantile(np.array([]), np.array([]), 1.0, 0.1)[0])
    for bad in (np.array([1.0, np.nan, 1.0]), np.array([1.0, -1.0, 1.0]), np.array([1.0, np.inf, 1.0])):
        try:
            T.weighted_conformal_quantile(scores, bad, 1.0, 0.1)
        except ValueError:
            continue
        raise AssertionError("invalid weights were accepted")


def test_effective_sample_size():
    assert T.effective_sample_size(np.ones(50)) == 50
    assert T.effective_sample_size(np.array([1.0, 0, 0, 0])) == 1
    assert abs(T.effective_sample_size(np.array([1.0, 1.0, 2.0])) - 16 / 6) < 1e-12
    assert T.effective_sample_size(np.zeros(4)) == 0.0
    assert abs(T.effective_sample_size(1e-12 * np.array([1.0, 1.0, 2.0])) - 16 / 6) < 1e-9


# ----------------------------------------------------------------------------- selection
def naive_benjamini_hochberg(p, q):
    m = len(p)
    best = 0
    for k in range(1, m + 1):
        if np.sort(p)[k - 1] <= q * k / m:
            best = k
    selected = np.zeros(m, dtype=bool)
    selected[np.argsort(p, kind="stable")[:best]] = True
    return selected


def test_benjamini_hochberg_against_the_definition():
    rng = np.random.default_rng(10)
    for _ in range(50):
        p = np.concatenate([rng.uniform(size=60), rng.uniform(size=15) ** 6])
        for q in (0.05, 0.1, 0.3):
            assert np.array_equal(T.benjamini_hochberg(p, q), naive_benjamini_hochberg(p, q))
    assert not T.benjamini_hochberg(np.array([0.5, 0.9]), 0.1).any()


def test_benjamini_hochberg_accepts_p_values_that_lie_exactly_on_the_line():
    # p_(k) = q k / m must count as "at or below the line" although q * k / m is rounded in floating point
    for m in (7, 20, 100, 493, 1000):
        for q in (0.05, 0.1, 0.15, 0.2, 0.3):
            for k in (1, m // 3 + 1, m):
                p = np.ones(m)
                p[:k] = (q * k) / m                                    # k p-values exactly on the line at rank k
                assert T.benjamini_hochberg(p, q).sum() == k, (m, q, k)
                p[:k] = np.nextafter((q * k) / m * (1 + 1e-9), 1.0)    # clearly above the line: nothing is selected
                assert T.benjamini_hochberg(p, q).sum() == 0, (m, q, k)


def test_conformal_pvalue_is_valid():
    # Pr{p <= t and the candidate is a non-hit} <= t for every t, also with tied scores
    rng = np.random.default_rng(15)
    n_cal, n_rep = 30, 40000
    for tied in (False, True):
        mu = rng.normal(size=(n_rep, n_cal + 1))
        y = mu + rng.standard_t(3, size=mu.shape)
        score = np.round(mu, 0) if tied else mu
        non_hit = y >= -0.5
        p = np.array([T.conformal_pvalues(s[:n_cal], h[:n_cal], s[n_cal:])[0] for s, h in zip(score, non_hit)])
        for t in (1 / 31, 2 / 31, 0.1, 0.2, 0.5):
            rate = np.mean((p <= t) & non_hit[:, n_cal])
            assert rate <= t + 4 * math.sqrt(t * (1 - t) / n_rep), (tied, t, rate)


def test_invalid_arguments_are_refused():
    scores = np.arange(1.0, 21.0)
    calls = [lambda: T.conformal_quantile(scores, 0.0), lambda: T.conformal_quantile(scores, 1.0),
             lambda: T.conformal_quantile(scores, 1.5), lambda: T.conformal_quantile(np.array([1.0, np.nan]), 0.1),
             lambda: T.normalized_conformal(scores, scores, np.zeros(20), scores, np.ones(20), 0.1),
             lambda: T.benjamini_hochberg(np.array([0.01, 0.5]), 1.0),
             lambda: T.weighted_conformal_quantile(scores, np.ones(19), 1.0, 0.1),
             lambda: T.weighted_conformal_quantile(np.array([1.0, np.nan]), np.ones(2), 1.0, 0.1),
             lambda: T.conformal_pvalues(np.array([1.0, np.nan]), np.array([True, True]), np.array([0.5])),
             lambda: T.conformal_pvalues(np.array([1.0, 2.0]), np.array([True, True]), np.array([np.nan]))]
    for call in calls:
        try:
            call()
        except ValueError:
            continue
        raise AssertionError("an invalid argument was accepted")


def test_plain_lists_are_accepted():
    y, mu = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0], [1.5, 2.5, 2.0, 4.0, 5.5, 6.0, 6.0, 8.5, 9.0]
    lower, upper, q = T.split_conformal(y, mu, [3.0, 4.0], 0.1)
    assert q == 1.0 and list(upper - lower) == [2.0, 2.0]
    lower, upper, q = T.normalized_conformal(y, mu, [1.0] * 9, [3.0, 4.0], [1.0, 2.0], 0.1)
    assert list(upper - lower) == [2.0, 4.0]
    lower, upper, q = T.cqr(y, [v - 1 for v in mu], [v + 1 for v in mu], [2.0, 3.0], [4.0, 5.0], 0.1)
    assert q == 0.0 and list(lower) == [2.0, 3.0]


def test_conformal_pvalue_formula():
    score_cal = np.array([3.0, 1.0, 2.0, 5.0, 4.0])
    null_cal = np.array([True, True, False, True, True])               # the calibration point with score 2 is a hit
    p = T.conformal_pvalues(score_cal, null_cal, np.array([0.5, 1.0, 2.5, 10.0]))
    # non-hit scores: 1, 3, 4, 5 -> counts of non-hits with score <= test score: 0, 1, 1, 4
    assert np.allclose(p, np.array([1, 2, 2, 5]) / 6)


def test_conformal_selection_controls_the_false_discovery_rate():
    rng = np.random.default_rng(11)
    c, n_cal, n_test, n_rep = -1.0, 400, 1500, 300
    for noise, q in ((0.3, 0.1), (0.6, 0.1), (1.0, 0.2)):
        fdp, power = [], []
        for _ in range(n_rep):
            mu = rng.normal(size=n_cal + n_test)                       # the "model prediction"
            y = mu + noise * rng.standard_t(4, size=n_cal + n_test)    # the truth; a hit has y < c
            cal, test = slice(0, n_cal), slice(n_cal, None)
            p = T.conformal_pvalues((mu - c)[cal], y[cal] >= c, (mu - c)[test])
            selected = T.benjamini_hochberg(p, q)
            hit = y[test] < c
            fdp.append((selected & ~hit).sum() / max(selected.sum(), 1))
            power.append((selected & hit).sum() / max(hit.sum(), 1))
        se = np.std(fdp) / math.sqrt(n_rep)
        assert np.mean(fdp) <= q + 4 * se, (noise, q, np.mean(fdp))    # FDR <= q
        assert np.mean(power) > 0.05                                   # and the procedure is not vacuous


def test_conformal_selection_with_heavily_tied_scores():
    # the p-values count a tie against the candidate; the false discovery rate must stay below q all the same
    rng = np.random.default_rng(16)
    c, n_cal, n_test, n_rep, q = -1.0, 1000, 1000, 200, 0.1
    for grid in (0.1, 0.5):                                            # scores rounded to a grid: 60 or 15 distinct values
        fdp, selected_any = [], 0
        for _ in range(n_rep):
            mu = rng.normal(size=n_cal + n_test)
            y = mu + 0.3 * rng.standard_t(4, size=n_cal + n_test)
            score = np.round((mu - c) / grid) * grid
            p = T.conformal_pvalues(score[:n_cal], y[:n_cal] >= c, score[n_cal:])
            selected = T.benjamini_hochberg(p, q)
            hit = y[n_cal:] < c
            fdp.append((selected & ~hit).sum() / max(selected.sum(), 1))
            selected_any += bool(selected.any())
        assert np.mean(fdp) <= q + 4 * np.std(fdp) / math.sqrt(n_rep), (grid, np.mean(fdp))
        assert selected_any > 0.9 * n_rep, (grid, selected_any)        # the test is not vacuous


def test_shortlist_is_empty_or_at_least_m_over_q_n_plus_1():
    # a conformal p-value is at least 1/(n+1), so BH can only select k >= m / (q (n+1)) candidates
    rng = np.random.default_rng(12)
    for n_cal, n_test, q in ((50, 1000, 0.1), (100, 1000, 0.1), (400, 1000, 0.05)):
        smallest = math.ceil(n_test / (q * (n_cal + 1)))
        for _ in range(200):
            mu = rng.normal(size=n_cal + n_test)
            y = mu + 0.3 * rng.normal(size=n_cal + n_test)
            p = T.conformal_pvalues(mu[:n_cal], y[:n_cal] >= -1.0, mu[n_cal:])
            k = int(T.benjamini_hochberg(p, q).sum())
            assert k == 0 or k >= smallest, (n_cal, k, smallest)


# ----------------------------------------------------------------------------- consistency with primer.py
def test_toolbox_file_matches_primer():
    source = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "primer.py"), encoding="utf-8").read()
    section = source.split("# >>> toolbox\n", 1)[1].split("# <<< toolbox", 1)[0].strip("\n")
    generated = open(T.__file__, encoding="utf-8").read()
    assert section in generated


if __name__ == "__main__":
    failed = 0
    for name, function in sorted(globals().items()):
        if name.startswith("test_") and callable(function):
            try:
                function()
                print(f"ok      {name}")
            except AssertionError as exc:
                failed += 1
                print(f"FAILED  {name}: {exc}")
    print("all tests passed" if not failed else f"{failed} test(s) failed")
    sys.exit(1 if failed else 0)
