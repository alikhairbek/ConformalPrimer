"""
conformal_toolbox.py -- the conformal prediction functions of the ConformalPrimer notebooks

Split conformal prediction, the Beta law of the coverage, CV+, normalized scores, conformalized
quantile regression, Mondrian quantiles, weighted conformal quantiles, conformal p-values and the
Benjamini–Hochberg procedure. Each function is derived step by step in the notebook named in its
docstring. The file needs only NumPy (and SciPy for `coverage_law`) and can be copied into any project.

Generated from primer.py by build_notebooks.py; edit primer.py, not this file.
"""

import numpy as np

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
