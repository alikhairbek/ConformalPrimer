"""
The numbers quoted in the text of the notebooks, checked against the results of a run.

    python tests/test_narrative.py                 uses outputs/ of the repository (run the notebooks first)
    python tests/test_narrative.py some/outputs    uses the nbN_results.json files in another folder

Every entry names the sentence it protects and the range within which the number must lie for the
sentence to remain true. The ranges are wide enough for runs with other library versions, which move
individual numbers by a few percent. If an entry fails, either the result has changed in a way that
matters, or the sentence in nbsrc/ has to be rewritten.
"""
import json
import math
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
import primer as P  # noqa: E402

OUT_DIR = os.path.join(HERE, "..", "outputs")
S, D = "Structure-only model", "Δ-learning model"
CLASSES = ("H transfer", "skeletal", "fragmentation", "carbene", "zwitterion")


class Numbers(dict):
    """The numbers of one notebook; a number that is not there counts as 'not a number' and is remembered."""

    absent = []

    def __missing__(self, key):
        Numbers.absent.append(key)
        return float("nan")


def load(out_dir):
    results = {}
    for name in P.NOTEBOOKS:
        path = os.path.join(out_dir, f"{name}_results.json")
        if os.path.exists(path):
            with open(path, encoding="utf-8") as fh:
                flat = P.flatten(json.load(fh)["results"])
            results[name] = Numbers({k: (float("inf") if v is None else v) for k, v in flat.items()})
    return results


def claims(r):
    """Yield (notebook, sentence, value, low, high)."""
    n1, n2, n3, n4, n5, n6 = (r.get(name, {}) for name in P.NOTEBOOKS)

    # ------------------------------------------------------------------------------------ Notebook 1
    if n1:
        yield "nb1", "RDB7 contains 11,926 gas-phase reactions", n1["n_reactions"], 11926, 11926
        yield "nb1", "no reactant on both sides of the split", n1["shared_reactants"], 0, 0
        yield "nb1", "a reaction family for about 12% of the reactions", n1["family_share"], 0.115, 0.13
        yield "nb1", "the MAE of the Δ-learning model is about 1.6 kcal/mol", n1["delta/errors/MAE"], 1.5, 1.72
        yield "nb1", "one prediction in ten is off by more than about 3.8 kcal/mol", n1["delta/errors/90"], 3.55, 4.05
        yield "nb1", "one in a hundred by more than about 11 kcal/mol", n1["delta/errors/99"], 9.8, 12.2
        yield "nb1", "the worst prediction by more than 30 kcal/mol", n1["delta/errors/largest"], 30, 80
        yield "nb1", "the '68%' interval of the Δ-learning model contains more than 80%", n1["delta/gaussian_coverage/0.68"], 0.80, 0.90
        for model in ("structure", "delta"):
            yield "nb1", f"the 99% Gaussian interval misses two to three times as often as promised ({model})", \
                (1 - n1[f"{model}/gaussian_coverage/0.99"]) / 0.01, 1.9, 3.5
        q90 = {c: n1[f"delta/class_q90/{c}"] for c in P.CLASS_ORDER}
        yield "nb1", "the 90th percentile differs by a factor of about three between the classes", max(q90.values()) / min(q90.values()), 2.4, 3.4
        yield "nb1", "... smallest errors: carbene products", float(min(q90, key=q90.get) == "carbene product"), 1, 1
        yield "nb1", "... largest errors: skeletal rearrangements", float(max(q90, key=q90.get) == "skeletal rearrangement"), 1, 1
        yield "README", "held-out MAE 8.6 kcal/mol (structure-only model)", n1["structure/errors/MAE"], 8.2, 9.0

    # ------------------------------------------------------------------------------------ Notebook 2
    if n2:
        yield "nb2", "n = 1000, k = 901", n2["single_split/k"], 901, 901
        for level in ("0.50", "0.68", "0.80", "0.90", "0.95", "0.99"):
            yield "nb2", f"the conformal intervals miss as often as promised (level {level}; observed / promised miss rate)", \
                (1 - n2[f"levels/{level}/conformal"]) / (1 - float(level)), 0.88, 1.12
            expected = math.ceil(1001 * float(level) - 1e-9) / 1001              # exact for n = 1000: k / (n + 1)
            yield "nb2", f"the conformal coverages differ from the promised levels by less than 0.001 (level {level})", \
                abs(n2[f"levels/{level}/conformal"] - float(level)), 0, 0.001
            yield "nb2", f"... the rounding in k puts the expected coverage up to 1/(n+1) above the level (level {level})", \
                expected - float(level), 0, 1 / 1001
            yield "nb2", f"... and the rest is the noise of the 1000 splits (level {level}; in standard errors)", \
                abs(n2[f"levels/{level}/conformal"] - expected) / n2[f"levels/{level}/conformal_se"], 0, 3.5
            yield "nb2", f"... whose standard error is a few ten-thousandths (level {level})", n2[f"levels/{level}/conformal_se"], 0.00005, 0.0009
        yield "nb2", "with 50 calibration reactions ... between about 81% ...", n2["coverage_vs_n/50/lo"], 0.795, 0.825
        yield "nb2", "... and 97%", n2["coverage_vs_n/50/hi"], 0.955, 0.978
        yield "nb2", "with 1000 it is pinned down to about ±2 percentage points (lower end)", n2["coverage_vs_n/1000/lo"], 0.874, 0.886
        yield "nb2", "with 1000 it is pinned down to about ±2 percentage points (upper end)", n2["coverage_vs_n/1000/hi"], 0.914, 0.926
        for n in (20, 50, 100, 200, 500, 1000, 2000):
            yield "nb2", f"the agreement with the Beta-binomial law is as close as 2000 repetitions allow (n = {n})", \
                n2[f"coverage_vs_n/{n}/mean"] - n2[f"coverage_vs_n/{n}/theory_mean"], -0.004, 0.004
        for key, value in (("pm5", 131), ("pm3", 374), ("pm2", 854), ("pm1", 3450)):
            yield "nb2", f"planning table ({key})", n2[f"n_needed/{key}"], value, value
        yield "nb2", "calibrating on the training set: covers only about three reactions in four", n2["pitfall_training/coverage"], 0.70, 0.80
        yield "nb2", "the empirical 90th percentile falls short for small n", n2["pitfall_quantile/20/plain"], 0.80, 0.88
        for n in (20, 50, 200, 1000):
            yield "nb2", f"the conformal quantile keeps the promise (n = {n})", n2[f"pitfall_quantile/{n}/conformal"], 0.898, 0.915
        for name in ("constant", "structure", "dft", "dft_shifted", "delta"):
            yield "nb2", f"all five rows have 90% coverage ({name})", n2[f"sharpness/{name}/coverage"], 0.895, 0.905
        yield "nb2", "'predict 80 kcal/mol for every reaction' is uselessly wide", n2["sharpness/constant/q"], 30, 45
        yield "nb2", "ωB97X-D3 overestimates almost nine barriers in ten", n2["dft_offset/share_positive"], 0.85, 0.92
        yield "nb2", "removing its mean error shrinks the interval by a third", n2["sharpness/dft_shifted/q"] / n2["sharpness/dft/q"], 0.58, 0.74
        yield "nb2", "the boosted trees fitted to 6000 reactions reached ±3.8", n2["sharpness/delta/q"], 3.55, 4.05
        yield "README", "90% conformal half-width ±20 kcal/mol (structure-only model)", n2["sharpness/structure/q"], 19, 21.5
        small = {key: n2[f"small_data/{key}"] for key in
                 (f"N={N} | {method}/{stat}" for N in (50, 100, 300, 1000) for method in ("split 50/50", "training residuals (wrong)", "CV+ (K = 10)")
                  for stat in ("coverage", "coverage_sd", "half_width", "half_width_sd"))}
        wrong = [small[f"N={N} | training residuals (wrong)/coverage"] for N in (50, 100, 300, 1000)]
        yield "nb2", "training residuals under-cover, the more so the fewer reactions there are", float(wrong[0] < wrong[1] < wrong[2] < wrong[3] < 0.90), 1, 1
        for N in (50, 100, 300, 1000):
            for method in ("split 50/50", "CV+ (K = 10)"):
                yield "nb2", f"split and CV+ both keep the promise on average (N = {N}, {method})", small[f"N={N} | {method}/coverage"], 0.893, 0.94
        yield "nb2", "split with 25 calibration reactions: expected coverage 24/26 = 92%", small["N=50 | split 50/50/coverage"], 0.905, 0.94
        for N in (50, 100):
            yield "nb2", f"CV+ gives narrower intervals (N = {N})", small[f"N={N} | CV+ (K = 10)/half_width"] / small[f"N={N} | split 50/50/half_width"], 0.5, 0.97
            yield "nb2", f"... that vary much less from one data set to the next (N = {N})", \
                small[f"N={N} | CV+ (K = 10)/half_width_sd"] / small[f"N={N} | split 50/50/half_width_sd"], 0.2, 0.8
        yield "nb2", "from about a thousand reactions on, the two are practically equivalent", \
            small["N=1000 | CV+ (K = 10)/half_width"] / small["N=1000 | split 50/50/half_width"], 0.95, 1.03
        yield "nb2", "with 100 reference barriers ... about ±6 kcal/mol", small["N=100 | CV+ (K = 10)/half_width"], 5.3, 6.5
        yield "nb2", "with 1000 about ±5", small["N=1000 | CV+ (K = 10)/half_width"], 4.4, 5.3

    # ------------------------------------------------------------------------------------ Notebook 3
    if n3:
        yield "nb3", "nearly all reactions with a carbene product are covered", n3["class_coverage_single_split/carbene"], 0.95, 1.0
        yield "nb3", "roughly one skeletal rearrangement in five falls outside", n3["class_coverage_single_split/skeletal"], 0.74, 0.85
        yield "nb3", "the in-sample residuals are about half as large", n3["sigma/in_sample_mae"] / n3["sigma/oof_mae"], 0.42, 0.66
        if n1:
            yield "nb3", "out-of-fold residuals have the same size as the held-out errors", n3["sigma/oof_mae"] / n1["delta/errors/MAE"], 0.93, 1.08
        yield "nb3", "the mean error grows about ninefold", n3["sigma/tenth_error/9"] / n3["sigma/tenth_error/0"], 7, 11
        yield "nb3", "sigma follows it closely", max(abs(n3[f"sigma/tenth_sigma/{i}"] / n3[f"sigma/tenth_error/{i}"] - 1) for i in range(10)), 0, 0.2
        yield "nb3", "the rank correlation for individual reactions is moderate", n3["sigma/spearman"], 0.4, 0.7
        yield "nb3", "the raw band is too narrow", n3["raw_quantile_band_coverage"], 0.6, 0.86
        yield "nb3", "about two thirds of the reactions get a narrower interval", n3["fraction_narrower_than_split"], 0.6, 0.74
        for method in ("split", "normalized", "CQR"):
            yield "nb3", f"all three methods have 90% coverage overall ({method})", n3[f"summary/{method}/coverage"], 0.895, 0.906
        for method in ("normalized", "CQR"):
            yield "nb3", f"smaller mean half-width than the constant interval ({method})", n3[f"summary/{method}/mean half-width"] / n3["summary/split/mean half-width"], 0.7, 0.97
            yield "nb3", f"a much smaller median half-width ({method})", n3[f"summary/{method}/median half-width"] / n3["summary/split/median half-width"], 0.5, 0.82
            yield "nb3", f"a clearly better interval score ({method})", n3[f"summary/{method}/interval score"] / n3["summary/split/interval score"], 0.6, 0.88
        yield "nb3", "the constant interval covers only about 80% of the skeletal rearrangements", n3["summary/split/class: skeletal"], 0.77, 0.83
        yield "nb3", "and about 97% of the reactions with a carbene product", n3["summary/split/class: carbene"], 0.955, 0.985
        yield "nb3", "the normalized interval brings every class to within about 1.5 points of 90%", \
            max(abs(n3[f"summary/normalized/class: {c}"] - 0.9) for c in CLASSES), 0, 0.019
        yield "nb3", "the constant interval covers practically all of the easiest fifth", n3["summary/split/sigma fifth 1"], 0.985, 1.0
        yield "nb3", "and only about two thirds of the hardest fifth", n3["summary/split/sigma fifth 5"], 0.62, 0.72
        for j in range(1, 6):
            yield "nb3", f"the normalized interval stays between about 88% and 93% (fifth {j})", n3[f"summary/normalized/sigma fifth {j}"], 0.872, 0.936
        yield "nb3", "CQR still over-covers easy reactions", n3["summary/CQR/sigma fifth 1"], 0.93, 1.0
        yield "nb3", "and under-covers hard reactions", n3["summary/CQR/sigma fifth 5"], 0.7, 0.875
        yield "nb3", "for CQR the wide intervals are covered less often than the narrow ones", \
            n3["summary/CQR/own-width fifth 1"] - n3["summary/CQR/own-width fifth 5"], 0.05, 0.3
        yield "nb3", "in-sample residuals: about 95% for the easiest fifth", n3["pitfall_in_sample/in-sample residuals/easiest fifth"], 0.935, 0.975
        yield "nb3", "in-sample residuals: about 85% for the hardest fifth", n3["pitfall_in_sample/in-sample residuals/hardest fifth"], 0.82, 0.875
        yield "nb3", "in-sample residuals: skeletal rearrangements fall back to about 87%", n3["pitfall_in_sample/in-sample residuals/skeletal rearrangements"], 0.835, 0.885
        yield "nb3", "the widest intervals belong to the largest errors", n3["widest_tenth_error/widest"] / n3["widest_tenth_error/others"], 2, 5

    # ------------------------------------------------------------------------------------ Notebook 4
    if n4:
        yield "nb4", "reactions with a carbene product need roughly ±2 kcal/mol", n4["single_split/q_class/carbene"], 1.8, 2.7
        yield "nb4", "skeletal rearrangements roughly ±6 to ±7 kcal/mol", n4["single_split/q_class/skeletal"], 5.6, 7.4
        for c in CLASSES:
            yield "nb4", f"with these half-widths every class is covered at close to 90% ({c})", n4[f"single_split/coverage_mondrian/{c}"], 0.86, 0.94
            yield "nb4", f"Mondrian, mean over 200 splits ({c})", n4[f"average/Mondrian/{c}"], 0.895, 0.91
        yield "nb4", "the smallest class has only about 140 calibration reactions", n4["class_spread/skeletal/mean n_g"], 130, 150
        yield "nb4", "with 50 calibration reactions: an infinite interval in almost every split", n4["budget/50/splits with an infinite interval"], 0.9, 1.0
        wider = sum(n4[f"class_spread/{c}/97.5%"] - n4[f"class_spread/{c}/2.5%"]
                    > n4[f"class_spread/{c}/Beta law 97.5%"] - n4[f"class_spread/{c}/Beta law 2.5%"] + 0.003 for c in CLASSES)
        yield "nb4", "for some classes the observed range is a little wider than the Beta law predicts", wider, 1, 5
        yield "nb4", "with 100 this still happens now and then", n4["budget/100/splits with an infinite interval"], 0.005, 0.15
        yield "nb4", "from 200 on the intervals are finite", n4["budget/200/splits with an infinite interval"], 0, 0
        yield "nb4", "only a few families have enough calibration reactions", n4["families/n_finite"], 3, 7
        yield "nb4", "some families of the test set never occur in the calibration set", n4["families/unseen_in_cal"], 1, 6
        for g in ("(no family)", "(other families)", "ketoenol"):
            yield "nb4", f"three categories survive; each is covered at 90% on average ({g})", n4[f"merged/{g}/coverage (mean)"], 0.893, 0.915
        yield "nb4", "known templates: about ±2 to ±3 kcal/mol (other families)", n4["merged/(other families)/half-width (mean)"], 1.5, 3.2
        yield "nb4", "known templates: about ±2 to ±3 kcal/mol (ketoenol)", n4["merged/ketoenol/half-width (mean)"], 1.8, 3.3
        yield "nb4", "the remainder: about ±4 kcal/mol", n4["merged/(no family)/half-width (mean)"], 3.7, 4.4
        yield "nb4", "Mondrian guarantees equal treatment of the classes (worst class)", n4["combined/Mondrian/worst-covered class"], 0.895, 0.91
        yield "nb4", "the normalized score comes close without a guarantee", n4["combined/normalized/worst-covered class"], 0.875, 0.91
        yield "nb4", "the combination does well on all three counts (worst class)", n4["combined/Mondrian + normalized/worst-covered class"], 0.893, 0.91
        yield "nb4", "the combination does well on all three counts (easiest fifth)", n4["combined/Mondrian + normalized/easiest fifth"], 0.875, 0.93
        yield "nb4", "the combination does well on all three counts (hardest fifth)", n4["combined/Mondrian + normalized/hardest fifth"], 0.875, 0.94
        yield "nb4", "and it keeps the small median half-width of the normalized score", \
            n4["combined/Mondrian + normalized/median half-width"] / n4["combined/normalized/median half-width"], 0.95, 1.06

    # ------------------------------------------------------------------------------------ Notebook 5
    if n5:
        yield "nb5", "roughly one in seven in the calibration set", n5["single_example/share_hard_cal"], 0.125, 0.165
        yield "nb5", "covers clearly less than 90% of the reactions in the new project", n5[f"mix/{D} | split conformal (no repair)/coverage"], 0.80, 0.875
        for method in ("weighted, class-frequency ratios", "Mondrian by class"):
            yield "nb5", f"class-frequency weights and Mondrian calibration bring the coverage back to 90% ({method})", \
                n5[f"mix/{D} | {method}/coverage"], 0.893, 0.910
        learned = n5[f"mix/{D} | weighted, learned weights/coverage"]
        learned_se = n5[f"mix/{D} | weighted, learned weights/s.d. of coverage"] / math.sqrt(40)
        yield "nb5", "so do the learned weights", learned, 0.889, 0.911
        yield "nb5", "... within the noise of their 40 repetitions (distance from 90% in standard errors)", abs(learned - 0.90) / learned_se, 0, 2.6
        yield "nb5", "the standard error of that mean is about 0.3 percentage points", 100 * learned_se, 0.22, 0.42
        for method in ("weighted, class-frequency ratios", "weighted, learned weights", "Mondrian by class"):
            yield "nb5", f"all repairs pay in the same way: the intervals are wider ({method})", \
                n5[f"mix/{D} | {method}/mean half-width"] / n5[f"mix/{D} | split conformal (no repair)/mean half-width"], 1.15, 1.6
        yield "nb5", "one calibration set and one deployment set: such coverages scatter by several percentage points (2 s.d.)", \
            2 * n5[f"mix/{D} | weighted, class-frequency ratios/s.d. of coverage"], 0.025, 0.06
        yield "nb5", "1500 calibration reactions are worth only a few hundred", n5["mix_diagnostics/effective sample size"], 250, 750
        for model in (S, D):
            unrepaired = n5[f"mix/{model} | split conformal (no repair)/s.d. of coverage"]
            for method in ("weighted, class-frequency ratios", "weighted, learned weights", "Mondrian by class"):
                if model == S and method == "weighted, learned weights":
                    continue
                yield "nb5", f"the standard deviations of the repaired rows are larger than that of the unrepaired row ({model}, {method})", \
                    n5[f"mix/{model} | {method}/s.d. of coverage"] / unrepaired, 1.0001, 2.0
            for method in ("split conformal (no repair)", "weighted, class-frequency ratios"):
                p = n5[f"mix/{model} | {method}/coverage"]
                yield "nb5", f"the sampling noise of the 600 deployment reactions, about 1.2 to 1.5 percentage points ({model}, {method})", \
                    100 * math.sqrt(p * (1 - p) / 600), 1.15, 1.55
        for model in (S, D):
            yield "nb5", f"larger molecules: short by two to three percentage points ({model})", 0.9 - n5[f"size/{model}/coverage on larger molecules"], 0.01, 0.045
        yield "nb5", "the classifier separates the two sets perfectly", n5["size_auc"], 0.995, 1.0
        yield "nb5", "weighted conformal prediction returns an infinite interval", n5["size_weighted_infinite"], 0.99, 1.0
        yield "nb5", "with 20 calculations ... between about 78%", n5["recalibration/20/5%"], 0.75, 0.81
        yield "nb5", "... and 98%", n5["recalibration/20/95%"], 0.97, 0.995
        yield "nb5", "with 200 between about 86%", n5["recalibration/200/5%"], 0.85, 0.87
        yield "nb5", "... and 93%", n5["recalibration/200/95%"], 0.925, 0.945
        unseen = [n5[f"unseen_class/{model} | {c}/coverage on the unseen class"] for model in (S, D) for c in CLASSES]
        yield "nb5", "unseen class: between about one half ...", min(unseen), 0.42, 0.6
        yield "nb5", "... and 86%", max(unseen), 0.82, 0.895
        ratio = max(n5[f"unseen_class/{model} | {c}/MAE, unseen class"] / n5[f"unseen_class/{model} | {c}/MAE, known classes"] for model in (S, D) for c in CLASSES)
        yield "nb5", "errors up to almost two and a half times larger", ratio, 2.0, 2.7
        for model in (S, D):
            for c in CLASSES:
                yield "nb5", f"fifty reference calculations restore the coverage ({model}, {c})", n5[f"unseen_class/{model} | {c}/coverage after 50 reference calculations"], 0.89, 0.915
            yield "nb5", f"known reactants: a loss of about half a percentage point, 0.5 to 0.6 points ({model})", n5[f"dependence/{model}/loss"], 0.45, 0.65
            yield "nb5", f"... more than three times the standard error of the mean ({model})", \
                n5[f"dependence/{model}/loss"] / n5[f"dependence/{model}/loss_se"], 3.0, 20
            yield "nb5", f"the half-width is about 2% too small ({model})", n5[f"dependence/{model}/width_ratio"], 0.96, 0.995
            yield "nb5", f"a single repetition scatters by about one percentage point ({model})", n5[f"dependence/{model}/loss_se"] * 60 ** 0.5, 0.7, 1.5
            yield "nb5", f"clusters: still 90% on average ({model})", n5[f"clusters/{model}/mean coverage, whole reactants"], 0.895, 0.905
        yield "nb5", "but it varies more than the Beta law predicts", n5["clusters_extra_scatter"], 0.08, 0.5
        yield "nb5", "AUC near 0.5 without a shift", n5["auc/same population (no shift)/AUC of the domain classifier"], 0.44, 0.56
        yield "nb5", "AUC between 0.5 and about 0.9: the sets differ but overlap", n5["auc/different mix of classes/AUC of the domain classifier"], 0.58, 0.8
        yield "nb5", "AUC close to 1 for the unseen class", n5["auc/unseen class (skeletal)/AUC of the domain classifier"], 0.99, 1.0
        yield "nb5", "the two shifts with an AUC of 1 cost very different amounts of coverage", \
            n5["auc/larger molecules/coverage of the Δ-learning intervals"] - n5["auc/unseen class (skeletal)/coverage of the Δ-learning intervals"], 0.1, 0.4
        for m in (10, 20, 50, 100, 200, 500):
            yield "nb5", f"without a shift the alarm rings in at most 5% of the cases (exact, m = {m})", n5[f"alarm_exact/{m}"], 0.0, 0.05
            yield "nb5", f"... and the simulation scatters around it (m = {m})", n5[f"alarm/same population (no shift)/{m}"] - n5[f"alarm_exact/{m}"], -0.02, 0.02
        yield "nb5", "a drastic shift is caught with 20 ...", n5["alarm/unseen class (skeletal)/20"], 0.75, 1.0
        yield "nb5", "... to 50 reference calculations", n5["alarm/unseen class (skeletal)/50"], 0.9, 1.0
        yield "nb5", "a loss of a few points is still missed almost half of the time with 500", n5["alarm/larger molecules/500"], 0.45, 0.62
        yield "nb5", "inspected again and again, false alarms become more frequent", n5["alarm_repeated_looks"], 0.08, 0.2

    # ------------------------------------------------------------------------------------ Notebook 6
    if n6:
        yield "nb6", "a half-life of a few milliseconds at 1000 K", n6["eyring_half_life_ms"], 2.7, 2.9
        yield "nb6", "in RDB7, reactions below 50 kcal/mol are rare", n6["hit_rate_hold"], 0.05, 0.09
        yield "nb6", "the first 800 of the 4935 candidates", n6["n_candidates"], 4935, 4935
        yield "nb6", "rule A, Δ-learning: only a few percent false", n6[f"single_split/{D} | A: predicted barrier below c/FDP"], 0.01, 0.08
        yield "nb6", "rule A, structure-only: more than one in five", n6[f"single_split/{S} | A: predicted barrier below c/FDP"], 0.2, 0.3
        yield "nb6", "rule B loses about 30% of the hits with the Δ-learning model", 1 - n6[f"single_split/{D} | B: 90% interval entirely below c/power"], 0.25, 0.35
        yield "nb6", "rule B: practically all of them with the structure-only model", n6[f"single_split/{S} | B: 90% interval entirely below c/power"], 0, 0.02
        for model in (S, D):
            yield "nb6", f"rule C: several hundred false discoveries ({model})", n6[f"single_split/{model} | C: p-value below 0.10 for each reaction/false discoveries"], 300, 800
        for model in ("structure", "delta"):
            for level in ("0.02", "0.05", "0.10", "0.20", "0.30"):
                yield "nb6", f"conformal selection: FDR at or below the target ({model}, q = {level})", \
                    n6[f"experiment/{model}/conformal selection | q={level}/FDR (mean FDP)"] - float(level), -1.0, min(0.006, 0.1 * float(level))
                yield "nb6", f"rule C exceeds every target by a wide margin ({model}, q = {level})", \
                    n6[f"experiment/{model}/C: p-value below q, no correction | q={level}/FDR (mean FDP)"] / float(level), 2.0, 30
        yield "nb6", "the Δ-learning model finds more than 90% of the hits already at q = 5%", n6["experiment/delta/conformal selection | q=0.05/power (mean)"], 0.9, 1.0
        yield "nb6", "the structure-only model finds less than a quarter of the hits at q = 10%", n6["experiment/structure/conformal selection | q=0.10/power (mean)"], 0.0, 0.25
        yield "nb6", "and needs q = 30% to find more than half of them", n6["experiment/structure/conformal selection | q=0.30/power (mean)"], 0.5, 0.75
        yield "nb6", "rule A delivers more than 20% false discoveries for the structure-only model", n6["experiment/structure/A: predicted barrier below c/FDR (mean FDP)"], 0.2, 0.3
        for model in (S, D):
            yield "nb6", f"at the same error rate rule A finds slightly more hits ({model})", \
                n6[f"versus_rule_a/{model}/power of rule A"] - n6[f"versus_rule_a/{model}/its power"], 0.0, 0.05
            yield "nb6", f"... 'the same error rate' ({model})", \
                abs(n6[f"versus_rule_a/{model}/FDR of rule A"] - n6[f"versus_rule_a/{model}/FDR of conformal selection at q = FDR of rule A"]), 0.0, 0.01
        yield "nb6", "an individual list can contain a larger fraction of false entries than q", n6["experiment/delta/conformal selection | q=0.10/FDP, 90th percentile"], 0.1, 0.25
        yield "nb6", "structure-only model at q = 10%: the list is empty in about a third of the splits", \
            n6["experiment/structure/conformal selection | q=0.10/splits with an empty list"], 0.25, 0.45
        yield "nb6", "... and the other lists contain 12% non-hits on average", \
            n6["experiment/structure/conformal selection | q=0.10/mean FDP of the non-empty lists"], 0.105, 0.14
        yield "nb6", "at q = 5%: empty in 85% of the splits", \
            n6["experiment/structure/conformal selection | q=0.05/splits with an empty list"], 0.78, 0.92
        yield "nb6", "... 13% non-hits in the others", \
            n6["experiment/structure/conformal selection | q=0.05/mean FDP of the non-empty lists"], 0.105, 0.155
        for level in ("0.05", "0.10", "0.20", "0.30"):
            yield "nb6", f"the list of the Δ-learning model is practically never empty (q = {level})", \
                n6[f"experiment/delta/conformal selection | q={level}/splits with an empty list"], 0.0, 0.01
        yield "nb6", "with 100 calibration reactions a list needs at least 578 entries", n6[f"resolution/{D} | n=100/smallest possible list m/(q(n+1))"], 578, 578
        for model in (S, D):
            yield "nb6", f"the list is then empty in the great majority of the splits ({model})", n6[f"resolution/{model} | n=100/splits with an empty list"], 0.85, 1.0
        yield "nb6", "from 200 on the Δ-learning model finds most hits", n6[f"resolution/{D} | n=200/power"], 0.8, 1.0
        yield "nb6", "for the structure-only model a larger calibration set keeps helping", \
            n6[f"resolution/{S} | n=2000/power"] - n6[f"resolution/{S} | n=500/power"], 0.02, 0.5
        yield "nb6", "the normalized score finds clearly more hits at the strict levels (structure-only, q = 10%)", \
            n6[f"scores/{S} | q=0.10 | power | (mu - c) / sigma"] - n6[f"scores/{S} | q=0.10 | power | mu - c"], 0.03, 0.4
        yield "nb6", "for the Δ-learning model there is little left to gain", \
            abs(n6[f"scores/{D} | q=0.10 | power | (mu - c) / sigma"] - n6[f"scores/{D} | q=0.10 | power | mu - c"]), 0, 0.03


def run(out_dir=OUT_DIR, verbose=True):
    results = load(out_dir)
    missing = [name for name in P.NOTEBOOKS if name not in results]
    Numbers.absent = []
    failed, n = [], 0
    for notebook, sentence, value, low, high in claims(results):
        n += 1
        if not low <= value <= high:                  # also true if the value is not a number
            failed.append((notebook, sentence, value, low, high))
    if verbose:
        for notebook, sentence, value, low, high in failed:
            shown = "not found in the results" if value != value else f"{value:.4g} is not in [{low}, {high}]"
            print(f"FAILED  {notebook}: {sentence}: {shown}")
        if Numbers.absent:
            print("numbers that the results do not contain:", ", ".join(sorted(set(Numbers.absent))))
        if missing:
            print("no results found for:", ", ".join(missing), "(run those notebooks first)")
        print(f"{n - len(failed)} of {n} quoted numbers agree with the results in {os.path.relpath(out_dir)}")
    return failed, missing


def test_quoted_numbers_agree_with_the_results():
    failed, missing = run(verbose=False)
    if len(missing) == len(P.NOTEBOOKS):          # nothing has been run yet: nothing to compare
        return
    assert not failed, failed[:5]


if __name__ == "__main__":
    failed, missing = run(sys.argv[1] if len(sys.argv) > 1 else OUT_DIR)
    sys.exit(1 if failed or missing else 0)
