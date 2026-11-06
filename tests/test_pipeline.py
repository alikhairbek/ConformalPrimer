"""
Tests of the data pipeline (need the RDB7 tables; they are downloaded on first use).

    python -m pytest tests/test_pipeline.py      (or:  python tests/test_pipeline.py)
"""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
import primer as P  # noqa: E402

_rdb7 = None


def problem():
    global _rdb7
    if _rdb7 is None:
        root = os.path.join(HERE, "..")
        _rdb7 = P.load_problem(os.path.join(root, "data"), os.path.join(root, "cache"), verbose=False)
    return _rdb7


def test_tables_are_the_zenodo_tables():
    rdb7 = problem()
    assert rdb7.n == 11926
    assert rdb7.df.attrs["fingerprints"] == P.REFERENCE_FINGERPRINTS
    assert all(verdict.endswith("the Zenodo file") and not verdict.startswith("DIFFERENT")
               for verdict in rdb7.df.attrs["data_status"].values())


def test_energies_do_not_depend_on_the_digits_in_the_file():
    # every energy is rounded to 1e-5 kcal/mol, so copies of the files with more digits give the same numbers
    rdb7 = problem()
    for column in ("barrier_cc", "dH_cc", "barrier_dft", "dH_dft"):
        values = rdb7.df[column].to_numpy(float)
        values = values[np.isfinite(values)]
        assert all(float(f"{v:.5f}") == v for v in values), column


def test_split_is_by_reactant():
    rdb7 = problem()
    assert len(set(rdb7.train) & set(rdb7.hold)) == 0 and len(rdb7.train) + len(rdb7.hold) == rdb7.n
    assert not set(rdb7.reactant[rdb7.train]) & set(rdb7.reactant[rdb7.hold])        # no reactant on both sides
    assert abs(len(rdb7.train) / rdb7.n - 0.5) < 0.02


def test_grouped_folds_keep_reactants_together():
    rdb7 = problem()
    groups = rdb7.reactant[rdb7.train]
    fold = P.grouped_folds(groups, 5, P.SEED)
    assert set(fold) == {0, 1, 2, 3, 4}
    assert all(len(set(fold[groups == g])) == 1 for g in np.unique(groups)[:200])
    assert np.array_equal(fold, P.grouped_folds(groups, 5, P.SEED))                  # reproducible


def test_every_reaction_has_one_class():
    rdb7 = problem()
    assert set(rdb7.rxn_class) == set(P.CLASS_ORDER)
    assert all(np.sum(rdb7.rxn_class == c) > 1000 for c in P.CLASS_ORDER)


def test_class_rules_on_known_reactions():
    def classify(rsmi, psmi):
        return P.reaction_class(P.describe_reaction(rsmi, psmi)[0])

    rdb7 = problem()
    # loss of carbon monoxide is a fragmentation although CO is written [C-]#[O+]
    with_co = [i for i, p in enumerate(rdb7.df.psmi) if "[C-:" in p and "#[O+:" in p and rdb7.desc["n_products"][i] >= 2
               and not rdb7.desc["zwitterion_product"][i] and not rdb7.desc["carbene_product"][i]]
    assert len(with_co) > 100 and all(rdb7.rxn_class[i] == "fragmentation" for i in with_co)
    # the keto-enol tautomerization of butanal, a ring opening of bicyclo[2.1.1]hexane, the dehydration of cyclopentanol
    assert rdb7.rxn_class[75] == "H-transfer isomerization" and P.readable(rdb7.df.rsmi[75]) == "CCCC=O"
    assert rdb7.rxn_class[8] == "skeletal rearrangement"
    assert rdb7.rxn_class[21] == "fragmentation"
    assert rdb7.rxn_class[7] == "carbene product"
    assert rdb7.rxn_class[0] == "zwitterionic product"
    # the charge rule itself
    from rdkit import Chem
    assert not P.has_charge_separation(Chem.MolFromSmiles("[C-]#[O+]"))              # carbon monoxide
    assert not P.has_charge_separation(Chem.MolFromSmiles("C[N+]#[C-]"))             # methyl isocyanide
    assert P.has_charge_separation(Chem.MolFromSmiles("C=[N+]=[N-]"))                # diazomethane
    assert not P.has_charge_separation(Chem.MolFromSmiles("CC=O"))


def test_what_the_classes_contain():
    # the statements of the class table in Notebook 1
    import collections

    from rdkit import Chem
    from rdkit.Chem.rdMolDescriptors import CalcMolFormula

    rdb7 = problem()
    desc, classes = rdb7.desc, rdb7.rxn_class
    heavy = [c for c in desc.columns if c.split("_")[0] in ("broken", "formed") and "H" not in c.split("_")[1].split("-")]
    heavy_change = (desc[heavy].sum(axis=1) > 0).to_numpy()

    # "H-transfer isomerization: ... in six cases out of ten combined with a change of the heavy-atom skeleton"
    h_transfer = classes == "H-transfer isomerization"
    assert np.all(desc["n_XH_broken"][h_transfer] > 0) and np.all(desc["n_products"][h_transfer] == 1)
    assert 0.57 < heavy_change[h_transfer].mean() < 0.67
    # "skeletal rearrangement: one product; no X-H bond is broken"
    skeletal = classes == "skeletal rearrangement"
    assert np.all(desc["n_XH_broken"][skeletal] == 0) and np.all(desc["n_products"][skeletal] == 1) and heavy_change[skeletal].all()

    # "fragmentation: loss of H2, H2O, CH2O, C2H4, CH4 or CO in seven cases out of ten"
    def smallest_product(psmi):
        formulas = []
        for fragment in Chem.GetMolFrags(P.parse_mapped_smiles(psmi), asMols=True, sanitizeFrags=False):
            fragment.UpdatePropertyCache(strict=False)
            formulas.append((sum(a.GetAtomicNum() > 1 for a in fragment.GetAtoms()), CalcMolFormula(fragment)))
        return min(formulas)[1]

    fragmentations = np.flatnonzero(classes == "fragmentation")
    assert np.all(desc["n_products"][fragmentations] >= 2)
    lost = collections.Counter(smallest_product(rdb7.df.psmi[i]) for i in fragmentations)
    assert [formula for formula, _ in lost.most_common(6)] == ["H2", "H2O", "CH2O", "C2H4", "CH4", "CO"]
    assert 0.66 < sum(count for _, count in lost.most_common(6)) / len(fragmentations) < 0.76

    # "carbene product: ... in about 2% of the cases a nitrene or two radical centers instead"
    carbenes = np.flatnonzero(classes == "carbene product")
    one_divalent_carbon = 0
    for i in carbenes:
        radicals = [(a.GetSymbol(), a.GetNumRadicalElectrons()) for a in P.parse_mapped_smiles(rdb7.df.psmi[i]).GetAtoms()
                    if a.GetNumRadicalElectrons() > 0]
        one_divalent_carbon += radicals == [("C", 2)]
    assert 0.97 < one_divalent_carbon / len(carbenes) < 0.99

    # "a reaction family for about 12% of the reactions"; the family used in Notebook 4 is the largest one
    families = rdb7.df.rmg_family
    assert 0.115 < families.notna().mean() < 0.13 and families.value_counts().index[0] == "ketoenol"
    # "about half of its 687 reactions shift a hydrogen atom and the other half a heavy-atom group"
    ketoenol = (families == "ketoenol").to_numpy()
    assert ketoenol.sum() == 687
    assert 0.45 < np.mean(desc["n_XH_broken"][ketoenol] > 0) < 0.56

    # "the loss of CO alone is a fragmentation"; "hydrogen isocyanide ... counts as a carbene product"
    def product_smiles(psmi):
        out = []
        for fragment in Chem.GetMolFrags(P.parse_mapped_smiles(psmi), asMols=True, sanitizeFrags=False):
            for atom in fragment.GetAtoms():
                atom.SetAtomMapNum(0)
            out.append(Chem.MolToSmiles(Chem.RemoveHs(fragment, sanitize=False)))
        return out

    products = [product_smiles(p) for p in rdb7.df.psmi]
    loses_co = np.array(["[C-]#[O+]" in p and len(p) >= 2 for p in products])
    only_co_is_charged = loses_co & ~np.asarray(desc["zwitterion_product"], bool) & ~np.asarray(desc["carbene_product"], bool)
    assert loses_co.sum() > 200 and np.all(classes[only_co_is_charged] == "fragmentation")
    assert np.mean(classes[loses_co] == "fragmentation") > 0.8
    hnc = np.array([any(f in ("[C]=N", "N=[C]") for f in p) for p in products])
    assert hnc.sum() > 30 and np.all(classes[hnc] == "carbene product")


def test_what_the_zwitterion_class_contains():
    # the zwitterion row of the class table in Notebook 1: three groups of about one third each
    from rdkit import Chem

    rdb7 = problem()

    def charge_pattern(psmi):
        mol = P.parse_mapped_smiles(psmi)
        dative = set()                                     # the conventional notation of CO and isocyanides does not count
        for bond in mol.GetBonds():
            a, b = bond.GetBeginAtom(), bond.GetEndAtom()
            for carbon, other in ((a, b), (b, a)):
                if (carbon.GetSymbol() == "C" and carbon.GetFormalCharge() == -1 and other.GetFormalCharge() == 1
                        and other.GetSymbol() in ("O", "N") and bond.GetBondTypeAsDouble() == 3.0):
                    dative.update((carbon.GetIdx(), other.GetIdx()))
        charged = [atom for atom in mol.GetAtoms() if atom.GetFormalCharge() != 0 and atom.GetIdx() not in dative]
        if len(charged) != 2 or charged[0].GetFormalCharge() + charged[1].GetFormalCharge() != 0:
            return "other"
        neg, pos = sorted(charged, key=lambda atom: atom.GetFormalCharge())
        distance = int(Chem.GetDistanceMatrix(mol)[neg.GetIdx(), pos.GetIdx()])
        if distance == 1:
            bond = mol.GetBondBetweenAtoms(neg.GetIdx(), pos.GetIdx())
            if neg.GetSymbol() == "C" and neg.GetDegree() == 2 and pos.GetSymbol() in ("O", "N") and bond.GetBondTypeAsDouble() == 2.0:
                return "carbene drawn as ylide"
            return "charges on neighboring atoms"
        return "charges farther apart"

    rows = np.flatnonzero(rdb7.rxn_class == "zwitterionic product")
    pattern = np.array([charge_pattern(rdb7.df.psmi[i]) for i in rows])
    assert 0.27 < np.mean(pattern == "carbene drawn as ylide") < 0.35          # "three cases in ten"
    assert 0.29 < np.mean(pattern == "charges on neighboring atoms") < 0.37    # "one third"
    assert 0.31 < np.mean(pattern == "charges farther apart") < 0.39           # "one third"
    assert np.mean(pattern == "other") < 0.03


def test_products_that_are_training_reactants():
    # "in about 70 held-out reactions the product is a reactant of the training half, and about 20 of these are
    #  the exact reverse of a training reaction"
    rdb7 = problem()
    reactant, product = rdb7.df.rinchi.to_numpy(), rdb7.df.pinchi.to_numpy()
    training_reactants = set(reactant[rdb7.train])
    training_reactions = set(zip(reactant[rdb7.train], product[rdb7.train]))
    assert 55 <= sum(product[i] in training_reactants for i in rdb7.hold) <= 85
    assert 12 <= sum((product[i], reactant[i]) in training_reactions for i in rdb7.hold) <= 28


def test_ring_count_is_the_number_of_independent_rings():
    from rdkit import Chem

    for smiles, rings in (("CCO", 0), ("C1CC1", 1), ("c1ccccc1", 1), ("C1CC2CC12", 2), ("C12C3C4C1C5C2C3C45", 5), ("C1CC1.C1CC1", 2)):
        assert P.count_rings(Chem.MolFromSmiles(smiles)) == rings, smiles


def test_models_do_not_see_the_held_out_half():
    rdb7 = problem()
    for name, mod in rdb7.models.items():
        assert np.array_equal(mod.train, rdb7.train)
        error = np.abs(rdb7.y - mod.mu)
        assert error[rdb7.train].mean() < 0.7 * error[rdb7.hold].mean()               # the fit is tighter on what it has seen
    assert np.abs(rdb7.y - rdb7.models["delta"].mu)[rdb7.hold].mean() < 2.0           # kcal/mol
    assert np.abs(rdb7.y - rdb7.models["structure"].mu)[rdb7.hold].mean() < 10.0


def test_difficulty_estimate_is_positive_and_informative():
    rdb7 = problem()
    sigma = P.fit_difficulty(rdb7, "delta", verbose=False)
    error = np.abs(rdb7.y - rdb7.models["delta"].mu)[rdb7.hold]
    assert sigma.min() > 0
    order = np.argsort(sigma[rdb7.hold])
    assert error[order[-500:]].mean() > 3 * error[order[:500]].mean()


def test_difficulty_cache_follows_the_settings():
    # changing a constant must give a new result, not the cached one
    rdb7 = problem()
    sigma = P.fit_difficulty(rdb7, "delta", verbose=False)
    old = P.DIFFICULTY_FLOOR
    try:
        P.DIFFICULTY_FLOOR = 0.05
        changed = P.fit_difficulty(rdb7, "delta", verbose=False)
    finally:
        P.DIFFICULTY_FLOOR = old
    assert changed.min() < sigma.min()
    assert np.array_equal(P.fit_difficulty(rdb7, "delta", verbose=False), sigma)


def test_reference_check_is_strict():
    saved = dict(P.REFERENCE_RESULTS)
    try:
        P.REFERENCE_RESULTS["demo"] = {"n_reactions": 11926, "coverage": 0.9, "half_width": 3.8, "selected": 56, "q_small": None}
        same = {"n_reactions": 11926, "coverage": 0.9, "half_width": 3.8, "selected": 56, "q_small": np.inf}
        assert len(P.compare("demo", same)["identical"]) == 5
        close = P.compare("demo", {"n_reactions": 11926, "coverage": 0.915, "half_width": 3.9, "selected": 58, "q_small": np.inf})
        assert sorted(close["close"]) == ["coverage", "half_width", "selected"] and not close["different"] and not close["missing"]
        bad = P.compare("demo", {"n_reactions": 11400, "coverage": 0.86, "half_width": np.nan, "q_small": 2.0})
        assert sorted(bad["different"]) == ["coverage", "n_reactions", "q_small"]     # a count must be identical; 0.04 is too far
        assert sorted(bad["missing"]) == ["half_width", "selected"]                   # not finite, or absent
        assert P.check("demo", {"n_reactions": 11400}) is False
    finally:
        P.REFERENCE_RESULTS.clear()
        P.REFERENCE_RESULTS.update(saved)


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
