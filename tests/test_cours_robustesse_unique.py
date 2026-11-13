"""Spearman (p. 77–79), sensibilité, alternative unique : MAUT, seuil ROC, distance à l'idéal (p. 81–89)."""

import numpy as np
import pytest

from core.alternative_unique.distance_ideal import distance_ideal
from core.alternative_unique.maut import maut, scores_maut, utilites
from core.alternative_unique.seuil import courbe_roc, indicateurs, seuil_optimal
from core.commun import rangs, rangs_moyens
from core.robustesse.sensibilite import balayage_poids, poids_modifies
from core.robustesse.spearman import (VALEURS_CRITIQUES_5PCT, matrice_spearman, spearman,
                                      valeur_critique_exacte)

TOPSIS = [2, 1, 3, 5, 4, 6]
VIKOR = [1, 2, 4, 3, 5, 6]
PROMETHEE = [2, 1, 3, 6, 4, 5]


def test_spearman_p79():
    assert spearman(TOPSIS, VIKOR) == pytest.approx(0.77, abs=0.005)
    assert spearman(TOPSIS, PROMETHEE) == pytest.approx(0.94, abs=0.005)
    assert spearman(VIKOR, PROMETHEE) == pytest.approx(0.60, abs=0.005)
    noms, M = matrice_spearman({"T": TOPSIS, "V": VIKOR, "P": PROMETHEE})
    np.testing.assert_allclose(np.diag(M), 1.0)


def test_spearman_valeur_critique_n6_p79():
    assert valeur_critique_exacte(6) == pytest.approx(0.886, abs=5e-4)
    assert VALEURS_CRITIQUES_5PCT[6] == pytest.approx(0.886, abs=5e-4)


@pytest.mark.parametrize("n", [4, 5, 6, 7, 8])
def test_spearman_table_figee_egale_au_recalcul(n):
    attendu = valeur_critique_exacte(n)
    figee = VALEURS_CRITIQUES_5PCT[n]
    assert (attendu is None and figee is None) or attendu == pytest.approx(figee, abs=1e-4)


def test_spearman_vikor_change_5_rangs_sur_6_p78():
    assert sum(a != b for a, b in zip(TOPSIS, VIKOR)) == 5


def test_rangs_et_exaequo():
    assert list(rangs([0.3, 0.9, 0.9, 0.1])) == [3, 1, 1, 4]
    assert list(rangs_moyens([0.3, 0.9, 0.9, 0.1])) == [3, 1.5, 1.5, 4]
    assert list(rangs([1, 0, 1], plus_grand_meilleur=False)) == [2, 1, 2]


def test_sensibilite_poids_modifies_somme_un():
    w = np.array([0.5, 0.3, 0.2])
    for t in (0.0, 0.25, 1.0):
        n = poids_modifies(w, 1, t)
        assert n.sum() == pytest.approx(1.0) and n[1] == pytest.approx(t)
    np.testing.assert_allclose(poids_modifies(w, 1, 0.3), w)        # point actuel inchangé
    ts, S, Rg = balayage_poids(w, 0, lambda v: np.array([v @ [1, 0, 0], v @ [0, 1, 1]]), nb=5)
    assert S.shape == (5, 2) and Rg[0, 1] == 1 and Rg[-1, 0] == 1


# ---------------------------------------------------------------- alternative unique
def test_maut_utilites_p82():
    np.testing.assert_allclose(utilites([5, 5], [0, 0], [10, 10], [1, -1]), [0.5, 0.5])
    np.testing.assert_allclose(utilites([10, 0], [0, 0], [10, 10], [1, -1]), [1.0, 1.0])
    r = maut([8, 2], [0, 0], [10, 10], [1, -1], [0.5, 0.5], tau=0.7)
    assert r.S == pytest.approx(0.8) and r.accepte and r.classe == "A"
    assert not maut([6, 5], [0, 0], [10, 10], [1, -1], [0.5, 0.5], tau=0.7).accepte


SCORES = [0.85, 0.78, 0.71, 0.69, 0.62, 0.54]
Y = [1, 1, 1, 0, 0, 0]


@pytest.mark.parametrize("tau, tpr, fpr, d, j", [
    (0.75, 0.67, 0.00, 0.33, 0.67), (0.65, 1.00, 0.33, 0.33, 0.67), (0.60, 1.00, 0.67, 0.67, 0.33)])
def test_seuil_tableau_p88(tau, tpr, fpr, d, j):
    i = indicateurs(SCORES, Y, tau)
    assert (i.TPR, i.FPR, i.d, i.J) == pytest.approx((tpr, fpr, d, j), abs=0.005)


def test_seuil_courbe_complete_e30():
    meilleur, _ = seuil_optimal(SCORES, Y, "youden")
    assert meilleur.TPR == 1 and meilleur.FPR == 0 and meilleur.tau == pytest.approx(0.71)
    points = courbe_roc(SCORES, Y)
    assert (points[0].TPR, points[0].FPR) == (0, 0) and (points[-1].TPR, points[-1].FPR) == (1, 1)


def test_seuil_cout_asymetrique():
    i = indicateurs(SCORES, Y, 0.65, c_fp=5, c_fn=1)
    assert i.cout == 5 * i.FP + 1 * i.FN


def test_distance_ideal_p89():
    r = distance_ideal([1.0, 0.0])
    assert r.D_plus == pytest.approx(1.0) and r.D_moins == pytest.approx(1.0) and r.C == pytest.approx(0.5)
    assert distance_ideal([1, 1, 1]).C == pytest.approx(1.0)


def test_scores_maut_vectorises():
    X = np.array([[8, 2], [6, 5]])
    np.testing.assert_allclose(scores_maut(X, [0, 0], [10, 10], [1, -1], [0.5, 0.5]), [0.8, 0.55])
