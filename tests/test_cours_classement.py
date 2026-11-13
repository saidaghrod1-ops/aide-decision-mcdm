"""Classement : WSM / WPM / WASPAS (p. 53–59), TOPSIS (p. 60–65), VIKOR (p. 66–72)."""

import numpy as np
import pytest

from core.classement.ponderation_simple import courbe_lambda, waspas
from core.classement.topsis import topsis
from core.classement.vikor import vikor
from core.commun import ErreurMethode


def test_waspas_machines_p58_reference(matrice):
    M = matrice("C4_waspas_machines")
    r = waspas(M.valeurs, M.sens, M.poids, 0.5)
    np.testing.assert_allclose(r.Q1, [0.7443, 0.8310, 0.8746, 0.9110], atol=1e-4)
    np.testing.assert_allclose(r.Q2, [0.7289, 0.8037, 0.8630, 0.9060], atol=1e-4)
    np.testing.assert_allclose(r.Q, [0.7366, 0.8174, 0.8688, 0.9085], atol=1e-4)
    assert list(r.rangs_Q) == [4, 3, 2, 1]
    # Contrôles indépendants : formules de la p. 56–57 et cas λ = 1 / λ = 0
    R = np.where(M.sens > 0, M.valeurs / M.valeurs.max(0), M.valeurs.min(0) / M.valeurs)
    np.testing.assert_allclose(r.Q1, R @ M.poids, rtol=1e-12)
    np.testing.assert_allclose(r.Q2, np.prod(R ** M.poids, axis=1), rtol=1e-12)
    np.testing.assert_allclose(waspas(M.valeurs, M.sens, M.poids, 1.0).Q, r.Q1)
    np.testing.assert_allclose(waspas(M.valeurs, M.sens, M.poids, 0.0).Q, r.Q2)
    lams, Q = courbe_lambda(M.valeurs, M.sens, M.poids, 11)
    np.testing.assert_allclose(Q[5], r.Q, rtol=1e-12)


def test_waspas_refuse_valeurs_negatives(matrice):
    M = matrice("C12_portefeuille_vukovic2020")
    with pytest.raises(ErreurMethode, match="strictement positives"):
        waspas(M.valeurs, M.sens, M.poids / M.poids.sum())


def test_topsis_voitures_p65_reference(matrice):
    M = matrice("C5_topsis_voitures")
    r = topsis(M.valeurs, M.sens, M.poids)
    np.testing.assert_allclose(r.RC, [0.4545, 0.5678, 0.3703, 0.5527], atol=1e-4)
    assert list(r.rangs) == [3, 1, 4, 2]          # M2 > M4 > M1 > M3
    np.testing.assert_allclose(r.I_plus, [0.1187, 0.0614, 0.2179, 0.1187], atol=1e-4)
    np.testing.assert_allclose(r.I_moins, [0.0791, 0.0409, 0.1937, 0.1780], atol=1e-4)
    np.testing.assert_allclose(r.R, M.valeurs / np.sqrt((M.valeurs ** 2).sum(0)), rtol=1e-12)
    assert np.all((r.RC >= 0) & (r.RC <= 1))


def test_topsis_poids_non_normalises_refuses(matrice):
    M = matrice("C5_topsis_voitures")
    with pytest.raises(ErreurMethode, match="somme des poids"):
        topsis(M.valeurs, M.sens, [0.2, 0.2, 0.4, 0.3])


def test_vikor_machines_p72_reference(matrice):
    M = matrice("C6_vikor_machines")
    r = vikor(M.valeurs, M.sens, M.poids, 0.5)
    np.testing.assert_allclose(r.S, [0.5, 0.375, 0.5])
    np.testing.assert_allclose(r.R, [0.5, 0.25, 0.5])
    np.testing.assert_allclose(r.Q, [1, 0, 1])
    assert r.DQ == pytest.approx(0.5) and r.avantage == pytest.approx(1.0)
    assert r.condition_avantage and r.condition_stabilite
    assert r.compromis == (1,)                     # A2
    assert list(r.rangs_Q) == [2, 1, 2]            # A1 et A3 ex æquo


def test_vikor_condition_avantage_non_verifiee():
    # Deux alternatives très proches en Q : ensemble de compromis élargi (p. 71)
    X = np.array([[10, 10], [9.9, 10.2], [5, 4], [4, 6]])
    r = vikor(X, [1, 1], [0.5, 0.5])
    assert not r.condition_avantage
    assert len(r.compromis) >= 2
    assert all(r.Q[k] - r.Q[r.compromis[0]] < r.DQ for k in r.compromis)


def test_vikor_condition_stabilite_non_verifiee():
    # A2 est la meilleure en Q (v = 0,2) et en R, mais pas en S : solutions A' et A'' (p. 71)
    X = np.array([[10, 1], [7, 3], [5, 10]], float)
    r = vikor(X, [1, 1], [0.5, 0.5], v=0.2)
    np.testing.assert_allclose(r.Q, [0.8, 0.2, 0.8])
    assert r.condition_avantage and not r.condition_stabilite
    assert r.compromis == (1, 0)
