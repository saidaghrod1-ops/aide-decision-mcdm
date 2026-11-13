"""Logique floue (p. 96–130) et MCDM flou (p. 131–150)."""

import json

import numpy as np
import pytest

from core.commun import ErreurMethode
from core.donnees import chemin, dataset, lire_tableau_csv
from core.flou import hesitant as hf
from core.flou import intervalles as itv
from core.flou import intuitionniste as ifs
from core.flou import nombres_flous as nf
from core.flou import orthopair as qr
from core.flou import pictural as pf
from core.flou.appartenance import (alpha_coupe, alpha_coupe_trapeze, gaussienne, proprietes,
                                    trapezoidale, triangulaire)
from core.mcdm_flou.alternative_unique import evaluation_floue
from core.mcdm_flou.topsis_flou import depuis_tableaux, normaliser, topsis_flou
from core.mcdm_flou.topsis_intuitionniste import ifwa, topsis_intuitionniste


# ---------------------------------------------------------------- ensembles flous
def test_age_moyen_p104():
    assert trapezoidale(np.array([45.0]), 25, 35, 55, 65)[0] == 1.0
    assert trapezoidale(np.array([30.0]), 25, 35, 55, 65)[0] == pytest.approx(0.5)
    assert trapezoidale(np.array([30.0]), 0, 0, 25, 35)[0] == pytest.approx(0.5)       # « autant que pour jeune »
    assert trapezoidale(np.array([0.0]), 0, 0, 25, 35)[0] == 1.0


def test_average_p106():
    x = np.linspace(0, 30, 3001)
    mu = triangulaire(x, 10, 15, 20)
    p = proprietes(x, mu)
    assert p.hauteur == 1 and p.normalise and p.convexe
    assert p.noyau == pytest.approx((15, 15))
    assert p.support[0] > 10 and p.support[1] < 20            # support ouvert ]10, 20[
    assert alpha_coupe(x, mu, 0.5) == pytest.approx((12.5, 17.5), abs=1e-9)
    assert alpha_coupe_trapeze(10, 15, 15, 20, 0.5) == (12.5, 17.5)


def test_gaussienne_et_convexite():
    x = np.linspace(-5, 5, 201)
    assert gaussienne(np.array([0.0]), 0, 1)[0] == 1
    assert proprietes(x, gaussienne(x, 0, 1)).convexe
    deux_bosses = np.maximum(triangulaire(x, -4, -3, -2), triangulaire(x, 2, 3, 4))
    assert not proprietes(x, deux_bosses).convexe


def test_nombres_flous_p109():
    A, B = np.array([4, 5, 6, 7.0]), np.array([1, 2, 3, 4.0])
    np.testing.assert_array_equal(nf.addition(A, B), [5, 7, 9, 11])
    np.testing.assert_array_equal(nf.soustraction(A, B), [0, 2, 4, 6])
    np.testing.assert_array_equal(nf.multiplication(A, B), [4, 10, 18, 28])
    np.testing.assert_array_equal(nf.scalaire(2, A), [8, 10, 12, 14])
    np.testing.assert_allclose(nf.division(A, B), [1, 5 / 3, 3, 7])
    with pytest.raises(ErreurMethode):
        nf.verifier([3, 2, 1, 0])


def test_intervalles_p110_111():
    A, B = [1, 3], [2, 5]
    assert itv.centre(A) == 2 and itv.largeur(A) == 1
    np.testing.assert_array_equal(itv.addition(A, B), [3, 8])
    np.testing.assert_array_equal(itv.soustraction(A, B), [-4, 1])
    np.testing.assert_array_equal(itv.multiplication([-1, 2], [3, 4]), [-4, 8])
    np.testing.assert_allclose(itv.division(A, B), [0.2, 1.5])
    assert itv.inferieur_ou_egal(A, B)
    with pytest.raises(ErreurMethode):
        itv.division(A, [-1, 1])


# ---------------------------------------------------------------- intuitionniste
def test_vote_et_hesitation_p112_114():
    with chemin(dataset("C14_ensembles_flous_exemples")["fichiers"]["json"]).open(encoding="utf-8") as f:
        v = json.load(f)["vote_intuitionniste"]
    a = ifs.depuis_vote(v["pour"], v["contre"], v["absents"])
    assert (a.mu, a.nu, a.pi) == pytest.approx((0.5, 0.3, 0.2))
    assert ifs.NFI(0.6, 0.3).pi == pytest.approx(0.1)
    with pytest.raises(ErreurMethode):
        ifs.NFI(0.7, 0.5)


def test_operations_intuitionnistes_p115_116():
    a, b = ifs.NFI(0.6, 0.3), ifs.NFI(0.5, 0.4)
    s = ifs.addition(a, b)
    assert (s.mu, s.nu) == pytest.approx((0.8, 0.12))
    p = ifs.multiplication(a, b)
    assert (p.mu, p.nu) == pytest.approx((0.3, 0.58))
    l = ifs.scalaire(2, a)
    assert (l.mu, l.nu) == pytest.approx((0.84, 0.09))
    w = ifs.ifwa([a, b], [0.5, 0.5])
    assert w.mu == pytest.approx(1 - np.sqrt(0.4 * 0.5)) and w.nu == pytest.approx(np.sqrt(0.3 * 0.4))
    m = ifs.ifwa([a, a], [0.3, 0.7])
    assert (m.mu, m.nu) == pytest.approx((a.mu, a.nu))     # idempotence


def test_score_et_precision_p117():
    assert ifs.comparer(ifs.NFI(0.6, 0.3), ifs.NFI(0.5, 0.3)) == 1
    assert ifs.comparer(ifs.NFI(0.4, 0.2), ifs.NFI(0.3, 0.1)) == 1    # même score, précision plus forte
    assert ifs.comparer(ifs.NFI(0.4, 0.2), ifs.NFI(0.4, 0.2)) == 0


# ---------------------------------------------------------------- orthopaires et picturaux
def test_q_rof_p118_121():
    assert qr.hesitation(0.6, 0.3, 1) == pytest.approx(0.1)
    assert qr.hesitation(0.6, 0.3, 2) == pytest.approx(np.sqrt(1 - 0.36 - 0.09))
    assert qr.hesitation(0.8, 0.7, 3) == pytest.approx((1 - 0.512 - 0.343) ** (1 / 3))   # formule corrigée E-2
    assert not qr.est_valide(0.8, 0.7, 2) and qr.est_valide(0.8, 0.7, 3)
    assert qr.plus_petit_q(0.8, 0.7) == 3
    pl, pu = qr.hesitation_intervalle(0.3, 0.5, 0.2, 0.4, 2)
    assert pl == pytest.approx(np.sqrt(1 - 0.25 - 0.16)) and pu == pytest.approx(np.sqrt(1 - 0.09 - 0.04))


def test_pictural_p122_124_et_e32():
    a = pf.creer(0.3, 0.3, 0.3)
    assert a.refus == pytest.approx(0.1)
    s = pf.addition(a, pf.creer(0.5, 0.2, 0.1))
    assert (s.mu, s.nu, s.pi) == pytest.approx((0.65, 0.06, 0.03))
    assert not pf.puissance(a, 2).valide           # E-32 : A^λ peut violer μ + ν + π ≤ 1
    assert not pf.scalaire(0.5, pf.creer(0, 0.5, 0.5)).valide


# ---------------------------------------------------------------- hésitants
def test_hesitants_p129():
    _, lignes = lire_tableau_csv(chemin(dataset("C11_hesitants_p129")["fichiers"]["tableau"]))
    h = [hf.lire(l[1]) for l in lignes]
    assert [hf.score(x) for x in h] == pytest.approx([0.45, 0.60, 0.45])
    assert [hf.deviation(x) for x in h] == pytest.approx([0.2291, 0.1633, 0.1225], abs=1e-4)
    assert hf.classer(h) == [1, 2, 0]              # h2 > h3 > h1


def test_hesitants_operations_et_distance_p127_130():
    np.testing.assert_allclose(hf.somme([0.2], [0.5]), [0.6])
    np.testing.assert_allclose(hf.produit([0.2, 0.4], [0.5]), [0.1, 0.2])
    np.testing.assert_allclose(hf.puissance([0.5], 2), [0.25])
    np.testing.assert_allclose(hf.scalaire(2, [0.5]), [0.75])
    assert hf.distance_hamming([0.2, 0.4], [0.2, 0.4]) == 0
    assert hf.distance_hamming([0.2, 0.4, 0.6], [0.3], "pessimiste") == pytest.approx((0.1 + 0.1 + 0.3) / 3)
    assert hf.distance_hamming([0.2, 0.4, 0.6], [0.3], "optimiste") == pytest.approx((0.1 + 0.1 + 0.3) / 3)


# ---------------------------------------------------------------- MCDM flou
def test_alternative_unique_floue_p134():
    _, lignes = lire_tableau_csv(chemin(dataset("C9_flou_alternative_unique")["fichiers"]["tableau"]))
    D = np.array([[float(x) for x in l[3:6]] for l in lignes])
    w = np.array([float(l[2]) for l in lignes])
    r = evaluation_floue(D, w)
    np.testing.assert_allclose(r.R, [[0.67, 0.80, 1.00], [0.85, 0.92, 1.00]], atol=0.005)
    np.testing.assert_allclose(r.S_flou, [0.74, 0.85, 1.00], atol=0.005)
    assert r.S == pytest.approx(0.86, abs=0.005) and r.largeur == pytest.approx(0.26, abs=0.005)


def _chen():
    f = dataset("C10_chen2000_recrutement")["fichiers"]
    echelle = lambda cle: {l[1]: tuple(float(x) for x in l[2:5]) for l in lire_tableau_csv(chemin(f[cle]))[1]}
    return depuis_tableaux(echelle("echelle_alternatives"), echelle("echelle_importance"),
                           lire_tableau_csv(chemin(f["evaluations"]))[1], lire_tableau_csv(chemin(f["poids"]))[1])


def test_topsis_flou_chen2000_p135_142_reference():
    alternatives, decideurs, criteres, E, P = _chen()
    assert (len(decideurs), len(alternatives), len(criteres)) == (3, 3, 5)
    r = topsis_flou(E, P, np.ones(5))
    np.testing.assert_allclose(r.X[0, 0], [17 / 3, 23 / 3, 28 / 3], atol=1e-9)      # (MG + G + MG)/3
    np.testing.assert_allclose(r.W[0], [0.7, 2.6 / 3, 2.9 / 3], atol=1e-9)           # (H + VH + MH)/3
    np.testing.assert_allclose(r.D_plus, [1.9456, 1.2589, 1.6020], atol=1e-4)
    np.testing.assert_allclose(r.D_moins, [3.4295, 4.1057, 3.7706], atol=1e-4)
    np.testing.assert_allclose(r.CC, [0.6380, 0.7653, 0.7018], atol=1e-4)
    assert list(r.rangs) == [3, 1, 2]                # A2 > A3 > A1


def test_topsis_flou_critere_negatif_triplet_valide_e4():
    X = np.array([[[2, 3, 4]], [[4, 5, 8.0]]])
    R = normaliser(X, [-1])
    assert np.all(np.diff(R, axis=2) >= 0)
    np.testing.assert_allclose(R[0, 0], [0.5, 2 / 3, 1.0])


def test_if_topsis_petit_exemple_calcule_a_la_main():
    # 1 décideur, 2 alternatives, 2 critères (+, −), poids (1, 0) pour isoler les calculs
    mu_e = np.array([[[0.6, 0.2], [0.3, 0.5]]]); nu_e = np.array([[[0.3, 0.7], [0.6, 0.4]]])
    mu_w = np.array([[1.0, 0.0]]); nu_w = np.array([[0.0, 1.0]])
    r = topsis_intuitionniste(mu_e, nu_e, mu_w, nu_w, [1.0], [1, -1])
    # Critère 2 de poids (0, 1) : a' = (0, 1) pour tous → aucune contribution à la séparation.
    d = np.sqrt(0.3 ** 2 + 0.3 ** 2 + 0.0 ** 2)
    assert r.S_plus == pytest.approx([0, d / 4]) and r.S_moins == pytest.approx([d / 4, 0])
    assert list(r.C) == pytest.approx([1.0, 0.0])
    mu, nu = ifwa(np.array([0.6, 0.6]), np.array([0.3, 0.3]), np.array([0.4, 0.6]))
    assert (mu, nu) == pytest.approx((0.6, 0.3))
