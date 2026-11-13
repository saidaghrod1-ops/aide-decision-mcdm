"""Pondération : AHP (p. 23–34), BWM (p. 35–43), entropie (p. 44–45), CRITIC (p. 46–51)."""

import numpy as np
import pytest

from core.commun import ErreurMethode
from core.donnees import chemin, dataset, lire_matrice_comparaison_csv
from core.ponderation.ahp import RI, classement_ahp, poids_ahp
from core.ponderation.bwm import bwm, construire_programme, controles, nb_comparaisons, violation_max
from core.ponderation.critic import poids_critic
from core.ponderation.entropie import poids_entropie

# ---------------------------------------------------------------- AHP
A_VOITURE = [[1, 7, 3], [1 / 7, 1, 1 / 3], [1 / 3, 3, 1]]


def _matrices_voiture():
    d = dataset("C1_ahp_voiture")
    _, A = lire_matrice_comparaison_csv(chemin(d["fichiers"]["criteres"]))
    locales = [lire_matrice_comparaison_csv(chemin(d["fichiers"]["alternatives"][c]))[1]
               for c in ("Coût", "Confort", "Sécurité")]
    return A, locales


def test_ahp_table_ri_p29():
    assert RI == {3: 0.58, 4: 0.9, 5: 1.12, 6: 1.24, 7: 1.32, 8: 1.41, 9: 1.45, 10: 1.56}


def test_ahp_voiture_poids_et_coherence_reference():
    r = poids_ahp(A_VOITURE)
    np.testing.assert_allclose(r.w, [0.6687, 0.0882, 0.2431], atol=1e-4)
    assert r.lambda_max == pytest.approx(3.0108, abs=1e-4)
    assert r.CI == pytest.approx(0.0054, abs=1e-4)
    assert r.CR == pytest.approx(0.0093, abs=1e-4)
    assert r.coherent
    # Contrôle indépendant : λmax = Σ sᵢ wᵢ (p. 28) et proximité du vecteur propre exact
    np.testing.assert_allclose(r.lambda_max, np.sum(np.sum(A_VOITURE, axis=0) * r.w), rtol=1e-12)
    np.testing.assert_allclose(r.w, r.w_exact, atol=2e-3)


def test_ahp_voiture_priorites_finales_reference():
    A, locales = _matrices_voiture()
    r = classement_ahp(A, locales)
    np.testing.assert_allclose(r.W_local[0], [0.875, 1 / 6, 0.1], atol=1e-4)    # Voiture 1
    np.testing.assert_allclose(r.finales, [0.6241, 0.3759], atol=1e-4)
    assert list(r.rangs) == [1, 2]
    np.testing.assert_allclose(r.finales, r.W_local @ r.criteres.w, rtol=1e-12)


def test_ahp_matrice_2x2_cr_sans_objet():
    r = poids_ahp([[1, 7], [1 / 7, 1]])
    assert r.CR is None and r.coherent and r.lambda_max == pytest.approx(2.0)


def test_ahp_refuse_n_superieur_a_10_et_non_reciproque():
    with pytest.raises(ErreurMethode):
        poids_ahp(np.ones((11, 11)))
    with pytest.raises(ErreurMethode, match="réciproque"):
        poids_ahp([[1, 3], [3, 1]])


def test_ahp_matrice_coherente_cr_nul():
    w = np.array([0.5, 0.3, 0.2])
    r = poids_ahp(np.outer(w, 1 / w))
    np.testing.assert_allclose(r.w, w, rtol=1e-12)
    assert r.CR == pytest.approx(0.0, abs=1e-12)


# ---------------------------------------------------------------- BWM
AB, AW = [1, 3, 4, 8], [8, 5, 3, 1]


def test_bwm_lignes_de_contraintes_p43():
    _, A_ub, b_ub, _, A_eq, b_eq = construire_programme(AB, AW, 0, 3)
    lignes = {tuple(l) for l in A_ub}
    for attendue in ([-1, 3, 0, 0, -1], [1, -3, 0, 0, -1], [0, -1, 0, 5, -1], [0, 1, 0, -5, -1]):
        assert tuple(float(x) for x in attendue) in lignes
    assert np.all(b_ub == 0)
    np.testing.assert_array_equal(A_eq, [[1, 1, 1, 1, 0]])


def test_bwm_xi_du_cours_p42():
    assert bwm(AB, AW, 0, 3).xi == pytest.approx(0.08, abs=0.005)


def test_bwm_poids_reference():
    r = bwm(AB, AW, 0, 3)
    np.testing.assert_allclose(r.w, [0.5629, 0.2156, 0.1617, 0.0599], atol=1e-4)
    assert r.xi == pytest.approx(0.0838, abs=1e-4)
    assert r.w.sum() == pytest.approx(1.0)
    assert max(r.ecarts_best.max(), r.ecarts_worst.max()) <= r.xi + 1e-9


@pytest.mark.xfail(strict=True, reason="E-1 : les poids imprimés p. 42 ne sont pas la solution du modèle du cours")
def test_bwm_poids_imprimes_p42():
    np.testing.assert_allclose(bwm(AB, AW, 0, 3).w, [0.552, 0.207, 0.155, 0.086], atol=5e-4)


def test_bwm_ecart_documente_poids_du_cours_violent_les_contraintes():
    assert violation_max(AB, AW, 0, 3, [0.552, 0.207, 0.155, 0.086]) == pytest.approx(0.223, abs=1e-3)


def test_bwm_nombre_de_comparaisons_et_controles_p35_36():
    assert nb_comparaisons(4) == (5, 6)
    assert controles(AB, AW, 0, 3) == []
    assert controles([1, 3, 9, 8], AW, 0, 3)          # la note max de BO ne porte pas sur W


# ---------------------------------------------------------------- CRITIC
def test_critic_machines_p51_reference(matrice):
    M = matrice("C3_critic_machines_decoupe")
    r = poids_critic(M.valeurs, M.sens)
    np.testing.assert_allclose(r.w, [0.1436, 0.1595, 0.2639, 0.4331], atol=1e-4)
    np.testing.assert_allclose(r.sigma, M_sigma := r.R.std(axis=0, ddof=1), rtol=1e-12)
    np.testing.assert_allclose(r.sigma, [0.4576, 0.4178, 0.4787, 0.4590], atol=1e-4)
    np.testing.assert_allclose(r.rho, np.corrcoef(r.R, rowvar=False), atol=1e-12)


def test_critic_invariant_au_sens_e25(matrice):
    M = matrice("C3_critic_machines_decoupe")
    autre = M.sens.copy(); autre[0] = 1
    np.testing.assert_allclose(poids_critic(M.valeurs, M.sens).w, poids_critic(M.valeurs, autre).w, atol=1e-12)


# ---------------------------------------------------------------- Entropie
def test_entropie_formules_p44_45(matrice):
    M = matrice("C3_critic_machines_decoupe")
    r = poids_entropie(M.valeurs, M.sens, "lineaire")
    assert r.w.sum() == pytest.approx(1.0)
    np.testing.assert_allclose(r.w, (1 - r.E) / (len(r.E) - r.E.sum()), rtol=1e-12)
    np.testing.assert_allclose(r.P.sum(axis=0), 1.0)
    assert np.all((r.E >= 0) & (r.E <= 1 + 1e-12))


def test_entropie_colonne_uniforme_poids_nul():
    X = np.array([[1, 5.0], [1, 9.0], [1, 2.0]])
    r = poids_entropie(X, [1, 1], "lineaire")
    assert r.E[0] == pytest.approx(1.0) and r.w[0] == pytest.approx(0.0)


def test_entropie_refuse_valeurs_negatives(matrice):
    M = matrice("C12_portefeuille_vukovic2020")
    with pytest.raises(ErreurMethode):
        poids_entropie(M.valeurs, M.sens, "lineaire")
    poids_entropie(M.valeurs, M.sens, "min-max")   # applicable après min-max (zéros : 0·ln 0 = 0)
