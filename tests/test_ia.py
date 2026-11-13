"""MCDM et IA (p. 91–94) : classification, valeurs SHAP exactes, prévision ARIMA."""

import numpy as np
import pytest

from core.classement.topsis import topsis
from core.ia.classification import entrainer, etiqueter, scores_wsm_reference, seuils_quartiles
from core.ia.dynamique import matrice_previsions, prevoir_arima
from core.ia.shapley import importance_globale, mise_a_jour_poids, poids_shap, shap_exact


def test_shap_exact_modele_lineaire_analytique():
    rng = np.random.default_rng(0)
    a = np.array([0.5, -2.0, 1.0, 3.0])
    f = lambda X: X @ a + 1.0
    X, Z = rng.normal(size=(7, 4)), rng.normal(size=(20, 4))
    phi, phi0 = shap_exact(f, X, Z)
    np.testing.assert_allclose(phi, a * (X - Z.mean(axis=0)), atol=1e-12)      # φ_j = a_j (x_j − E z_j)
    np.testing.assert_allclose(phi0 + phi.sum(axis=1), f(X), atol=1e-12)       # efficacité


def test_shap_exact_interactions_efficacite():
    rng = np.random.default_rng(1)
    f = lambda X: X[:, 0] * X[:, 1] + np.maximum(X[:, 2], 0)
    X, Z = rng.normal(size=(5, 3)), rng.normal(size=(15, 3))
    phi, phi0 = shap_exact(f, X, Z)
    np.testing.assert_allclose(phi0 + phi.sum(axis=1), f(X), atol=1e-12)


def test_mise_a_jour_des_poids_p92():
    w_ahp = np.array([0.5, 0.3, 0.2])
    phi = np.array([[0.1, -0.3, 0.0], [-0.1, 0.1, 0.2]])
    w_shap = poids_shap(phi)
    np.testing.assert_allclose(importance_globale(phi), [0.1, 0.2, 0.1])
    np.testing.assert_allclose(w_shap, [0.25, 0.5, 0.25])
    np.testing.assert_allclose(mise_a_jour_poids(w_ahp, w_shap, 1.0), w_ahp)
    np.testing.assert_allclose(mise_a_jour_poids(w_ahp, w_shap, 0.0), w_shap)
    assert mise_a_jour_poids(w_ahp, w_shap, 0.4).sum() == pytest.approx(1.0)


def test_classes_quartiles_et_classifieur():
    rng = np.random.default_rng(2)
    X = rng.uniform(1, 10, size=(200, 3))
    S = scores_wsm_reference(X, [1, -1, 1], [0.5, 0.2, 0.3])
    classes = etiqueter(S, seuils_quartiles(S))
    assert set(classes) == {"A", "B", "C", "D"}
    assert abs((classes == "A").mean() - 0.25) < 0.02
    r = entrainer(X, classes, n_arbres=50)
    assert r.exactitude > 0.6 and r.confusion.sum() == r.n_test
    # Une alternative nouvelle est notée sur l'échelle du dataset de référence
    s_nouv = scores_wsm_reference(X[:1], [1, -1, 1], [0.5, 0.2, 0.3], reference=X)
    assert s_nouv[0] == pytest.approx(S[0])


def test_arima_prevision_ar1():
    rng = np.random.default_rng(3)
    phi, moyenne, n = 0.6, 8.0, 400
    y = np.empty(n); y[0] = moyenne
    for t in range(1, n):
        y[t] = moyenne + phi * (y[t - 1] - moyenne) + rng.normal(scale=0.2)
    p = prevoir_arima(y, (1, 0, 0))
    attendu = moyenne + phi * (y[-1] - moyenne)
    assert p.valeur == pytest.approx(attendu, abs=0.1)


def test_chaine_prevision_matrice_topsis_p94():
    rng = np.random.default_rng(4)
    niveaux = {("A1", "r"): 8, ("A1", "s"): 0.25, ("A2", "r"): 6, ("A2", "s"): 0.15}
    series = {k: v + rng.normal(scale=0.01 * v, size=40) for k, v in niveaux.items()}
    M = matrice_previsions(series, ["A1", "A2"], ["r", "s"], (0, 0, 0))
    np.testing.assert_allclose(M, [[8, 0.25], [6, 0.15]], rtol=0.02)
    assert topsis(M, [1, -1], [0.5, 0.5]).RC.shape == (2,)
