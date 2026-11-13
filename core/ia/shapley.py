"""Scénario 2 : XAI (SHAP) comme approche de pondération (cours p. 92).

    Ŝ = f(x) appris par un modèle de ML ; Ŝ = φ0 + Σ_j φ_j (SHAP) ;
    w_j^new = α w_j^AHP + (1 − α) |φ_j| / Σ |φ_j|

Valeurs SHAP calculées exactement : énumération des 2^n coalitions, valeur d'une coalition S
= moyenne de f sur un échantillon de fond où les critères hors de S sont remplacés (espérance
interventionnelle). Importance globale |φ_j| = moyenne des |φ_ij| sur les alternatives (E-16).
"""

from __future__ import annotations

import itertools
from math import factorial
from typing import Callable

import numpy as np

from core.commun import ErreurMethode, verifier_poids


def shap_exact(f: Callable[[np.ndarray], np.ndarray], X_expliquer, X_fond) -> tuple[np.ndarray, float]:
    """Retourne (φ : k × n, φ0 = E[f(fond)])."""
    X = np.asarray(X_expliquer, float)
    Z = np.asarray(X_fond, float)
    k, n = X.shape
    if n > 12:
        raise ErreurMethode("calcul exact limité à 12 critères (2^n coalitions)")
    masques = np.array(list(itertools.product([0, 1], repeat=n)), dtype=bool)   # (2^n × n)
    b = len(Z)
    # Pour chaque alternative et chaque coalition : b lignes hybrides.
    hybrides = np.where(masques[None, :, None, :], X[:, None, None, :], Z[None, None, :, :])  # k × 2^n × b × n
    valeurs = f(hybrides.reshape(-1, n)).reshape(k, len(masques), b).mean(axis=2)          # k × 2^n
    index = {tuple(m): i for i, m in enumerate(masques)}
    phi = np.zeros((k, n))
    for j in range(n):
        for i, m in enumerate(masques):
            if m[j]:
                continue
            taille = int(m.sum())
            poids = factorial(taille) * factorial(n - taille - 1) / factorial(n)
            avec = m.copy(); avec[j] = True
            phi[:, j] += poids * (valeurs[:, index[tuple(avec)]] - valeurs[:, i])
    phi0 = float(f(Z).mean())
    return phi, phi0


def importance_globale(phi) -> np.ndarray:
    return np.abs(np.asarray(phi, float)).mean(axis=0)


def poids_shap(phi) -> np.ndarray:
    imp = importance_globale(phi)
    if imp.sum() == 0:
        raise ErreurMethode("toutes les contributions SHAP sont nulles")
    return imp / imp.sum()


def mise_a_jour_poids(w_ahp, w_shap, alpha: float) -> np.ndarray:
    if not 0 <= alpha <= 1:
        raise ErreurMethode("α doit appartenir à [0, 1]")
    w_ahp = verifier_poids(w_ahp, len(w_ahp))
    w_shap = verifier_poids(w_shap, len(w_ahp))
    return alpha * w_ahp + (1 - alpha) * w_shap


def entrainer_regresseur(X, S, n_arbres: int = 200, graine: int = 42):
    """Ŝ = f(x) par Random Forest ; retourne (modèle, R² hors échantillon)."""
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.model_selection import train_test_split

    Xa, Xt, Sa, St = train_test_split(np.asarray(X, float), np.asarray(S, float), test_size=0.25, random_state=graine)
    modele = RandomForestRegressor(n_estimators=n_arbres, random_state=graine, n_jobs=1)
    modele.fit(Xa, Sa)
    return modele, float(modele.score(Xt, St))
