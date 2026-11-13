"""Méthodes de pondération simple : WSM, WPM, WASPAS (cours p. 53–59).

    1) A⁺, A⁻ ; 2) r_ij linéaire (p. 55) ;
    3.1) Q⁽¹⁾ = Σ w_j r_ij ; 3.2) Q⁽²⁾ = Π r_ij^{w_j} ; 3.3) Q = λ Q⁽¹⁾ + (1 − λ) Q⁽²⁾, λ ∈ [0, 1]
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from core.commun import ErreurMethode, normalisation_lineaire, rangs, tableau, verifier_poids, verifier_sens


@dataclass(frozen=True, eq=False)
class PonderationSimpleResultat:
    ideal_positif: np.ndarray   # x_j⁺ (meilleure valeur selon le sens)
    ideal_negatif: np.ndarray   # x_j⁻
    R: np.ndarray
    termes_somme: np.ndarray    # w_j r_ij
    Q1: np.ndarray              # WSM
    facteurs_produit: np.ndarray  # r_ij^{w_j}
    Q2: np.ndarray              # WPM
    lam: float
    Q: np.ndarray               # WASPAS
    rangs_Q1: np.ndarray
    rangs_Q2: np.ndarray
    rangs_Q: np.ndarray


def solutions_ideales(X, sens) -> tuple[np.ndarray, np.ndarray]:
    X = tableau(X)
    s = verifier_sens(sens, X.shape[1])
    return np.where(s > 0, X.max(axis=0), X.min(axis=0)), np.where(s > 0, X.min(axis=0), X.max(axis=0))


def waspas(X, sens, w, lam: float = 0.5) -> PonderationSimpleResultat:
    X = tableau(X)
    w = verifier_poids(w, X.shape[1])
    if not 0.0 <= lam <= 1.0:
        raise ErreurMethode("λ doit appartenir à [0, 1]")
    plus, moins = solutions_ideales(X, sens)
    R = normalisation_lineaire(X, sens)
    termes = R * w
    Q1 = termes.sum(axis=1)
    facteurs = R ** w
    Q2 = facteurs.prod(axis=1)
    Q = lam * Q1 + (1 - lam) * Q2
    return PonderationSimpleResultat(plus, moins, R, termes, Q1, facteurs, Q2, float(lam), Q,
                                     rangs(Q1), rangs(Q2), rangs(Q))


def wsm(X, sens, w) -> np.ndarray:
    return waspas(X, sens, w, 1.0).Q1


def wpm(X, sens, w) -> np.ndarray:
    return waspas(X, sens, w, 0.0).Q2


def courbe_lambda(X, sens, w, nb: int = 101) -> tuple[np.ndarray, np.ndarray]:
    """Q_i(λ) pour λ ∈ [0, 1] : matrice (nb × m)."""
    r = waspas(X, sens, w, 0.5)
    lams = np.linspace(0, 1, nb)
    return lams, np.outer(lams, r.Q1) + np.outer(1 - lams, r.Q2)
