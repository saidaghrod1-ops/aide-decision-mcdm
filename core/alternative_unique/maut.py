"""MAUT — théorie de l'utilité multi-attributs pour une alternative isolée (cours p. 81–84).

    S(A) = Σ_j w_j u_j(x_j)
    critère bénéfice : u = (x − x_min)/(x_max − x_min) ; critère coût : u = (x − x_max)/(x_min − x_max)
    Décision : accepter si S(A) ≥ τ, rejeter sinon (τ ∈ [0, 1]).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from core.commun import ErreurMethode, verifier_poids, verifier_sens


@dataclass(frozen=True, eq=False)
class MautResultat:
    u: np.ndarray
    contributions: np.ndarray    # w_j u_j
    S: float
    tau: float
    accepte: bool
    classe: str
    hors_bornes: np.ndarray      # critères dont la valeur sort de [x_min, x_max]


def utilites(x, xmin, xmax, sens) -> np.ndarray:
    x, xmin, xmax = (np.asarray(v, float) for v in (x, xmin, xmax))
    s = verifier_sens(sens, x.shape[-1])
    if np.any(xmax <= xmin):
        raise ErreurMethode("chaque critère doit vérifier x_min < x_max")
    return np.where(s > 0, (x - xmin) / (xmax - xmin), (x - xmax) / (xmin - xmax))


def classe(S: float, seuils_classes=(0.8, 0.6)) -> str:
    """Classe A/B/C (sortie citée p. 81 ; seuils réglables, E-18)."""
    a, b = seuils_classes
    return "A" if S >= a else ("B" if S >= b else "C")


def maut(x, xmin, xmax, sens, w, tau: float = 0.7, seuils_classes=(0.8, 0.6)) -> MautResultat:
    x = np.asarray(x, float)
    w = verifier_poids(w, len(x))
    if not 0 <= tau <= 1:
        raise ErreurMethode("τ doit appartenir à [0 ; 1]")
    u = utilites(x, xmin, xmax, sens)
    contrib = w * u
    S = float(contrib.sum())
    hors = (x < np.asarray(xmin, float)) | (x > np.asarray(xmax, float))
    return MautResultat(u, contrib, S, float(tau), S >= tau, classe(S, seuils_classes), hors)


def scores_maut(X, xmin, xmax, sens, w) -> np.ndarray:
    """Scores S_i de plusieurs alternatives (historique de décisions, p. 85–86)."""
    X = np.asarray(X, float)
    w = verifier_poids(w, X.shape[1])
    return utilites(X, xmin, xmax, sens) @ w
