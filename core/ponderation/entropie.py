"""Méthode de l'entropie (cours p. 44–45).

    p_ij = d_ij / Σ_i d_ij,   E_j = −(1/ln m) Σ_i p_ij ln p_ij,   w_j = (1 − E_j) / (n − Σ_k E_k)

Le cours part d'une « matrice normalisée D » sans préciser la normalisation (E-10) :
par défaut, normalisation linéaire du cours (p. 55) ; convention 0 · ln 0 = 0.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from core.commun import ErreurMethode, normaliser, tableau


@dataclass(frozen=True, eq=False)
class EntropieResultat:
    normalisation: str
    D: np.ndarray
    P: np.ndarray
    PlnP: np.ndarray
    k: float
    E: np.ndarray
    diversification: np.ndarray   # 1 − E_j
    w: np.ndarray


def poids_entropie(X, sens, normalisation: str = "lineaire") -> EntropieResultat:
    X = tableau(X)
    m, n = X.shape
    if m < 2:
        raise ErreurMethode("l'entropie demande au moins deux alternatives (k = 1/ln m)")
    D = normaliser(X, sens, normalisation)
    if np.any(D < 0):
        raise ErreurMethode("l'entropie demande des valeurs normalisées positives ou nulles")
    sommes = D.sum(axis=0)
    if np.any(sommes == 0):
        raise ErreurMethode("une colonne normalisée est nulle : p_ij n'est pas défini")
    P = D / sommes
    with np.errstate(divide="ignore", invalid="ignore"):
        PlnP = np.where(P > 0, P * np.log(P), 0.0)
    k = 1.0 / np.log(m)
    E = -k * PlnP.sum(axis=0)
    d = 1.0 - E
    if np.isclose(d.sum(), 0.0):
        raise ErreurMethode("toutes les entropies valent 1 : aucun critère ne discrimine les alternatives")
    w = d / (n - E.sum())
    return EntropieResultat(normalisation, D, P, PlnP, k, E, d, w)
