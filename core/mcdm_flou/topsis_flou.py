"""TOPSIS floue de Chen (2000), cours p. 135–142.

    3) x̃_ij = (1/K)(x̃_ij1 ⊕ … ⊕ x̃_ijK) ; w̃_j = (1/K)(w̃_j1 ⊕ … ⊕ w̃_jK)
    4) critère + : r̃_ij = (a/c_j*, b/c_j*, c/c_j*), c_j* = max_i c_ij
       critère − : r̃_ij = (a_j⁻/c, a_j⁻/b, a_j⁻/a), a_j⁻ = min_i a_ij   (ordre corrigé, E-4)
    5) ṽ_ij = r̃_ij ⊗ w̃_j        6) v* = (1, 1, 1), v⁻ = (0, 0, 0)
    7) D_i* = Σ_j d(ṽ_ij, v*), D_i⁻ = Σ_j d(ṽ_ij, v⁻), d = √((1/3) Σ (écarts)²)
    8) CC_i = D_i⁻ / (D_i⁻ + D_i*)    9) tri décroissant de CC
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from core.commun import ErreurMethode, rangs, verifier_sens
from core.mcdm_flou.alternative_unique import verifier_triangulaires


@dataclass(frozen=True, eq=False)
class TopsisFlouResultat:
    X: np.ndarray        # (m × n × 3) matrice collective
    W: np.ndarray        # (n × 3) poids collectifs
    R: np.ndarray        # normalisée
    V: np.ndarray        # pondérée
    d_plus: np.ndarray   # (m × n) d(ṽ_ij, v*)
    d_moins: np.ndarray
    D_plus: np.ndarray
    D_moins: np.ndarray
    CC: np.ndarray
    rangs: np.ndarray


def agreger(evaluations) -> np.ndarray:
    """Moyenne des K décideurs (axe 0) : (1/K)(x̃1 ⊕ … ⊕ x̃K)."""
    E = verifier_triangulaires(evaluations)
    return E.mean(axis=0)


def normaliser(X, sens) -> np.ndarray:
    X = verifier_triangulaires(X)
    s = verifier_sens(sens, X.shape[1])
    R = np.empty_like(X)
    for j in range(X.shape[1]):
        col = X[:, j, :]
        if s[j] > 0:
            c_etoile = col[:, 2].max()
            if c_etoile <= 0:
                raise ErreurMethode("c_j* doit être strictement positif")
            R[:, j, :] = col / c_etoile
        else:
            a_moins = col[:, 0].min()
            if np.any(col <= 0):
                raise ErreurMethode("pour un critère coût, les nombres flous doivent être strictement positifs")
            R[:, j, :] = a_moins / col[:, ::-1]
    return R


def topsis_flou(evaluations, poids, sens) -> TopsisFlouResultat:
    """evaluations : (K × m × n × 3) ; poids : (K × n × 3) ; sens : (n,)."""
    E = verifier_triangulaires(evaluations)
    P = verifier_triangulaires(poids)
    if E.ndim != 4 or P.ndim != 3 or E.shape[0] != P.shape[0] or E.shape[2] != P.shape[1]:
        raise ErreurMethode("dimensions incohérentes entre évaluations (K × m × n × 3) et poids (K × n × 3)")
    X = agreger(E)
    W = agreger(P)
    R = normaliser(X, sens)
    V = R * W[None, :, :]
    d_plus = np.sqrt(((V - 1.0) ** 2).mean(axis=2))
    d_moins = np.sqrt((V ** 2).mean(axis=2))
    D_plus, D_moins = d_plus.sum(axis=1), d_moins.sum(axis=1)
    CC = D_moins / (D_moins + D_plus)
    return TopsisFlouResultat(X, W, R, V, d_plus, d_moins, D_plus, D_moins, CC, rangs(CC))


def depuis_tableaux(echelle_alt: dict, echelle_imp: dict, evaluations: list, poids: list):
    """Construit les tableaux numériques à partir des formats longs.

    evaluations : lignes (alternative, decideur, critere, terme) ; poids : lignes (decideur, critere, terme).
    Retourne (alternatives, decideurs, criteres, E (K × m × n × 3), P (K × n × 3)).
    """
    alternatives = list(dict.fromkeys(l[0] for l in evaluations))
    decideurs = list(dict.fromkeys(l[1] for l in evaluations))
    criteres = list(dict.fromkeys(l[2] for l in evaluations))
    E = np.full((len(decideurs), len(alternatives), len(criteres), 3), np.nan)
    for a, k, c, t in evaluations:
        if t not in echelle_alt:
            raise ErreurMethode(f"terme « {t} » absent de l'échelle d'évaluation")
        E[decideurs.index(k), alternatives.index(a), criteres.index(c)] = echelle_alt[t]
    P = np.full((len(decideurs), len(criteres), 3), np.nan)
    for k, c, t in poids:
        if k not in decideurs or c not in criteres:
            raise ErreurMethode(f"poids pour un décideur ou un critère inconnu : {k}, {c}")
        if t not in echelle_imp:
            raise ErreurMethode(f"terme « {t} » absent de l'échelle d'importance")
        P[decideurs.index(k), criteres.index(c)] = echelle_imp[t]
    if np.isnan(E).any():
        raise ErreurMethode("évaluation manquante : chaque décideur doit noter chaque alternative sur chaque critère")
    if np.isnan(P).any():
        raise ErreurMethode("poids manquant : chaque décideur doit évaluer l'importance de chaque critère")
    return alternatives, decideurs, criteres, E, P
