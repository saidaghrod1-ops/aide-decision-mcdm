"""Évaluer une seule alternative avec des nombres flous triangulaires (cours p. 134).

    1. Normalisation : (a, b, c) ÷ c_max (critère bénéfice)
    2. Agrégation : S̃ = Σ w_i · x̃_i
    3. Défuzzification (centre de gravité) : S = (a + b + c)/3 ; largeur c − a = incertitude
Le cours ne traite que des critères bénéfice ici ; pour un critère coût, la normalisation
de la TOPSIS floue (p. 136, ordre corrigé E-4) est appliquée : (a⁻/c, a⁻/b, a⁻/a).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from core.commun import ErreurMethode, verifier_poids, verifier_sens


@dataclass(frozen=True, eq=False)
class EvaluationFloue:
    D: np.ndarray        # (n × 3)
    R: np.ndarray        # normalisée
    termes: np.ndarray   # w_i x̃_i
    S_flou: np.ndarray   # (3,)
    S: float
    largeur: float


def verifier_triangulaires(T) -> np.ndarray:
    T = np.asarray(T, float)
    if T.ndim < 2 or T.shape[-1] != 3:
        raise ErreurMethode("chaque évaluation est un triplet (a, b, c)")
    if np.any(np.diff(T, axis=-1) < 0):
        raise ErreurMethode("chaque nombre flou triangulaire doit vérifier a ≤ b ≤ c")
    if np.any(T < 0):
        raise ErreurMethode("les nombres flous doivent être positifs")
    return T


def evaluation_floue(D, w, sens=None) -> EvaluationFloue:
    D = verifier_triangulaires(D)
    n = D.shape[0]
    w = verifier_poids(w, n)
    s = verifier_sens(np.ones(n) if sens is None else sens, n)
    R = np.empty_like(D)
    for j in range(n):
        a, b, c = D[j]
        if s[j] > 0:
            if c <= 0:
                raise ErreurMethode("c_max doit être strictement positif")
            R[j] = D[j] / c
        else:
            if a <= 0:
                raise ErreurMethode("pour un critère coût, a doit être strictement positif")
            R[j] = np.array([a / c, a / b, a / a])
    termes = R * w[:, None]
    S_flou = termes.sum(axis=0)
    return EvaluationFloue(D, R, termes, S_flou, float(S_flou.mean()), float(S_flou[2] - S_flou[0]))
