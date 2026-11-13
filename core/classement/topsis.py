"""TOPSIS (Hwang et Yoon, 1981 ; cours p. 60–65).

    Étape 1 : r_ij = x_ij / √(Σ_i x_ij²)        Étape 2 : v_ij = w_j × r_ij
    Étape 3 : I⁺ et I⁻ selon J (critères +) et J' (critères −)
    Étape 4 : S_i⁺ = √(Σ_j (v_j* − v_ij)²), S_i⁻ = √(Σ_j (v_j' − v_ij)²)
    Étape 5 : RC_i* = S_i⁻ / (S_i⁻ + S_i⁺) ; la meilleure alternative a le RC le plus élevé.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from core.commun import ErreurMethode, normalisation_vectorielle, rangs, tableau, verifier_poids, verifier_sens


@dataclass(frozen=True, eq=False)
class TopsisResultat:
    R: np.ndarray
    V: np.ndarray
    I_plus: np.ndarray
    I_moins: np.ndarray
    S_plus: np.ndarray
    S_moins: np.ndarray
    RC: np.ndarray
    rangs: np.ndarray


def topsis(X, sens, w) -> TopsisResultat:
    X = tableau(X)
    s = verifier_sens(sens, X.shape[1])
    w = verifier_poids(w, X.shape[1])
    R = normalisation_vectorielle(X)
    V = R * w
    I_plus = np.where(s > 0, V.max(axis=0), V.min(axis=0))
    I_moins = np.where(s > 0, V.min(axis=0), V.max(axis=0))
    S_plus = np.sqrt(((I_plus - V) ** 2).sum(axis=1))
    S_moins = np.sqrt(((I_moins - V) ** 2).sum(axis=1))
    total = S_plus + S_moins
    if np.any(total == 0):
        raise ErreurMethode("toutes les alternatives sont identiques : RC n'est pas défini")
    RC = S_moins / total
    return TopsisResultat(R, V, I_plus, I_moins, S_plus, S_moins, RC, rangs(RC))
