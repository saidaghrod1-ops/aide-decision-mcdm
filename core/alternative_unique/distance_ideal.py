"""Distance à l'idéal pour une alternative isolée (logique de TOPSIS, cours p. 89).

    D⁺ = √(Σ_j (x_j − x_j⁺)²),  D⁻ = √(Σ_j (x_j − x_j⁻)²),  C = D⁻ / (D⁺ + D⁻)

Calcul sur les utilités normalisées u_j ∈ [0, 1] (standardisation recommandée p. 54, E-15) :
le point idéal vaut alors A⁺ = (1, …, 1) et l'anti-idéal A⁻ = (0, …, 0).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from core.commun import ErreurMethode


@dataclass(frozen=True, eq=False)
class DistanceIdealResultat:
    u: np.ndarray
    D_plus: float
    D_moins: float
    C: float


def distance_ideal(u) -> DistanceIdealResultat:
    u = np.asarray(u, float)
    if u.ndim != 1 or len(u) == 0:
        raise ErreurMethode("il faut un vecteur d'utilités non vide")
    D_plus = float(np.sqrt(np.sum((u - 1.0) ** 2)))
    D_moins = float(np.sqrt(np.sum(u ** 2)))
    if D_plus + D_moins == 0:
        raise ErreurMethode("C n'est pas défini")
    return DistanceIdealResultat(u, D_plus, D_moins, D_moins / (D_plus + D_moins))
