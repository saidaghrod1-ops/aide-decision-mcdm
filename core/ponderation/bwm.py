"""BWM — Best-Worst Method, modèle linéaire du cours (p. 35–43).

    Min ξ  s.c.  |w_B − a_Bj w_j| ≤ ξ,  |w_j − a_jW w_W| ≤ ξ,  Σ w_j = 1,  w_j ≥ 0,  ξ ≥ 0

Variables x = (w_1, …, w_n, ξ), objectif f(x) = ξ, contraintes sous la forme A x ≤ b (p. 39–43).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import linprog

from core.commun import ErreurMethode


@dataclass(frozen=True, eq=False)
class BwmResultat:
    best: int
    worst: int
    aB: np.ndarray
    aW: np.ndarray
    c: np.ndarray
    A_ub: np.ndarray
    b_ub: np.ndarray
    libelles: tuple[str, ...]     # une description par ligne de A_ub
    A_eq: np.ndarray
    b_eq: np.ndarray
    w: np.ndarray
    xi: float
    ecarts_best: np.ndarray       # |w_B − a_Bj w_j|
    ecarts_worst: np.ndarray      # |w_j − a_jW w_W|

    @property
    def n(self) -> int:
        return len(self.w)


def nb_comparaisons(n: int) -> tuple[int, int]:
    """(BWM : 2n − 3, AHP : n(n − 1)/2), p. 35."""
    return 2 * n - 3, n * (n - 1) // 2


def controles(aB, aW, best: int, worst: int) -> list[str]:
    """Contrôles de bon sens du cours (p. 36) ; retourne la liste des anomalies."""
    aB, aW = np.asarray(aB, float), np.asarray(aW, float)
    pb = []
    if aB[best] != 1:
        pb.append("a_BB doit valoir 1 (le meilleur critère comparé à lui-même).")
    if aW[worst] != 1:
        pb.append("a_WW doit valoir 1 (le pire critère comparé à lui-même).")
    if aB[worst] != aB.max():
        pb.append("La note la plus élevée du vecteur BO doit porter sur le critère W.")
    if np.any((aB < 1) | (aB > 9)) or np.any((aW < 1) | (aW > 9)):
        pb.append("Les notes doivent être comprises entre 1 et 9.")
    if best == worst:
        pb.append("Le meilleur et le pire critère doivent être différents.")
    return pb


def construire_programme(aB, aW, best: int, worst: int):
    """Construit c, A_ub, b_ub, A_eq, b_eq dans l'ordre de la p. 40 (lignes triviales omises)."""
    aB, aW = np.asarray(aB, float), np.asarray(aW, float)
    n = len(aB)
    if len(aW) != n or n < 2:
        raise ErreurMethode("les vecteurs BO et OW doivent avoir la même longueur (au moins 2)")
    lignes, libelles = [], []
    for j in range(n):
        if j != best:   # écart au meilleur critère B
            r = np.zeros(n + 1); r[best] += 1; r[j] -= aB[j]; r[n] = -1
            lignes.append(r); libelles.append(f"w_B − a_B{j + 1} w_{j + 1} − ξ ≤ 0")
            lignes.append(-r + np.r_[np.zeros(n), -2.0]); libelles.append(f"−ξ − w_B + a_B{j + 1} w_{j + 1} ≤ 0")
        if j != worst:  # écart au pire critère W
            r = np.zeros(n + 1); r[j] += 1; r[worst] -= aW[j]; r[n] = -1
            lignes.append(r); libelles.append(f"w_{j + 1} − a_{j + 1}W w_W − ξ ≤ 0")
            lignes.append(-r + np.r_[np.zeros(n), -2.0]); libelles.append(f"−ξ − w_{j + 1} + a_{j + 1}W w_W ≤ 0")
    A_ub = np.array(lignes)
    b_ub = np.zeros(len(lignes))
    A_eq = np.r_[np.ones(n), 0.0].reshape(1, -1)
    b_eq = np.array([1.0])
    c = np.r_[np.zeros(n), 1.0]
    return c, A_ub, b_ub, tuple(libelles), A_eq, b_eq


def bwm(aB, aW, best: int, worst: int) -> BwmResultat:
    aB, aW = np.asarray(aB, float), np.asarray(aW, float)
    if np.any(aB <= 0) or np.any(aW <= 0):
        raise ErreurMethode("les notes BO et OW doivent être strictement positives")
    c, A_ub, b_ub, libelles, A_eq, b_eq = construire_programme(aB, aW, best, worst)
    n = len(aB)
    res = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, bounds=[(0, None)] * (n + 1), method="highs")
    if not res.success:
        raise ErreurMethode(f"le programme linéaire n'a pas de solution : {res.message}")
    w, xi = res.x[:n], float(res.x[n])
    ecarts_best = np.abs(w[best] - aB * w)
    ecarts_worst = np.abs(w - aW * w[worst])
    return BwmResultat(best, worst, aB, aW, c, A_ub, b_ub, libelles, A_eq, b_eq, w, xi, ecarts_best, ecarts_worst)


def violation_max(aB, aW, best: int, worst: int, w) -> float:
    """Plus grand écart d'un vecteur de poids quelconque aux égalités idéales (utilisé pour E-1)."""
    aB, aW, w = np.asarray(aB, float), np.asarray(aW, float), np.asarray(w, float)
    return float(max(np.max(np.abs(w[best] - aB * w)), np.max(np.abs(w - aW * w[worst]))))
