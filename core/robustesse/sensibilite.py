"""Analyse de sensibilité aux poids (cours p. 12 et 32).

Le cours recommande de « faire varier les poids et vérifier la stabilité du classement ».
Mise en œuvre : on fait varier un poids w_k de 0 à 1 ; les autres poids sont réajustés
proportionnellement pour que Σ w_j = 1 reste vrai.
"""

from __future__ import annotations

from typing import Callable

import numpy as np

from core.commun import ErreurMethode, rangs


def poids_modifies(w, k: int, t: float) -> np.ndarray:
    w = np.asarray(w, float)
    n = len(w)
    nouveau = np.empty(n)
    autres = np.arange(n) != k
    reste = 1.0 - w[k]
    if reste > 1e-12:
        nouveau[autres] = w[autres] * (1.0 - t) / reste
    else:
        nouveau[autres] = (1.0 - t) / (n - 1)
    nouveau[k] = t
    return nouveau


def balayage_poids(w, k: int, scores: Callable[[np.ndarray], np.ndarray],
                   plus_grand_meilleur: bool = True, nb: int = 101):
    """Retourne (t, scores (nb × m), rangs (nb × m)) quand w_k parcourt [0, 1]."""
    if len(w) < 2:
        raise ErreurMethode("il faut au moins deux critères pour une analyse de sensibilité")
    ts = np.linspace(0.0, 1.0, nb)
    S = np.array([scores(poids_modifies(w, k, t)) for t in ts])
    Rg = np.array([rangs(s, plus_grand_meilleur) for s in S])
    return ts, S, Rg


def changements_de_premier(ts, Rg) -> list[tuple[float, int, int]]:
    """Valeurs de t où la meilleure alternative change : (t, ancienne, nouvelle)."""
    premiers = [int(np.argmin(r)) for r in Rg]
    return [(float(ts[i]), premiers[i - 1], premiers[i]) for i in range(1, len(ts)) if premiers[i] != premiers[i - 1]]
