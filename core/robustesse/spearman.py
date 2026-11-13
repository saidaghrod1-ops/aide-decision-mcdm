"""Corrélation de Spearman entre deux classements (cours p. 77–79).

    ρ = 1 − 6 Σ d_i² / (n (n² − 1)),  d_i = rang (méthode 1) − rang (méthode 2)

La formule du cours suppose l'absence d'ex æquo ; en cas d'ex æquo, elle est appliquée
aux rangs moyens et devient approximative (E-21).
"""

from __future__ import annotations

import itertools
from collections import Counter

import numpy as np

from core.commun import ErreurMethode, a_des_exaequo

# Valeurs critiques exactes, test bilatéral à 5 % : P(ρ ≥ c) ≤ 2,5 % sous l'hypothèse
# d'indépendance (loi de permutation exacte). n = 6 → 0,886, comme dans le cours (p. 79).
# Pour n = 4, aucun ρ n'est significatif à 5 %. Vérifiées par tests/test_cours_spearman.py.
VALEURS_CRITIQUES_5PCT = {4: None, 5: 1.0, 6: 0.8857, 7: 0.7857, 8: 0.7381, 9: 0.7, 10: 0.6485}


def spearman(r1, r2) -> float:
    r1, r2 = np.asarray(r1, float), np.asarray(r2, float)
    n = len(r1)
    if n != len(r2) or n < 2:
        raise ErreurMethode("les deux classements doivent porter sur les mêmes n ≥ 2 alternatives")
    d = r1 - r2
    return float(1 - 6 * np.sum(d ** 2) / (n * (n ** 2 - 1)))


def matrice_spearman(classements: dict[str, np.ndarray]) -> tuple[list[str], np.ndarray]:
    noms = list(classements)
    k = len(noms)
    M = np.ones((k, k))
    for i in range(k):
        for j in range(i + 1, k):
            M[i, j] = M[j, i] = spearman(classements[noms[i]], classements[noms[j]])
    return noms, M


def valeur_critique(n: int) -> float | None:
    """Valeur critique (bilatérale, 5 %) pour 4 ≤ n ≤ 10 ; None si non disponible."""
    return VALEURS_CRITIQUES_5PCT.get(n)


def valeur_critique_exacte(n: int, alpha: float = 0.05) -> float | None:
    """Recalcul par énumération des n! permutations (utilisé par les tests, n ≤ 8)."""
    base = list(range(n))
    compte = Counter(sum((i - j) ** 2 for i, j in zip(base, p)) for p in itertools.permutations(base))
    total, cumul, critique = sum(compte.values()), 0, None
    for D in sorted(compte):
        cumul += compte[D]
        if cumul / total <= alpha / 2 + 1e-15:
            critique = D
        else:
            break
    return None if critique is None else 1 - 6 * critique / (n * (n * n - 1))


def contient_exaequo(*classements) -> bool:
    return any(a_des_exaequo(r) for r in classements)
