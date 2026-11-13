"""Ensembles flous hésitants (Torra, 2010 ; cours p. 125–130).

h = {γ1, …, γl} ⊂ [0, 1]
    h^α = ∪ {γ^α} ;  αh = ∪ {1 − (1 − γ)^α} ;  h1 ⊕ h2 = ∪ {γ1 + γ2 − γ1γ2} ;  h1 ⊗ h2 = ∪ {γ1γ2}
    s(h) = (1/l) Σ γ ;  σ(h) = [(1/l) Σ (γ − s(h))²]^(1/2)
    Comparaison : score, puis déviation (la plus faible déviation l'emporte).
    Distance de Hamming : (1/l) Σ |h1^σ(k) − h2^σ(k)|, l = max(l(h1), l(h2)) ; l'ensemble le plus court
    est prolongé par son minimum (pessimiste) ou son maximum (optimiste), choix non précisé par le cours (E-20).
"""

from __future__ import annotations

from functools import cmp_to_key

import numpy as np

from core.commun import ErreurMethode


def creer(valeurs) -> np.ndarray:
    h = np.asarray(sorted(float(v) for v in valeurs))
    if len(h) == 0:
        raise ErreurMethode("un élément hésitant contient au moins une valeur")
    if np.any((h < 0) | (h > 1)):
        raise ErreurMethode("les valeurs d'un élément hésitant appartiennent à [0, 1]")
    return h


def lire(texte: str) -> np.ndarray:
    """« 0.2;0.3;0.5 » → élément hésitant."""
    try:
        return creer(float(v) for v in str(texte).replace(",", ";").split(";") if v.strip())
    except ValueError as exc:
        raise ErreurMethode(f"« {texte} » : valeurs séparées par « ; » attendues") from exc


def _union(valeurs) -> np.ndarray:
    return np.unique(np.round(np.asarray(list(valeurs), float), 12))


def puissance(h, a: float) -> np.ndarray:
    if a <= 0:
        raise ErreurMethode("α doit être strictement positif")
    return _union(g ** a for g in creer(h))


def scalaire(a: float, h) -> np.ndarray:
    if a <= 0:
        raise ErreurMethode("α doit être strictement positif")
    return _union(1 - (1 - g) ** a for g in creer(h))


def somme(h1, h2) -> np.ndarray:
    return _union(g1 + g2 - g1 * g2 for g1 in creer(h1) for g2 in creer(h2))


def produit(h1, h2) -> np.ndarray:
    return _union(g1 * g2 for g1 in creer(h1) for g2 in creer(h2))


def score(h) -> float:
    return float(np.mean(creer(h)))


def deviation(h) -> float:
    h = creer(h)
    return float(np.sqrt(np.mean((h - h.mean()) ** 2)))


def comparer(h1, h2) -> int:
    """+1 si h1 > h2, −1 si h1 < h2, 0 si égaux (p. 129)."""
    s1, s2 = score(h1), score(h2)
    if abs(s1 - s2) > 1e-12:
        return 1 if s1 > s2 else -1
    d1, d2 = deviation(h1), deviation(h2)
    if abs(d1 - d2) > 1e-12:
        return 1 if d1 < d2 else -1
    return 0


def classer(elements) -> list[int]:
    return sorted(range(len(elements)), key=cmp_to_key(lambda i, j: comparer(elements[j], elements[i])))


def prolonger(h, longueur: int, extension: str = "pessimiste") -> np.ndarray:
    h = creer(h)
    if extension not in ("pessimiste", "optimiste"):
        raise ErreurMethode("extension « pessimiste » ou « optimiste »")
    ajout = h.min() if extension == "pessimiste" else h.max()
    return np.sort(np.r_[h, np.full(longueur - len(h), ajout)])


def distance_hamming(h1, h2, extension: str = "pessimiste") -> float:
    l = max(len(creer(h1)), len(creer(h2)))
    a, b = prolonger(h1, l, extension), prolonger(h2, l, extension)
    return float(np.mean(np.abs(a - b)))
