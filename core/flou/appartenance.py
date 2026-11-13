"""Ensembles flous et fonctions d'appartenance (cours p. 97–107)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from core.commun import ErreurMethode


def caracteristique(x, seuil: float) -> np.ndarray:
    """Ensemble classique (p. 98) : 1_A(x) = 1 si x ≥ seuil, 0 sinon (ex. adulte : 18 ans)."""
    return (np.asarray(x, float) >= seuil).astype(float)


def trapezoidale(x, a: float, b: float, c: float, d: float) -> np.ndarray:
    """p. 102 : 0 si x ≤ a ; (x − a)/(b − a) si a ≤ x ≤ b ; 1 si b ≤ x ≤ c ; (d − x)/(d − c) si c ≤ x ≤ d ; 0 si x ≥ d."""
    if not a <= b <= c <= d:
        raise ErreurMethode("il faut a ≤ b ≤ c ≤ d")
    x = np.asarray(x, float)
    mu = np.zeros_like(x)
    if b > a:
        montee = (x > a) & (x < b)
        mu[montee] = (x[montee] - a) / (b - a)
    mu[(x >= b) & (x <= c)] = 1.0
    if d > c:
        descente = (x > c) & (x < d)
        mu[descente] = (d - x[descente]) / (d - c)
    return mu


def triangulaire(x, a: float, b: float, c: float) -> np.ndarray:
    """p. 103 : cas b = c du trapèze."""
    return trapezoidale(x, a, b, b, c)


def gaussienne(x, m: float, sigma: float) -> np.ndarray:
    """p. 103 : μ(x) = exp(−(x − m)² / (2σ²))."""
    if sigma <= 0:
        raise ErreurMethode("σ doit être strictement positif")
    x = np.asarray(x, float)
    return np.exp(-((x - m) ** 2) / (2 * sigma ** 2))


def appartenance(x, type_: str, parametres) -> np.ndarray:
    if type_ == "triangulaire":
        return triangulaire(x, *parametres)
    if type_ == "trapezoidale":
        return trapezoidale(x, *parametres)
    if type_ == "gaussienne":
        return gaussienne(x, *parametres)
    raise ErreurMethode(f"type de fonction d'appartenance « {type_} » inconnu")


def par_points(x, points_x, points_mu) -> np.ndarray:
    """Fonction d'appartenance linéaire par morceaux définie par des points (x, μ)."""
    px, pm = np.asarray(points_x, float), np.asarray(points_mu, float)
    if len(px) < 2 or np.any(np.diff(px) <= 0):
        raise ErreurMethode("les abscisses doivent être strictement croissantes (au moins deux points)")
    if np.any((pm < 0) | (pm > 1)):
        raise ErreurMethode("les degrés d'appartenance doivent appartenir à [0, 1]")
    return np.interp(np.asarray(x, float), px, pm, left=pm[0], right=pm[-1])


# --------------------------------------------------------------------------
# Propriétés (p. 105–107), calculées sur une grille de l'univers X
# --------------------------------------------------------------------------
@dataclass(frozen=True, eq=False)
class Proprietes:
    hauteur: float
    normalise: bool
    support: tuple[float, float] | None     # enveloppe [min, max] des x où μ > 0
    noyau: tuple[float, float] | None       # enveloppe des x où μ = 1
    convexe: bool


def _enveloppe(x, masque):
    if not masque.any():
        return None
    return float(x[masque].min()), float(x[masque].max())


def est_convexe(mu, tol: float = 1e-9) -> bool:
    """μ(λx₁ + (1 − λ)x₂) ≥ min(μ(x₁), μ(x₂)) sur une grille ⇔ μ croît puis décroît."""
    mu = np.asarray(mu, float)
    for i in range(1, len(mu) - 1):
        if mu[i] < min(mu[:i].max(), mu[i + 1:].max()) - tol:
            return False
    return True


def proprietes(x, mu) -> Proprietes:
    x, mu = np.asarray(x, float), np.asarray(mu, float)
    h = float(mu.max())
    return Proprietes(h, bool(np.isclose(h, 1.0)), _enveloppe(x, mu > 1e-12),
                      _enveloppe(x, mu >= 1 - 1e-12), est_convexe(mu))


def alpha_coupe(x, mu, alpha: float) -> tuple[float, float] | None:
    """α-cut(A) = {x ∈ X | μ_A(x) ≥ α} (enveloppe sur la grille)."""
    x, mu = np.asarray(x, float), np.asarray(mu, float)
    return _enveloppe(x, mu >= alpha - 1e-12)


def alpha_coupe_trapeze(a: float, b: float, c: float, d: float, alpha: float) -> tuple[float, float]:
    """α-coupe exacte d'un trapèze : [a + α(b − a), d − α(d − c)]."""
    if not 0 < alpha <= 1:
        raise ErreurMethode("α doit appartenir à ]0, 1]")
    return a + alpha * (b - a), d - alpha * (d - c)
