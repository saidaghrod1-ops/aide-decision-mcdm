"""Ensembles et nombres flous intuitionnistes (Atanassov, 1986 ; cours p. 112–117).

α = (μ, ν), μ + ν ≤ 1, hésitation π = 1 − μ − ν.
    α1 + α2 = (μ1 + μ2 − μ1μ2, ν1ν2)        α1 × α2 = (μ1μ2, ν1 + ν2 − ν1ν2)
    λα = (1 − (1 − μ)^λ, ν^λ)               α^λ = (μ^λ, 1 − (1 − ν)^λ),  λ > 0
    IFWA_σ(α1, …, αk) = (1 − Π (1 − μj)^σj, Π νj^σj)
    Score S(α) = μ − ν ; précision H(α) = μ + ν.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import cmp_to_key

import numpy as np

from core.commun import ErreurMethode

_TOL = 1e-9


@dataclass(frozen=True)
class NFI:
    mu: float
    nu: float

    def __post_init__(self) -> None:
        if not (-_TOL <= self.mu <= 1 + _TOL and -_TOL <= self.nu <= 1 + _TOL):
            raise ErreurMethode("μ et ν doivent appartenir à [0, 1]")
        if self.mu + self.nu > 1 + _TOL:
            raise ErreurMethode(f"μ + ν = {self.mu + self.nu:.4f} > 1 : ce n'est pas un nombre intuitionniste")

    @property
    def pi(self) -> float:
        return 1.0 - self.mu - self.nu


def depuis_vote(pour: float, contre: float, absents: float) -> NFI:
    """Exemple p. 112 : μ = part des votes pour, ν = part des votes contre, π = part des absents."""
    total = pour + contre + absents
    if total <= 0 or min(pour, contre, absents) < 0:
        raise ErreurMethode("les effectifs doivent être positifs")
    return NFI(pour / total, contre / total)


def addition(a: NFI, b: NFI) -> NFI:
    return NFI(a.mu + b.mu - a.mu * b.mu, a.nu * b.nu)


def multiplication(a: NFI, b: NFI) -> NFI:
    return NFI(a.mu * b.mu, a.nu + b.nu - a.nu * b.nu)


def scalaire(lam: float, a: NFI) -> NFI:
    if lam <= 0:
        raise ErreurMethode("λ doit être strictement positif")
    return NFI(1 - (1 - a.mu) ** lam, a.nu ** lam)


def puissance(a: NFI, lam: float) -> NFI:
    if lam <= 0:
        raise ErreurMethode("λ doit être strictement positif")
    return NFI(a.mu ** lam, 1 - (1 - a.nu) ** lam)


def ifwa(alphas, sigma) -> NFI:
    sigma = np.asarray(sigma, float)
    if len(alphas) != len(sigma) or len(alphas) == 0:
        raise ErreurMethode("il faut un poids par nombre intuitionniste")
    if np.any(sigma < 0) or abs(sigma.sum() - 1) > 1e-6:
        raise ErreurMethode("les poids σ doivent être positifs et de somme 1")
    mu = np.array([a.mu for a in alphas]); nu = np.array([a.nu for a in alphas])
    return NFI(float(1 - np.prod((1 - mu) ** sigma)), float(np.prod(nu ** sigma)))


def score(a: NFI) -> float:
    return a.mu - a.nu


def precision(a: NFI) -> float:
    return a.mu + a.nu


def comparer(a: NFI, b: NFI) -> int:
    """+1 si a > b, −1 si a < b, 0 si a = b (règles p. 117)."""
    sa, sb = score(a), score(b)
    if abs(sa - sb) > 1e-12:
        return 1 if sa > sb else -1
    ha, hb = precision(a), precision(b)
    if abs(ha - hb) > 1e-12:
        return 1 if ha > hb else -1
    return 0


def classer(alphas) -> list[int]:
    """Indices du plus grand au plus petit selon score puis précision."""
    return sorted(range(len(alphas)), key=cmp_to_key(lambda i, j: comparer(alphas[j], alphas[i])))
