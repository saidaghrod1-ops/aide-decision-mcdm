"""Ensembles flous picturaux (cours p. 122–124), notation du cours :
μ = appartenance positive, ν = appartenance négative, π = appartenance neutre ;
μ + ν + π ≤ 1 ; degré de refus = 1 − (μ + ν + π).

Opérations de la p. 124 (exposant n remplacé par λ, E-3) :
    A ⊕ B = (μA + μB − μAμB, νAνB, πAπB)
    λA    = (1 − (1 − μA)^λ, νA^λ, πA^λ)
    A^λ   = (μA^λ, 1 − (1 − νA)^λ, 1 − (1 − πA)^λ)
Ces deux dernières formules peuvent produire un triplet qui viole μ + ν + π ≤ 1 (E-32) :
le résultat est alors retourné tel quel et signalé par `valide = False`.
"""

from __future__ import annotations

from dataclasses import dataclass

from core.commun import ErreurMethode

_TOL = 1e-9


@dataclass(frozen=True)
class NFP:
    mu: float
    nu: float
    pi: float

    @property
    def refus(self) -> float:
        return 1.0 - (self.mu + self.nu + self.pi)

    @property
    def valide(self) -> bool:
        return all(-_TOL <= v <= 1 + _TOL for v in (self.mu, self.nu, self.pi)) and self.refus >= -_TOL


def creer(mu: float, nu: float, pi: float) -> NFP:
    a = NFP(mu, nu, pi)
    if not a.valide:
        raise ErreurMethode("il faut μ, ν, π ∈ [0, 1] et μ + ν + π ≤ 1")
    return a


def addition(a: NFP, b: NFP) -> NFP:
    return NFP(a.mu + b.mu - a.mu * b.mu, a.nu * b.nu, a.pi * b.pi)


def scalaire(lam: float, a: NFP) -> NFP:
    if lam <= 0:
        raise ErreurMethode("λ doit être strictement positif")
    return NFP(1 - (1 - a.mu) ** lam, a.nu ** lam, a.pi ** lam)


def puissance(a: NFP, lam: float) -> NFP:
    if lam <= 0:
        raise ErreurMethode("λ doit être strictement positif")
    return NFP(a.mu ** lam, 1 - (1 - a.nu) ** lam, 1 - (1 - a.pi) ** lam)
