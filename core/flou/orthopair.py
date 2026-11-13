"""Ensembles flous pythagoriciens, q-Rung Orthopair et q-ROF à valeur d'intervalle (cours p. 118–121).

    Intuitionniste : μ + ν ≤ 1          (q = 1)
    Pythagoricien  : μ² + ν² ≤ 1        (q = 2), π = √(1 − μ² − ν²)
    q-ROF          : μ^q + ν^q ≤ 1,     π = (1 − μ^q − ν^q)^(1/q)   (signe corrigé, E-2)
    IVq-ROF        : μ = [μL, μU], ν = [νL, νU], μU^q + νU^q ≤ 1,
                     πL = (1 − μU^q − νU^q)^(1/q), πU = (1 − μL^q − νL^q)^(1/q)
"""

from __future__ import annotations

import math

from core.commun import ErreurMethode

_TOL = 1e-9


def _verifier_degres(*valeurs: float) -> None:
    if any(v < -_TOL or v > 1 + _TOL for v in valeurs):
        raise ErreurMethode("les degrés doivent appartenir à [0, 1]")


def est_valide(mu: float, nu: float, q: float) -> bool:
    _verifier_degres(mu, nu)
    if q < 1:
        raise ErreurMethode("q doit être supérieur ou égal à 1")
    return mu ** q + nu ** q <= 1 + _TOL


def hesitation(mu: float, nu: float, q: float) -> float:
    if not est_valide(mu, nu, q):
        raise ErreurMethode(f"μ^q + ν^q > 1 pour q = {q:g} : le couple n'appartient pas à cet ensemble")
    return max(0.0, 1 - mu ** q - nu ** q) ** (1 / q)


def plus_petit_q(mu: float, nu: float, q_max: int = 50) -> int | None:
    """Plus petit entier q ≥ 1 tel que μ^q + ν^q ≤ 1 (None si μ = ν = 1)."""
    _verifier_degres(mu, nu)
    for q in range(1, q_max + 1):
        if mu ** q + nu ** q <= 1 + _TOL:
            return q
    return None


def frontiere(q: float, nb: int = 201) -> tuple[list[float], list[float]]:
    """Courbe μ^q + ν^q = 1 (figure p. 120)."""
    mus = [i / (nb - 1) for i in range(nb)]
    return mus, [max(0.0, 1 - m ** q) ** (1 / q) for m in mus]


def hesitation_intervalle(mu_l: float, mu_u: float, nu_l: float, nu_u: float, q: float) -> tuple[float, float]:
    _verifier_degres(mu_l, mu_u, nu_l, nu_u)
    if mu_l > mu_u or nu_l > nu_u:
        raise ErreurMethode("il faut μL ≤ μU et νL ≤ νU")
    if q < 1:
        raise ErreurMethode("q doit être supérieur ou égal à 1")
    if mu_u ** q + nu_u ** q > 1 + _TOL:
        raise ErreurMethode("il faut μU^q + νU^q ≤ 1")
    return (max(0.0, 1 - mu_u ** q - nu_u ** q) ** (1 / q), max(0.0, 1 - mu_l ** q - nu_l ** q) ** (1 / q))


def ecart_pythagoricien(mu: float, nu: float) -> float:
    return math.sqrt(max(0.0, 1 - mu ** 2 - nu ** 2))
