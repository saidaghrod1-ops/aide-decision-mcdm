"""Nombres flous trapézoïdaux et arithmétique floue (cours p. 108–109).

N(a, b, c, d) ; si b = c, nombre flou triangulaire ; si a = b = c = d, nombre réel.
Pour A(a1, a2, a3, a4) et B(b1, b2, b3, b4) positifs :
    A + B = (a1 + b1, a2 + b2, a3 + b3, a4 + b4)
    A − B = (a1 − b4, a2 − b3, a3 − b2, a4 − b1)
    A ⊗ B = (a1 b1, a2 b2, a3 b3, a4 b4)
    λA    = (λa1, λa2, λa3, λa4)
    A ⊘ B = (a1/b4, a2/b3, a3/b2, a4/b1)
"""

from __future__ import annotations

import numpy as np

from core.commun import ErreurMethode


def verifier(A, nom: str = "A") -> np.ndarray:
    A = np.asarray(A, float)
    if A.shape != (4,):
        raise ErreurMethode(f"{nom} doit avoir quatre composantes (a, b, c, d)")
    if np.any(np.diff(A) < 0):
        raise ErreurMethode(f"{nom} : il faut a ≤ b ≤ c ≤ d")
    return A


def triangulaire_vers_trapeze(a: float, b: float, c: float) -> np.ndarray:
    return np.array([a, b, b, c], float)


def addition(A, B) -> np.ndarray:
    return verifier(A) + verifier(B, "B")


def soustraction(A, B) -> np.ndarray:
    return verifier(A) - verifier(B, "B")[::-1]


def multiplication(A, B) -> np.ndarray:
    A, B = verifier(A), verifier(B, "B")
    if np.any(A < 0) or np.any(B < 0):
        raise ErreurMethode("la multiplication du cours est définie pour des nombres flous positifs")
    return A * B


def scalaire(lam: float, A) -> np.ndarray:
    if lam < 0:
        raise ErreurMethode("λ doit être positif ou nul")
    return lam * verifier(A)


def division(A, B) -> np.ndarray:
    A, B = verifier(A), verifier(B, "B")
    if np.any(A < 0) or np.any(B <= 0):
        raise ErreurMethode("la division du cours suppose A positif et B strictement positif")
    return A / B[::-1]


def est_triangulaire(A) -> bool:
    A = verifier(A)
    return bool(np.isclose(A[1], A[2]))
