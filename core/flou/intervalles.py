"""Nombres d'intervalle (cours p. 110–111).

A = [a1, a2] (a1 ≤ a2, E-27) ; centre m(A) = (a1 + a2)/2 ; largeur w(A) = (a2 − a1)/2.
"""

from __future__ import annotations

import numpy as np

from core.commun import ErreurMethode


def verifier(A, nom: str = "A") -> np.ndarray:
    A = np.asarray(A, float)
    if A.shape != (2,) or A[0] > A[1]:
        raise ErreurMethode(f"{nom} doit être un intervalle [a1, a2] avec a1 ≤ a2")
    return A


def centre(A) -> float:
    A = verifier(A)
    return float((A[0] + A[1]) / 2)


def largeur(A) -> float:
    A = verifier(A)
    return float((A[1] - A[0]) / 2)


def scalaire(lam: float, A) -> np.ndarray:
    if lam < 0:
        raise ErreurMethode("λ doit être positif ou nul")
    return lam * verifier(A)


def addition(A, B) -> np.ndarray:
    return verifier(A) + verifier(B, "B")


def soustraction(A, B) -> np.ndarray:
    A, B = verifier(A), verifier(B, "B")
    return np.array([A[0] - B[1], A[1] - B[0]])


def multiplication(A, B) -> np.ndarray:
    A, B = verifier(A), verifier(B, "B")
    p = [A[0] * B[0], A[0] * B[1], A[1] * B[0], A[1] * B[1]]
    return np.array([min(p), max(p)])


def division(A, B) -> np.ndarray:
    A, B = verifier(A), verifier(B, "B")
    if B[0] <= 0 <= B[1]:
        raise ErreurMethode("la division exige 0 ∉ B")
    return np.array([A[0] / B[1], A[1] / B[0]])


def inferieur_ou_egal(A, B) -> bool:
    """A ≤ B ⇔ m(A) ≤ m(B)."""
    return centre(A) <= centre(B)
