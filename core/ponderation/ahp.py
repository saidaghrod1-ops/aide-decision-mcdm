"""AHP — Analytic Hierarchy Process (cours p. 23–34).

Méthode approchée du cours : sommes des colonnes, normalisation, moyenne des lignes,
λmax = Σ sᵢwᵢ, CI = (λmax − n)/(n − 1), CR = CI/RI, cohérence si CR < 0,1.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from core.commun import ErreurMethode, rangs

# Table des indices aléatoires RI, telle qu'imprimée p. 29 (RI(10) = 1,56, voir E-11).
RI = {3: 0.58, 4: 0.9, 5: 1.12, 6: 1.24, 7: 1.32, 8: 1.41, 9: 1.45, 10: 1.56}
SEUIL_CR = 0.1
TOLERANCE_RECIPROCITE = 0.02   # les matrices du cours arrondissent 1/7 à 0,143 (p. 33)


@dataclass(frozen=True, eq=False)
class AhpPoids:
    A: np.ndarray            # matrice de comparaison par paires
    sommes: np.ndarray       # s_j = Σ_i α_ij
    N: np.ndarray            # α_ij / s_j
    w: np.ndarray            # moyenne des lignes de N (vecteur propre approché)
    lambda_max: float        # Σ s_i w_i (p. 28)
    CI: float | None         # None si n ≤ 2 (E-12)
    RI: float | None
    CR: float | None
    coherent: bool           # CR < 0,1 ; toujours vrai pour n ≤ 2
    w_exact: np.ndarray      # vecteur propre principal exact (AW = λmax W, p. 25)
    lambda_exact: float

    @property
    def n(self) -> int:
        return len(self.w)


def poids_ahp(A) -> AhpPoids:
    A = np.asarray(A, dtype=float)
    if A.ndim != 2 or A.shape[0] != A.shape[1] or A.shape[0] < 2:
        raise ErreurMethode("la matrice de comparaison doit être carrée, d'ordre au moins 2")
    n = A.shape[0]
    if n > 10:
        raise ErreurMethode("la table RI du cours (p. 29) s'arrête à n = 10")
    if not np.all(np.isfinite(A)) or np.any(A <= 0):
        raise ErreurMethode("les jugements doivent être strictement positifs")
    if np.any(np.abs(np.diag(A) - 1) > 1e-12):
        raise ErreurMethode("la diagonale doit valoir 1")
    if np.max(np.abs(A * A.T - 1)) > TOLERANCE_RECIPROCITE:
        raise ErreurMethode("la matrice n'est pas réciproque (α_ji doit valoir 1/α_ij)")

    sommes = A.sum(axis=0)
    N = A / sommes
    w = N.mean(axis=1)
    lam = float(sommes @ w)
    if n > 2:
        CI = (lam - n) / (n - 1)
        CR = CI / RI[n]
        coherent = CR < SEUIL_CR
        ri = RI[n]
    else:
        CI = CR = ri = None
        coherent = True

    valeurs, vecteurs = np.linalg.eig(A)
    k = int(np.argmax(valeurs.real))
    w_exact = np.abs(vecteurs[:, k].real)
    w_exact = w_exact / w_exact.sum()
    return AhpPoids(A, sommes, N, w, lam, CI, ri, CR, bool(coherent), w_exact, float(valeurs.real[k]))


@dataclass(frozen=True, eq=False)
class AhpClassement:
    criteres: AhpPoids
    locales: tuple[AhpPoids, ...]   # une matrice d'alternatives par critère
    W_local: np.ndarray             # (m alternatives × n critères) : w_ji du cours
    finales: np.ndarray             # Σ_j w_j w_ji
    contributions: np.ndarray       # w_j w_ji
    rangs: np.ndarray


def classement_ahp(A_criteres, matrices_alternatives) -> AhpClassement:
    pc = poids_ahp(A_criteres)
    if len(matrices_alternatives) != pc.n:
        raise ErreurMethode("il faut une matrice de comparaison des alternatives par critère")
    locales = tuple(poids_ahp(B) for B in matrices_alternatives)
    m = locales[0].n
    if any(l.n != m for l in locales):
        raise ErreurMethode("toutes les matrices d'alternatives doivent avoir la même taille")
    W_local = np.column_stack([l.w for l in locales])
    contributions = W_local * pc.w
    finales = contributions.sum(axis=1)
    return AhpClassement(pc, locales, W_local, finales, contributions, rangs(finales))


def priorites_finales(W_local: np.ndarray, w: np.ndarray) -> np.ndarray:
    """Utilisé par l'analyse de sensibilité (étape 5, p. 32)."""
    return np.asarray(W_local) @ np.asarray(w)
