"""Outils communs : validation, normalisations du cours, rangs."""

from __future__ import annotations

import numpy as np


class ErreurMethode(ValueError):
    """Une méthode ne s'applique pas aux données fournies (message en français)."""


def tableau(X, nom: str = "la matrice") -> np.ndarray:
    X = np.asarray(X, dtype=float)
    if X.ndim != 2 or X.shape[0] < 1 or X.shape[1] < 1:
        raise ErreurMethode(f"{nom} doit être un tableau à deux dimensions non vide")
    if not np.all(np.isfinite(X)):
        raise ErreurMethode(f"{nom} contient des valeurs manquantes")
    return X


def verifier_sens(sens, n: int) -> np.ndarray:
    s = np.asarray(sens, dtype=int)
    if s.shape != (n,) or not np.all(np.isin(s, (-1, 1))):
        raise ErreurMethode("le sens de chaque critère doit valoir +1 (bénéfice) ou −1 (coût)")
    return s


def verifier_poids(w, n: int, tol: float = 1e-3) -> np.ndarray:
    """Le cours impose Σ wⱼ = 1 (p. 53) : aucune renormalisation silencieuse."""
    w = np.asarray(w, dtype=float)
    if w.shape != (n,):
        raise ErreurMethode(f"il faut {n} poids (un par critère)")
    if not np.all(np.isfinite(w)) or np.any(w < 0):
        raise ErreurMethode("les poids doivent être positifs ou nuls")
    if abs(w.sum() - 1.0) > tol:
        raise ErreurMethode(f"la somme des poids vaut {w.sum():.4f} au lieu de 1")
    return w


# --------------------------------------------------------------------------
# Normalisations (une par famille de méthodes, comme dans le cours)
# --------------------------------------------------------------------------
def normalisation_lineaire(X, sens) -> np.ndarray:
    """p. 55 : critère + : xᵢⱼ / max xᵢⱼ ; critère − : min xᵢⱼ / xᵢⱼ."""
    X = tableau(X)
    s = verifier_sens(sens, X.shape[1])
    if np.any(X <= 0):
        raise ErreurMethode("la normalisation linéaire du cours (p. 55) suppose des valeurs strictement positives")
    return np.where(s > 0, X / X.max(axis=0), X.min(axis=0) / X)


def normalisation_min_max(X, sens) -> np.ndarray:
    """p. 47 : critère + : (x − min)/(max − min) ; critère − : (max − x)/(max − min).

    Une colonne constante (max = min) ne discrimine pas les alternatives : elle est mise à 0.
    """
    X = tableau(X)
    s = verifier_sens(sens, X.shape[1])
    mn, mx = X.min(axis=0), X.max(axis=0)
    etendue = mx - mn
    sure = np.where(etendue > 0, etendue, 1.0)
    R = np.where(s > 0, (X - mn) / sure, (mx - X) / sure)
    return np.where(etendue > 0, R, 0.0)


def normalisation_vectorielle(X) -> np.ndarray:
    """p. 62 : rᵢⱼ = xᵢⱼ / √(Σᵢ xᵢⱼ²) (le sens est traité plus tard, via I⁺ et I⁻)."""
    X = tableau(X)
    normes = np.sqrt((X ** 2).sum(axis=0))
    if np.any(normes == 0):
        raise ErreurMethode("une colonne ne contient que des zéros : normalisation vectorielle impossible")
    return X / normes


NORMALISATIONS = {
    "lineaire": "linéaire (p. 55)",
    "min-max": "min-max (p. 47)",
    "vectorielle": "vectorielle (p. 62)",
}


def normaliser(X, sens, methode: str) -> np.ndarray:
    if methode == "lineaire":
        return normalisation_lineaire(X, sens)
    if methode == "min-max":
        return normalisation_min_max(X, sens)
    if methode == "vectorielle":
        return normalisation_vectorielle(X)
    raise ErreurMethode(f"normalisation « {methode} » inconnue")


# --------------------------------------------------------------------------
# Rangs
# --------------------------------------------------------------------------
def _arrondi(x) -> np.ndarray:
    return np.round(np.asarray(x, dtype=float), 10)


def rangs(scores, plus_grand_meilleur: bool = True) -> np.ndarray:
    """Rang 1 = meilleure alternative ; ex æquo au rang minimal (1, 2, 2, 4)."""
    v = _arrondi(scores)
    v = -v if plus_grand_meilleur else v
    return np.array([1 + int(np.sum(v < x)) for x in v])


def rangs_moyens(scores, plus_grand_meilleur: bool = True) -> np.ndarray:
    """Rangs moyens en cas d'ex æquo (1, 2.5, 2.5, 4), utilisés pour Spearman."""
    v = _arrondi(scores)
    v = -v if plus_grand_meilleur else v
    return np.array([np.sum(v < x) + (np.sum(v == x) + 1) / 2 for x in v], dtype=float)


def a_des_exaequo(valeurs) -> bool:
    v = _arrondi(valeurs)
    return len(np.unique(v)) < len(v)
