"""CRITIC — CRiteria Importance Through Intercriteria Correlation (cours p. 46–51).

    r_ij min-max selon le sens ; σ_j (dénominateur m − 1) ; ρ_jk de Pearson ;
    C_j = σ_j Σ_k (1 − |ρ_jk|) ;  w_j = C_j / Σ C_j
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from core.commun import ErreurMethode, normalisation_min_max, tableau


@dataclass(frozen=True, eq=False)
class CriticResultat:
    R: np.ndarray
    moyennes: np.ndarray
    sigma: np.ndarray
    rho: np.ndarray
    un_moins_abs_rho: np.ndarray
    conflit: np.ndarray       # Σ_k (1 − |ρ_jk|)
    C: np.ndarray
    w: np.ndarray


def poids_critic(X, sens) -> CriticResultat:
    X = tableau(X)
    m, n = X.shape
    if m < 3:
        raise ErreurMethode("CRITIC demande au moins trois alternatives (corrélations)")
    R = normalisation_min_max(X, sens)
    moyennes = R.mean(axis=0)
    sigma = R.std(axis=0, ddof=1)
    centre = R - moyennes
    produit = centre.T @ centre
    normes = np.sqrt(np.diag(produit))
    with np.errstate(divide="ignore", invalid="ignore"):
        rho = produit / np.outer(normes, normes)
    # Critère constant : σ_j = 0, corrélation indéfinie, et C_j = 0 de toute façon.
    rho = np.where(np.isfinite(rho), rho, 0.0)
    np.fill_diagonal(rho, 1.0)
    un_moins = 1.0 - np.abs(rho)
    conflit = un_moins.sum(axis=1)
    C = sigma * conflit
    if C.sum() <= 0:
        raise ErreurMethode("tous les indices C_j sont nuls : les critères ne discriminent pas les alternatives")
    return CriticResultat(R, moyennes, sigma, rho, un_moins, conflit, C, C / C.sum())
