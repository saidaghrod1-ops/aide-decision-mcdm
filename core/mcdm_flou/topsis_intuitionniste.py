"""TOPSIS floue intuitionniste en groupe (cours p. 143–150).

Notation de la section : n alternatives A_i, m critères C_j, l décideurs D_k de poids σ_k (E-22).
    3) a_ij = IFWA_σ(α_ij¹, …, α_ijˡ)            4) w_j = IFWA_σ(w_j¹, …, w_jˡ)  (indices corrigés, E-23)
    5) a'_ij = a_ij ⊗ w_j = (μ_a μ_w, ν_a + ν_w − ν_a ν_w)   (produit de la p. 115, E-24)
    6) I⁺ : μ max / ν min pour J1 (critères +), μ min / ν max pour J2 (critères −) ; I⁻ à l'inverse
    7) S_i± = (1/2m) Σ_j √((Δμ)² + (Δν)² + (Δπ)²)   (formule du cours, E-8)
    8) C_i = S_i⁻ / (S_i⁻ + S_i⁺), tri décroissant
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from core.commun import ErreurMethode, rangs, verifier_sens

_TOL = 1e-9


@dataclass(frozen=True, eq=False)
class IfTopsisResultat:
    mu: np.ndarray        # (n × m) matrice agrégée
    nu: np.ndarray
    w_mu: np.ndarray      # (m,) poids agrégés
    w_nu: np.ndarray
    mu_p: np.ndarray      # (n × m) matrice pondérée a'_ij
    nu_p: np.ndarray
    pi_p: np.ndarray
    I_plus: np.ndarray    # (m × 3) : μ, ν, π
    I_moins: np.ndarray
    S_plus: np.ndarray
    S_moins: np.ndarray
    C: np.ndarray
    rangs: np.ndarray


def _verifier_nfi(mu, nu):
    mu, nu = np.asarray(mu, float), np.asarray(nu, float)
    if np.any(mu < -_TOL) or np.any(nu < -_TOL) or np.any(mu + nu > 1 + _TOL):
        raise ErreurMethode("chaque nombre intuitionniste doit vérifier μ, ν ≥ 0 et μ + ν ≤ 1")
    return mu, nu


def ifwa(mu, nu, sigma, axe: int = 0):
    """IFWA le long d'un axe (les décideurs)."""
    mu, nu = _verifier_nfi(mu, nu)
    sigma = np.asarray(sigma, float)
    if np.any(sigma < 0) or abs(sigma.sum() - 1) > 1e-6:
        raise ErreurMethode("les poids des décideurs σ doivent être positifs et de somme 1")
    forme = [1] * mu.ndim
    forme[axe] = -1
    s = sigma.reshape(forme)
    return 1 - np.prod((1 - mu) ** s, axis=axe), np.prod(nu ** s, axis=axe)


def topsis_intuitionniste(mu_eval, nu_eval, mu_poids, nu_poids, sigma, sens) -> IfTopsisResultat:
    """mu_eval, nu_eval : (l × n × m) ; mu_poids, nu_poids : (l × m) ; sigma : (l,) ; sens : (m,)."""
    mu_eval, nu_eval = _verifier_nfi(mu_eval, nu_eval)
    mu_poids, nu_poids = _verifier_nfi(mu_poids, nu_poids)
    if mu_eval.ndim != 3 or mu_poids.ndim != 2 or mu_eval.shape[0] != mu_poids.shape[0] \
            or mu_eval.shape[2] != mu_poids.shape[1] or len(sigma) != mu_eval.shape[0]:
        raise ErreurMethode("dimensions incohérentes entre évaluations, poids des critères et poids des décideurs")
    s = verifier_sens(sens, mu_eval.shape[2])
    m = mu_eval.shape[2]
    mu, nu = ifwa(mu_eval, nu_eval, sigma)
    w_mu, w_nu = ifwa(mu_poids, nu_poids, sigma)
    mu_p = mu * w_mu
    nu_p = nu + w_nu - nu * w_nu
    pi_p = 1 - mu_p - nu_p
    plus = s > 0
    mu_plus = np.where(plus, mu_p.max(axis=0), mu_p.min(axis=0))
    nu_plus = np.where(plus, nu_p.min(axis=0), nu_p.max(axis=0))
    mu_moins = np.where(plus, mu_p.min(axis=0), mu_p.max(axis=0))
    nu_moins = np.where(plus, nu_p.max(axis=0), nu_p.min(axis=0))
    I_plus = np.column_stack([mu_plus, nu_plus, 1 - mu_plus - nu_plus])
    I_moins = np.column_stack([mu_moins, nu_moins, 1 - mu_moins - nu_moins])
    trip = np.stack([mu_p, nu_p, pi_p], axis=2)              # (n × m × 3)
    S_plus = np.sqrt(((trip - I_plus[None]) ** 2).sum(axis=2)).sum(axis=1) / (2 * m)
    S_moins = np.sqrt(((trip - I_moins[None]) ** 2).sum(axis=2)).sum(axis=1) / (2 * m)
    total = S_plus + S_moins
    if np.any(total == 0):
        raise ErreurMethode("toutes les alternatives sont identiques : C n'est pas défini")
    C = S_moins / total
    return IfTopsisResultat(mu, nu, w_mu, w_nu, mu_p, nu_p, pi_p, I_plus, I_moins, S_plus, S_moins, C, rangs(C))


def depuis_tableaux(echelle_eval: dict, echelle_poids: dict, evaluations: list, poids: list, sigma_par_decideur: dict):
    """Formats longs → tableaux. echelles : terme → (μ, ν)."""
    alternatives = list(dict.fromkeys(l[0] for l in evaluations))
    decideurs = list(dict.fromkeys(l[1] for l in evaluations))
    criteres = list(dict.fromkeys(l[2] for l in evaluations))
    l, n, m = len(decideurs), len(alternatives), len(criteres)
    mu_e, nu_e = np.full((l, n, m), np.nan), np.full((l, n, m), np.nan)
    for a, k, c, t in evaluations:
        if t not in echelle_eval:
            raise ErreurMethode(f"terme « {t} » absent de l'échelle d'évaluation")
        mu_e[decideurs.index(k), alternatives.index(a), criteres.index(c)], \
            nu_e[decideurs.index(k), alternatives.index(a), criteres.index(c)] = echelle_eval[t]
    mu_w, nu_w = np.full((l, m), np.nan), np.full((l, m), np.nan)
    for k, c, t in poids:
        if k not in decideurs or c not in criteres:
            raise ErreurMethode(f"poids pour un décideur ou un critère inconnu : {k}, {c}")
        if t not in echelle_poids:
            raise ErreurMethode(f"terme « {t} » absent de l'échelle d'importance")
        mu_w[decideurs.index(k), criteres.index(c)], nu_w[decideurs.index(k), criteres.index(c)] = echelle_poids[t]
    if np.isnan(mu_e).any():
        raise ErreurMethode("évaluation manquante : chaque décideur doit évaluer chaque alternative sur chaque critère")
    if np.isnan(mu_w).any():
        raise ErreurMethode("poids manquant : chaque décideur doit évaluer l'importance de chaque critère")
    manquants = [k for k in decideurs if k not in sigma_par_decideur]
    if manquants:
        raise ErreurMethode(f"poids σ manquant pour : {', '.join(manquants)}")
    sigma = np.array([sigma_par_decideur[k] for k in decideurs], float)
    return alternatives, decideurs, criteres, mu_e, nu_e, mu_w, nu_w, sigma
