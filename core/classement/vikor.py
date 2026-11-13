"""VIKOR — méthode de classement de compromis (cours p. 66–72).

    S_i = Σ_j w_j (x_j* − x_ij)/(x_j* − x_j⁻),  R_i = max_j [w_j (x_j* − x_ij)/(x_j* − x_j⁻)]
    Q_i = v (S_i − S*)/(S⁻ − S*) + (1 − v)(R_i − R*)/(R⁻ − R*),  v = 0,5 en général
    Plus S, R, Q sont petits, meilleure est l'alternative (tri croissant, E-5).
    Conditions (p. 70) : avantage acceptable Q(A'') − Q(A') ≥ 1/(m − 1) ;
    stabilité acceptable : A' est aussi la meilleure en termes de S et de R.
    Ensembles de compromis (p. 71).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from core.commun import ErreurMethode, rangs, tableau, verifier_poids, verifier_sens

_TOL = 1e-12


@dataclass(frozen=True, eq=False)
class VikorResultat:
    meilleures: np.ndarray    # x_j*
    pires: np.ndarray         # x_j⁻
    termes: np.ndarray        # w_j (x_j* − x_ij)/(x_j* − x_j⁻)
    S: np.ndarray
    R: np.ndarray
    S_etoile: float
    S_moins: float
    R_etoile: float
    R_moins: float
    v: float
    Q: np.ndarray
    rangs_S: np.ndarray
    rangs_R: np.ndarray
    rangs_Q: np.ndarray
    ordre_Q: np.ndarray       # indices triés par Q croissant
    DQ: float
    avantage: float           # Q(A'') − Q(A')
    condition_avantage: bool
    condition_stabilite: bool
    compromis: tuple[int, ...]
    explication: str


def vikor(X, sens, w, v: float = 0.5) -> VikorResultat:
    X = tableau(X)
    s = verifier_sens(sens, X.shape[1])
    w = verifier_poids(w, X.shape[1])
    m = X.shape[0]
    if m < 2:
        raise ErreurMethode("VIKOR demande au moins deux alternatives")
    if not 0.0 <= v <= 1.0:
        raise ErreurMethode("v doit appartenir à [0, 1]")
    meilleures = np.where(s > 0, X.max(axis=0), X.min(axis=0))
    pires = np.where(s > 0, X.min(axis=0), X.max(axis=0))
    ecart = meilleures - pires
    sur = np.where(ecart != 0, ecart, 1.0)
    termes = np.where(ecart != 0, w * (meilleures - X) / sur, 0.0)
    S, R = termes.sum(axis=1), termes.max(axis=1)
    S_et, S_mo, R_et, R_mo = S.min(), S.max(), R.min(), R.max()
    partie_S = (S - S_et) / (S_mo - S_et) if S_mo > S_et else np.zeros(m)
    partie_R = (R - R_et) / (R_mo - R_et) if R_mo > R_et else np.zeros(m)
    Q = v * partie_S + (1 - v) * partie_R

    ordre = np.argsort(np.round(Q, 12), kind="stable")
    a1, a2 = int(ordre[0]), int(ordre[1])
    DQ = 1.0 / (m - 1)
    avantage = float(Q[a2] - Q[a1])
    c1 = avantage >= DQ - _TOL
    c2 = bool(S[a1] <= S_et + _TOL and R[a1] <= R_et + _TOL)
    if c1 and c2:
        compromis = (a1,)
        expl = "Les deux conditions sont vérifiées : la solution de compromis est unique."
    elif not c1:
        compromis = tuple(int(k) for k in ordre if Q[k] - Q[a1] < DQ - _TOL)
        expl = ("L'avantage acceptable n'est pas vérifié : l'ensemble de compromis contient toutes les "
                "alternatives A^k telles que Q(A^k) − Q(A') < 1/(m − 1) (p. 71).")
    else:
        compromis = (a1, a2)
        expl = ("Seule la stabilité acceptable n'est pas vérifiée : les solutions de compromis sont A' et A'' "
                "(p. 71).")
    return VikorResultat(meilleures, pires, termes, S, R, float(S_et), float(S_mo), float(R_et), float(R_mo),
                         float(v), Q, rangs(S, False), rangs(R, False), rangs(Q, False), ordre, DQ, avantage,
                         bool(c1), c2, compromis, expl)
