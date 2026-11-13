"""Choix du seuil de décision à partir de décisions passées (cours p. 85–88).

    ŷ_i = 1 si S_i ≥ τ, 0 sinon ;  TPR = TP/(TP + FN) ;  FPR = FP/(FP + TN)
    Méthode 1 : minimiser d = √(FPR² + (1 − TPR)²) ;  Méthode 2 : maximiser J = TPR − FPR
    Outils également cités (p. 85, 88) : exactitude (TP + TN)/N ; coût c_FP·FP + c_FN·FN.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from core.commun import ErreurMethode


@dataclass(frozen=True)
class Indicateurs:
    tau: float
    TP: int
    FP: int
    FN: int
    TN: int
    TPR: float
    FPR: float
    d: float
    J: float
    exactitude: float
    cout: float


def _verifier(scores, y):
    scores, y = np.asarray(scores, float), np.asarray(y, int)
    if scores.shape != y.shape or scores.ndim != 1:
        raise ErreurMethode("il faut un score et une décision par alternative")
    if not np.all(np.isin(y, (0, 1))):
        raise ErreurMethode("les décisions doivent valoir 1 (accepter) ou 0 (rejeter)")
    if y.sum() == 0 or y.sum() == len(y):
        raise ErreurMethode("il faut au moins une bonne et une mauvaise alternative pour calculer TPR et FPR")
    return scores, y


def indicateurs(scores, y, tau: float, c_fp: float = 1.0, c_fn: float = 1.0) -> Indicateurs:
    scores, y = _verifier(scores, y)
    yh = (scores >= tau - 1e-12).astype(int)
    TP = int(np.sum((yh == 1) & (y == 1))); FN = int(np.sum((yh == 0) & (y == 1)))
    FP = int(np.sum((yh == 1) & (y == 0))); TN = int(np.sum((yh == 0) & (y == 0)))
    TPR, FPR = TP / (TP + FN), FP / (FP + TN)
    return Indicateurs(float(tau), TP, FP, FN, TN, TPR, FPR, float(np.hypot(FPR, 1 - TPR)), TPR - FPR,
                       (TP + TN) / len(y), c_fp * FP + c_fn * FN)


def seuils_candidats(scores) -> np.ndarray:
    """Tous les seuils qui produisent un point différent : chaque score observé, plus un seuil au-dessus du max."""
    s = np.unique(np.asarray(scores, float))[::-1]
    return np.r_[s[0] + 1e-6, s]


def courbe_roc(scores, y, c_fp: float = 1.0, c_fn: float = 1.0) -> list[Indicateurs]:
    return [indicateurs(scores, y, t, c_fp, c_fn) for t in seuils_candidats(scores)]


CRITERES = {
    "distance": ("Distance au point idéal d (min)", lambda i: -i.d),
    "youden": ("Indice de Youden J (max)", lambda i: i.J),
    "exactitude": ("Exactitude (max)", lambda i: i.exactitude),
    "cout": ("Coût des erreurs (min)", lambda i: -i.cout),
}


def seuil_optimal(scores, y, critere: str, c_fp: float = 1.0, c_fn: float = 1.0) -> tuple[Indicateurs, list[Indicateurs]]:
    """Meilleur seuil selon le critère ; retourne aussi tous les seuils ex æquo."""
    if critere not in CRITERES:
        raise ErreurMethode(f"critère « {critere} » inconnu")
    points = courbe_roc(scores, y, c_fp, c_fn)
    valeur = CRITERES[critere][1]
    meilleure = max(valeur(p) for p in points)
    ex_aequo = [p for p in points if abs(valeur(p) - meilleure) < 1e-12]
    return ex_aequo[0], ex_aequo
