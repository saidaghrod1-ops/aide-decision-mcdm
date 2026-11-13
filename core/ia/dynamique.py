"""MCDM dynamique (cours p. 93–94) : prévoir chaque critère à t + 1, puis classer.

    x̂_ij(t + 1) = Modèle(x_ij(t))  — modèle ARIMA, premier modèle cité par le cours.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np

from core.commun import ErreurMethode


@dataclass(frozen=True, eq=False)
class Prevision:
    valeur: float
    ecart_type_residus: float
    ajustes: np.ndarray


def prevoir_arima(serie, ordre=(1, 0, 0)) -> Prevision:
    from statsmodels.tsa.arima.model import ARIMA

    y = np.asarray(serie, float)
    if y.ndim != 1 or len(y) < 10:
        raise ErreurMethode("il faut une série d'au moins 10 observations")
    if not np.all(np.isfinite(y)):
        raise ErreurMethode("la série contient des valeurs manquantes")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ajuste = ARIMA(y, order=tuple(int(o) for o in ordre)).fit()
        prevision = float(ajuste.forecast(1)[0])
    residus = np.asarray(ajuste.resid)[max(ordre[1], 1):]
    return Prevision(prevision, float(np.std(residus, ddof=1)), np.asarray(ajuste.fittedvalues))


def matrice_previsions(series: dict, alternatives: list, criteres: list, ordre=(1, 0, 0)) -> np.ndarray:
    """series[(alternative, critère)] = tableau des observations → matrice (m × n) des prévisions."""
    M = np.empty((len(alternatives), len(criteres)))
    for i, a in enumerate(alternatives):
        for j, c in enumerate(criteres):
            if (a, c) not in series:
                raise ErreurMethode(f"série manquante : {a} / {c}")
            M[i, j] = prevoir_arima(series[(a, c)], ordre).valeur
    return M
