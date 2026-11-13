"""Scénario 1 : de MCDM à la classification supervisée (cours p. 91).

1) Pondération par MCDM ; 2) score S_i par une technique additive (WSM, normalisation linéaire p. 55) ;
3) étiquettes A, B, C, D à partir de S_i (seuils non précisés par le cours : quartiles par défaut, E-17) ;
4) apprentissage d'un modèle de ML (Random Forest) ; 5) prédiction de la classe d'une nouvelle alternative.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from core.commun import ErreurMethode, tableau, verifier_poids, verifier_sens

CLASSES = ("A", "B", "C", "D")


def scores_wsm_reference(X, sens, w, reference=None) -> np.ndarray:
    """Q⁽¹⁾ avec la normalisation linéaire du cours, les max/min étant pris sur `reference`
    (par défaut X lui-même) : une nouvelle alternative est ainsi notée sur la même échelle."""
    X = tableau(X)
    ref = X if reference is None else tableau(reference)
    s = verifier_sens(sens, X.shape[1])
    w = verifier_poids(w, X.shape[1])
    if np.any(X <= 0) or np.any(ref <= 0):
        raise ErreurMethode("la normalisation linéaire du cours (p. 55) suppose des valeurs strictement positives")
    R = np.where(s > 0, X / ref.max(axis=0), ref.min(axis=0) / X)
    return R @ w


def seuils_quartiles(S) -> np.ndarray:
    """Seuils décroissants (t_A, t_B, t_C) : A si S ≥ t_A, B si S ≥ t_B, C si S ≥ t_C, D sinon."""
    return np.quantile(np.asarray(S, float), [0.75, 0.5, 0.25])


def etiqueter(S, seuils) -> np.ndarray:
    S = np.asarray(S, float)
    tA, tB, tC = seuils
    if not tA >= tB >= tC:
        raise ErreurMethode("les seuils doivent être décroissants : A ≥ B ≥ C")
    return np.where(S >= tA, "A", np.where(S >= tB, "B", np.where(S >= tC, "C", "D")))


@dataclass(frozen=True, eq=False)
class ResultatClassification:
    modele: object
    exactitude: float
    confusion: np.ndarray      # lignes = vraies classes, colonnes = classes prédites (ordre CLASSES)
    n_apprentissage: int
    n_test: int


def entrainer(X, etiquettes, n_arbres: int = 200, profondeur: int | None = None,
              part_test: float = 0.25, graine: int = 42) -> ResultatClassification:
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import accuracy_score, confusion_matrix
    from sklearn.model_selection import train_test_split

    X = tableau(X)
    y = np.asarray(etiquettes)
    if len(np.unique(y)) < 2:
        raise ErreurMethode("il faut au moins deux classes différentes pour entraîner un classifieur")
    Xa, Xt, ya, yt = train_test_split(X, y, test_size=part_test, random_state=graine, stratify=y)
    modele = RandomForestClassifier(n_estimators=n_arbres, max_depth=profondeur, random_state=graine)
    modele.fit(Xa, ya)
    pred = modele.predict(Xt)
    return ResultatClassification(modele, float(accuracy_score(yt, pred)),
                                  confusion_matrix(yt, pred, labels=list(CLASSES)), len(ya), len(yt))
