"""Lecture et écriture des jeux de données (CSV), catalogue des datasets.

Module pur (bibliothèque standard + numpy) : aucune dépendance à Streamlit.

Format d'une matrice de décision (CSV UTF-8, séparateur virgule, point décimal) :

    alternative,Qualité,Prix,Délai,RSE
    @sens,+,-,-,+
    @poids,0.25,0.25,0.25,0.25
    @unite,/10,DH/u,jours,/100
    F1,9.2,145,12,70

Les lignes « @ » sont facultatives : sens « + » et poids égaux par défaut.
Les nombres acceptent les fractions (« 1/7 »), utiles pour les matrices AHP.
"""

from __future__ import annotations

import csv
import io
import json
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from typing import Iterable

import numpy as np

RACINE = Path(__file__).resolve().parent.parent
DOSSIER_DONNEES = RACINE / "data"
FICHIER_CATALOGUE = DOSSIER_DONNEES / "catalogue.json"

_SENS = {
    "+": 1, "+1": 1, "1": 1, "max": 1, "benefice": 1, "bénéfice": 1,
    "-": -1, "−": -1, "-1": -1, "min": -1, "cout": -1, "coût": -1,
}
_LIGNES_META = ("@sens", "@poids", "@unite")


class ErreurDonnees(ValueError):
    """Erreur de format ou de contenu d'un jeu de données (message en français)."""


# --------------------------------------------------------------------------
# Nombres et sens
# --------------------------------------------------------------------------
def lire_nombre(texte: object) -> float:
    """Convertit une cellule en nombre : « 0.5 », « 7 », « 1/7 », « −3 »."""
    if isinstance(texte, (int, float, np.integer, np.floating)):
        return float(texte)
    s = str(texte).strip().replace("−", "-").replace(" ", "")
    if s == "":
        raise ErreurDonnees("cellule vide")
    try:
        if "/" in s:
            return float(Fraction(s))
        return float(s)
    except (ValueError, ZeroDivisionError) as exc:
        raise ErreurDonnees(f"« {texte} » n'est pas un nombre valide") from exc


def lire_sens(texte: object) -> int:
    """Convertit « + », « - », « max », « min »… en +1 (bénéfice) ou −1 (coût)."""
    cle = str(texte).strip().lower()
    if cle not in _SENS:
        raise ErreurDonnees(f"sens « {texte} » inconnu : utiliser + (bénéfice) ou - (coût)")
    return _SENS[cle]


def texte_sens(s: int) -> str:
    return "+" if s > 0 else "-"


def format_nombre(x: float, fractions: bool = False) -> str:
    """Écrit un nombre sans bruit d'arrondi ; « 1/7 » si fractions=True et x ≈ 1/k."""
    if fractions and 0 < x < 1:
        k = round(1 / x)
        if k >= 2 and abs(1 / k - x) < 1e-9:
            return f"1/{k}"
    if float(x).is_integer():
        return str(int(x))
    return repr(float(x))


# --------------------------------------------------------------------------
# Matrice de décision
# --------------------------------------------------------------------------
@dataclass(frozen=True, eq=False)
class MatriceDecision:
    """Matrice de décision du cours : m alternatives (lignes) × n critères (colonnes)."""

    alternatives: tuple[str, ...]
    criteres: tuple[str, ...]
    valeurs: np.ndarray            # (m, n)
    sens: np.ndarray               # (n,) : +1 bénéfice, −1 coût
    poids: np.ndarray | None = None  # (n,) ou None
    unites: tuple[str, ...] | None = None

    def __post_init__(self) -> None:
        X = np.asarray(self.valeurs, dtype=float)
        object.__setattr__(self, "valeurs", X)
        object.__setattr__(self, "sens", np.asarray(self.sens, dtype=int))
        m, n = len(self.alternatives), len(self.criteres)
        if m == 0 or n == 0:
            raise ErreurDonnees("la matrice doit contenir au moins une alternative et un critère")
        if X.shape != (m, n):
            raise ErreurDonnees(f"dimensions incohérentes : {X.shape} au lieu de ({m}, {n})")
        if not np.all(np.isfinite(X)):
            raise ErreurDonnees("la matrice contient des valeurs manquantes ou non numériques")
        if self.sens.shape != (n,) or not np.all(np.isin(self.sens, (-1, 1))):
            raise ErreurDonnees("le sens de chaque critère doit valoir + ou -")
        if len(set(self.alternatives)) != m:
            raise ErreurDonnees("deux alternatives portent le même nom")
        if len(set(self.criteres)) != n:
            raise ErreurDonnees("deux critères portent le même nom")
        if self.poids is not None:
            w = np.asarray(self.poids, dtype=float)
            object.__setattr__(self, "poids", w)
            if w.shape != (n,):
                raise ErreurDonnees("il faut un poids par critère")
            if not np.all(np.isfinite(w)) or np.any(w < 0):
                raise ErreurDonnees("les poids doivent être des nombres positifs ou nuls")
        if self.unites is not None and len(self.unites) != n:
            raise ErreurDonnees("il faut une unité par critère (éventuellement vide)")

    @property
    def m(self) -> int:
        return len(self.alternatives)

    @property
    def n(self) -> int:
        return len(self.criteres)

    def poids_ou_egaux(self) -> np.ndarray:
        """Poids fournis, sinon poids égaux 1/n."""
        return self.poids if self.poids is not None else np.full(self.n, 1.0 / self.n)


def _lire_texte(source: str | Path | io.IOBase | bytes) -> str:
    if isinstance(source, bytes):
        return source.decode("utf-8-sig")
    if isinstance(source, Path):
        return source.read_text(encoding="utf-8-sig")
    if hasattr(source, "read"):
        contenu = source.read()
        return contenu.decode("utf-8-sig") if isinstance(contenu, bytes) else contenu
    texte = str(source)
    if "\n" not in texte and texte.lower().endswith(".csv") and Path(texte).exists():
        return Path(texte).read_text(encoding="utf-8-sig")
    return texte


def lire_lignes_csv(source: str | Path | io.IOBase | bytes) -> list[list[str]]:
    """Lit un CSV en lignes de cellules (lignes vides ignorées, espaces retirés)."""
    texte = _lire_texte(source)
    lignes = []
    for ligne in csv.reader(io.StringIO(texte)):
        cellules = [c.strip() for c in ligne]
        if any(cellules):
            lignes.append(cellules)
    if not lignes:
        raise ErreurDonnees("le fichier CSV est vide")
    return lignes


def lire_matrice_csv(source: str | Path | io.IOBase | bytes) -> MatriceDecision:
    """Lit une matrice de décision au format décrit en tête de module."""
    lignes = lire_lignes_csv(source)
    entete = lignes[0]
    criteres = tuple(entete[1:])
    n = len(criteres)
    if n == 0:
        raise ErreurDonnees("l'en-tête doit contenir au moins un critère après la colonne « alternative »")
    meta: dict[str, list[str]] = {}
    alternatives, valeurs = [], []
    for num, ligne in enumerate(lignes[1:], start=2):
        if len(ligne) != n + 1:
            raise ErreurDonnees(f"ligne {num} : {len(ligne)} cellules au lieu de {n + 1}")
        etiquette = ligne[0]
        if etiquette.startswith("@"):
            cle = etiquette.lower()
            if cle not in _LIGNES_META:
                raise ErreurDonnees(f"ligne {num} : « {etiquette} » inconnue (attendu : @sens, @poids ou @unite)")
            meta[cle] = ligne[1:]
            continue
        try:
            valeurs.append([lire_nombre(c) for c in ligne[1:]])
        except ErreurDonnees as exc:
            raise ErreurDonnees(f"ligne {num} ({etiquette}) : {exc}") from exc
        alternatives.append(etiquette)
    if not alternatives:
        raise ErreurDonnees("aucune alternative trouvée")
    try:
        sens = [lire_sens(s) for s in meta["@sens"]] if "@sens" in meta else [1] * n
        poids = [lire_nombre(p) for p in meta["@poids"]] if "@poids" in meta else None
    except ErreurDonnees as exc:
        raise ErreurDonnees(f"ligne de métadonnées : {exc}") from exc
    unites = tuple(meta["@unite"]) if "@unite" in meta else None
    return MatriceDecision(tuple(alternatives), criteres, np.array(valeurs), np.array(sens),
                           None if poids is None else np.array(poids), unites)


def matrice_vers_csv(M: MatriceDecision, inclure_poids: bool = True) -> str:
    """Écrit une matrice de décision au format CSV de l'application."""
    sortie = io.StringIO()
    ecr = csv.writer(sortie, lineterminator="\n")
    ecr.writerow(["alternative", *M.criteres])
    ecr.writerow(["@sens", *[texte_sens(s) for s in M.sens]])
    if inclure_poids and M.poids is not None:
        ecr.writerow(["@poids", *[format_nombre(w) for w in M.poids]])
    if M.unites is not None and any(M.unites):
        ecr.writerow(["@unite", *M.unites])
    for nom, ligne in zip(M.alternatives, M.valeurs):
        ecr.writerow([nom, *[format_nombre(x) for x in ligne]])
    return sortie.getvalue()


# --------------------------------------------------------------------------
# Matrices de comparaison par paires (AHP)
# --------------------------------------------------------------------------
def completer_reciproque(A: np.ndarray) -> np.ndarray:
    """Complète une matrice de comparaison : diagonale = 1, α_ji = 1/α_ij si α_ji manque (NaN)."""
    A = np.array(A, dtype=float)
    if A.ndim != 2 or A.shape[0] != A.shape[1]:
        raise ErreurDonnees("la matrice de comparaison doit être carrée")
    n = A.shape[0]
    for i in range(n):
        if np.isnan(A[i, i]):
            A[i, i] = 1.0
        if A[i, i] != 1.0:
            raise ErreurDonnees(f"l'élément diagonal ({i + 1}, {i + 1}) doit valoir 1")
    for i in range(n):
        for j in range(n):
            if np.isnan(A[i, j]):
                if np.isnan(A[j, i]):
                    raise ErreurDonnees(f"les éléments ({i + 1}, {j + 1}) et ({j + 1}, {i + 1}) sont tous deux vides")
                A[i, j] = 1.0 / A[j, i]
    if np.any(A <= 0):
        raise ErreurDonnees("les jugements de comparaison doivent être strictement positifs")
    return A


def ecart_reciprocite(A: np.ndarray) -> float:
    """Plus grand écart |α_ij · α_ji − 1| : 0 pour une matrice parfaitement réciproque."""
    A = np.asarray(A, dtype=float)
    return float(np.max(np.abs(A * A.T - 1.0)))


def lire_matrice_comparaison_csv(source: str | Path | io.IOBase | bytes) -> tuple[tuple[str, ...], np.ndarray]:
    """Lit une matrice de comparaison carrée (cellules vides = réciproque à compléter)."""
    lignes = lire_lignes_csv(source)
    etiquettes = tuple(lignes[0][1:])
    n = len(etiquettes)
    if len(lignes) - 1 != n:
        raise ErreurDonnees(f"matrice non carrée : {n} colonnes et {len(lignes) - 1} lignes")
    A = np.full((n, n), np.nan)
    for i, ligne in enumerate(lignes[1:]):
        if len(ligne) != n + 1:
            ligne = ligne + [""] * (n + 1 - len(ligne))
        if ligne[0] != etiquettes[i]:
            raise ErreurDonnees(f"ligne {i + 2} : « {ligne[0]} » au lieu de « {etiquettes[i]} »")
        for j, cellule in enumerate(ligne[1:n + 1]):
            if cellule != "":
                A[i, j] = lire_nombre(cellule)
    return etiquettes, completer_reciproque(A)


def matrice_comparaison_vers_csv(etiquettes: Iterable[str], A: np.ndarray) -> str:
    etiquettes = list(etiquettes)
    sortie = io.StringIO()
    ecr = csv.writer(sortie, lineterminator="\n")
    ecr.writerow(["", *etiquettes])
    for nom, ligne in zip(etiquettes, np.asarray(A, dtype=float)):
        ecr.writerow([nom, *[format_nombre(x, fractions=True) for x in ligne]])
    return sortie.getvalue()


# --------------------------------------------------------------------------
# Tableaux génériques (rangs, scores, évaluations linguistiques…)
# --------------------------------------------------------------------------
def lire_tableau_csv(source: str | Path | io.IOBase | bytes,
                     colonnes_attendues: Iterable[str] | None = None) -> tuple[list[str], list[list[str]]]:
    """Lit un tableau CSV (en-tête + lignes de texte) et vérifie les colonnes attendues."""
    lignes = lire_lignes_csv(source)
    entete, corps = lignes[0], lignes[1:]
    if colonnes_attendues is not None:
        manquantes = [c for c in colonnes_attendues if c not in entete]
        if manquantes:
            raise ErreurDonnees(f"colonnes manquantes : {', '.join(manquantes)}")
    for num, ligne in enumerate(corps, start=2):
        if len(ligne) != len(entete):
            raise ErreurDonnees(f"ligne {num} : {len(ligne)} cellules au lieu de {len(entete)}")
    return entete, corps


def tableau_vers_csv(entete: Iterable[str], lignes: Iterable[Iterable[object]]) -> str:
    sortie = io.StringIO()
    ecr = csv.writer(sortie, lineterminator="\n")
    ecr.writerow(list(entete))
    for ligne in lignes:
        ecr.writerow([format_nombre(c) if isinstance(c, (int, float, np.floating)) else c for c in ligne])
    return sortie.getvalue()


# --------------------------------------------------------------------------
# Catalogue
# --------------------------------------------------------------------------
def charger_catalogue() -> list[dict]:
    """Liste des datasets décrits dans data/catalogue.json."""
    with FICHIER_CATALOGUE.open(encoding="utf-8") as f:
        return json.load(f)["datasets"]


def dataset(identifiant: str) -> dict:
    for d in charger_catalogue():
        if d["id"] == identifiant:
            return d
    raise ErreurDonnees(f"dataset « {identifiant} » absent du catalogue")


def chemin(relatif: str) -> Path:
    """Chemin absolu d'un fichier de data/ (toujours résolu depuis le dépôt)."""
    return DOSSIER_DONNEES / relatif


def fichiers_du_dataset(d: dict) -> list[str]:
    """Tous les chemins relatifs de fichiers cités par une entrée du catalogue."""
    res: list[str] = []

    def parcourir(v: object) -> None:
        if isinstance(v, str):
            res.append(v)
        elif isinstance(v, dict):
            for x in v.values():
                parcourir(x)
        elif isinstance(v, list):
            for x in v:
                parcourir(x)

    parcourir(d.get("fichiers", {}))
    return res
