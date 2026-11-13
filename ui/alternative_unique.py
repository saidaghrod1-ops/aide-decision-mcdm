"""Éditeur partagé du dossier unique (critère, sens, unité, poids, valeur, bornes) — pages MAUT et distance à l'idéal."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import streamlit as st

from core.donnees import lire_sens
from ui.donnees import charger_tableau, editeur_tableau


@dataclass(frozen=True, eq=False)
class Dossier:
    criteres: tuple[str, ...]
    sens: np.ndarray
    unites: tuple[str, ...]
    poids: np.ndarray
    valeurs: np.ndarray
    xmin: np.ndarray
    xmax: np.ndarray


def editeur_dossier(cle: str, identifiant: str = "D5_credit") -> Dossier | None:
    table = editeur_tableau(cle, charger_tableau(identifiant, "dossier"), titre="Alternative à évaluer et bornes de référence",
                            colonnes_attendues=["critere", "sens", "unite", "poids", "valeur", "x_min", "x_max"],
                            nom_fichier="dossier.csv",
                            column_config={"sens": st.column_config.SelectboxColumn("sens", options=["+", "-"]),
                                           "poids": st.column_config.NumberColumn("poids", min_value=0.0, format="%.3f")})
    try:
        dossier = Dossier(tuple(str(c) for c in table["critere"]), np.array([lire_sens(s) for s in table["sens"]]),
                          tuple("" if pd.isna(u) else str(u) for u in table["unite"]),
                          *(pd.to_numeric(table[c]).to_numpy(float) for c in ("poids", "valeur", "x_min", "x_max")))
    except (ValueError, KeyError) as exc:
        st.error(f"Tableau invalide : {exc}")
        return None
    if any(np.isnan(v).any() for v in (dossier.poids, dossier.valeurs, dossier.xmin, dossier.xmax)):
        st.error("Chaque critère doit avoir un poids, une valeur et deux bornes.")
        return None
    if abs(dossier.poids.sum() - 1) > 1e-3:
        st.error(f"La somme des poids vaut {dossier.poids.sum():.4f} au lieu de 1.")
        return None
    st.caption("Les bornes x_min et x_max sont des références fixées par la politique (experts, réglementation, "
               "objectifs stratégiques) : le cours ne les précise pas (E-31).")
    return dossier
