"""Éléments d'affichage communs : en-tête de page, étapes pas à pas, tableaux, notes."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd
import streamlit as st


def entete(titre: str, categorie: str, resume: str | None = None) -> None:
    st.title(titre)
    st.caption(categorie)
    if resume:
        st.write(resume)


def note(texte: str) -> None:
    """Remarque courte : écart ou précision par rapport à la formule imprimée dans le cours."""
    st.caption(f"ℹ️ {texte}")


def etape(titre: str, ouverte: bool = False):
    """Dépliant d'une étape du calcul (usage : `with etape("Étape 1 — …"):`)."""
    return st.expander(titre, expanded=ouverte)


def matrice(M, lignes: Sequence[str], colonnes: Sequence[str], decimales: int = 4) -> None:
    st.dataframe(pd.DataFrame(np.round(np.asarray(M, float), decimales), index=list(lignes), columns=list(colonnes)))


def vecteur(v, etiquettes: Sequence[str], nom: str, decimales: int = 4) -> None:
    st.dataframe(pd.DataFrame({nom: np.round(np.asarray(v, float), decimales)}, index=list(etiquettes)).T)


def classement(alternatives: Sequence[str], colonnes: dict, rangs, decimales: int = 4) -> pd.DataFrame:
    """Tableau final trié par rang : colonnes = {nom: valeurs}."""
    df = pd.DataFrame({"Alternative": list(alternatives),
                       **{k: np.round(np.asarray(v, float), decimales) for k, v in colonnes.items()},
                       "Rang": np.asarray(rangs, int)})
    df = df.sort_values(["Rang", "Alternative"]).reset_index(drop=True)
    st.dataframe(df, hide_index=True)
    return df


def format_poids(criteres: Sequence[str], w, decimales: int = 4) -> str:
    return " ; ".join(f"{c} = {x:.{decimales}f}" for c, x in zip(criteres, w))


def triplets(T, decimales: int = 3) -> list[str]:
    """Formate un tableau (… × 3) de nombres flous en « (a ; b ; c) »."""
    T = np.asarray(T, float)
    return [f"({t[0]:.{decimales}f} ; {t[1]:.{decimales}f} ; {t[2]:.{decimales}f})" for t in T.reshape(-1, 3)]


def matrice_triplets(T, lignes: Sequence[str], colonnes: Sequence[str], decimales: int = 3) -> None:
    T = np.asarray(T, float)
    textes = np.array(triplets(T, decimales), dtype=object).reshape(T.shape[0], T.shape[1])
    st.dataframe(pd.DataFrame(textes, index=list(lignes), columns=list(colonnes)))
