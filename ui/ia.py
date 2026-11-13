"""Éléments partagés des pages « MCDM et IA » : jeu D6 et choix des poids initiaux."""

from __future__ import annotations

import numpy as np
import streamlit as st

from core.commun import ErreurMethode
from core.ponderation.ahp import poids_ahp
from core.ponderation.critic import poids_critic
from core.ponderation.entropie import poids_entropie
from ui.donnees import editeur_comparaison

# Matrice AHP d'exemple pour les 4 critères (Qualité, Prix, Délai, RSE), cohérente (CR ≈ 0,008),
# dans l'ordre de préférence de l'exemple BWM du cours (p. 37).
AHP_EXEMPLE = np.array([[1, 3, 4, 8], [1 / 3, 1, 2, 3], [1 / 4, 1 / 2, 1, 2], [1 / 8, 1 / 3, 1 / 2, 1]])


def choisir_poids(cle: str, X, sens, criteres, defaut: str = "AHP") -> np.ndarray | None:
    """Pondération par MCDM (AHP, CRITIC, entropie ou saisie manuelle), première étape des scénarios p. 91–92."""
    sources = ["AHP", "CRITIC", "Entropie", "Saisie manuelle"]
    source = st.radio("Méthode de pondération", sources, index=sources.index(defaut), horizontal=True, key=f"{cle}__source")
    try:
        if source == "AHP":
            defaut_A = AHP_EXEMPLE if len(criteres) == 4 else np.ones((len(criteres),) * 2)
            res = editeur_comparaison(f"{cle}__ahp", criteres, defaut_A, titre="Matrice de comparaison des critères (AHP)")
            if res is None:
                return None
            p = poids_ahp(res[1])
            if not p.coherent:
                st.error(f"CR = {p.CR:.3f} ≥ 0,1 : matrice non consistante.")
                return None
            st.caption(f"CR = {p.CR:.4f} < 0,1 : matrice consistante.")
            return p.w
        if source == "CRITIC":
            return poids_critic(X, sens).w
        if source == "Entropie":
            return poids_entropie(X, sens, "lineaire").w
    except ErreurMethode as exc:
        st.error(str(exc))
        return None
    colonnes = st.columns(len(criteres))
    w = np.array([c.number_input(nom, 0.0, 1.0, round(1 / len(criteres), 3), 0.01, key=f"{cle}__w{j}")
                  for j, (c, nom) in enumerate(zip(colonnes, criteres))])
    if abs(w.sum() - 1) > 1e-3:
        st.error(f"La somme des poids vaut {w.sum():.3f} au lieu de 1.")
        return None
    return w
