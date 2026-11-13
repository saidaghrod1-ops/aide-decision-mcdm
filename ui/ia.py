"""Choix des poids initiaux des pages « MCDM et IA »."""

from __future__ import annotations

import numpy as np
import streamlit as st

from core.commun import ErreurMethode
from core.ponderation.ahp import poids_ahp
from core.ponderation.critic import poids_critic
from core.ponderation.entropie import poids_entropie
from ui.donnees import editeur_comparaison

# Matrice AHP d'exemple pour les 4 critères (Qualité, Prix, Délai, RSE), cohérente (CR ≈ 0,008),
# dans l'ordre de préférence de l'exemple BWM du cours.
AHP_EXEMPLE = np.array([[1, 3, 4, 8], [1 / 3, 1, 2, 3], [1 / 4, 1 / 2, 1, 2], [1 / 8, 1 / 3, 1 / 2, 1]])


def choisir_poids(cle: str, X, sens, criteres) -> np.ndarray | None:
    """Étape 1 des scénarios : pondération par MCDM (AHP, CRITIC ou entropie)."""
    source = st.radio("Méthode de pondération", ["AHP", "CRITIC", "Entropie"], horizontal=True, key=f"{cle}__source")
    try:
        if source == "AHP":
            defaut = AHP_EXEMPLE if len(criteres) == 4 else np.ones((len(criteres),) * 2)
            res = editeur_comparaison(f"{cle}__ahp", criteres, defaut)
            if res is None:
                return None
            p = poids_ahp(res[1])
            if not p.coherent:
                st.error(f"CR = {p.CR:.3f} ≥ 0,1 : matrice non consistante.")
                return None
            return p.w
        if source == "CRITIC":
            return poids_critic(X, sens).w
        return poids_entropie(X, sens, "lineaire").w
    except ErreurMethode as exc:
        st.error(str(exc))
        return None
