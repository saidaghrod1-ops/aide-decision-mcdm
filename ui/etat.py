"""État partagé entre les pages (st.session_state), notamment les poids du chaînage."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import streamlit as st

_CLE_POIDS = "poids_publies"


@dataclass(frozen=True, eq=False)
class PoidsPublies:
    source: str                 # ex. « BWM — Choisir un fournisseur »
    criteres: tuple[str, ...]
    poids: np.ndarray


def publier_poids(source: str, criteres: tuple[str, ...], poids: np.ndarray) -> None:
    """Rend un vecteur de poids disponible pour la page de chaînage."""
    st.session_state.setdefault(_CLE_POIDS, {})[source] = PoidsPublies(source, tuple(criteres), np.asarray(poids, float))


def poids_publies() -> dict[str, PoidsPublies]:
    return dict(st.session_state.get(_CLE_POIDS, {}))
