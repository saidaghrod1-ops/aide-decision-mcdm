import numpy as np
import pandas as pd
import streamlit as st

from core.classement.ponderation_simple import waspas
from core.classement.topsis import topsis
from core.classement.vikor import vikor
from core.commun import ErreurMethode, rangs, rangs_moyens
from core.robustesse.spearman import matrice_spearman, valeur_critique
from ui import affichage as af
from ui import graphiques as gr
from ui.donnees import OPTIONS_MATRICES, charger_matrice, choisir_dataset, editeur_matrice, poids_valides

af.entete("Comparaison multi-méthodes", "Robustesse et comparaison des méthodes",
          "Appliquer plusieurs méthodes au même problème et comparer les classements (corrélation de Spearman).")

ident = choisir_dataset("comparaison", OPTIONS_MATRICES)
M = editeur_matrice(f"comparaison_{ident}", charger_matrice(ident), nom_fichier="comparaison.csv")
if M is None:
    st.stop()
w = poids_valides(M, f"comparaison_{ident}")
if w is None:
    st.stop()

calculs = {"WSM": lambda: (waspas(M.valeurs, M.sens, w, 1.0).Q1, True),
           "WPM": lambda: (waspas(M.valeurs, M.sens, w, 0.0).Q2, True),
           "WASPAS": lambda: (waspas(M.valeurs, M.sens, w, 0.5).Q, True),
           "TOPSIS": lambda: (topsis(M.valeurs, M.sens, w).RC, True),
           "VIKOR": lambda: (vikor(M.valeurs, M.sens, w, 0.5).Q, False)}
classements = {}
for nom, calcul in calculs.items():
    try:
        s, plus_grand = calcul()
        classements[nom] = (rangs(s, plus_grand), rangs_moyens(s, plus_grand))
    except ErreurMethode as exc:
        st.caption(f"{nom} ne s'applique pas à ces données : {exc}.")
if len(classements) < 2:
    st.error("Il faut au moins deux méthodes applicables.")
    st.stop()

st.dataframe(pd.DataFrame({m: r[0] for m, r in classements.items()}, index=M.alternatives))
noms, R = matrice_spearman({m: r[1] for m, r in classements.items()})
gr.afficher(gr.heatmap(R, noms, noms, "Spearman entre les classements", divergente=True, zmin=-1, zmax=1, decimales=2))
c = valeur_critique(M.m)
if c is not None:
    st.caption(f"Seuil de significativité pour n = {M.m} (5 %) : ρ ≥ {c:.3f}.")
meilleures = {m: M.alternatives[int(np.argmin(r[0]))] for m, r in classements.items()}
if len(set(meilleures.values())) == 1:
    st.success(f"Toutes les méthodes désignent **{next(iter(meilleures.values()))}** : choix robuste.")
else:
    st.warning("Les méthodes ne désignent pas la même meilleure alternative : "
               + " ; ".join(f"{m} → {a}" for m, a in meilleures.items()) + ".")
