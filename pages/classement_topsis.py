import numpy as np
import pandas as pd
import streamlit as st

from core.classement.topsis import topsis
from core.commun import ErreurMethode
from ui import affichage as af
from ui import graphiques as gr
from ui.donnees import charger_matrice, choisir_dataset, editeur_matrice, options_matrices, poids_valides

af.entete("TOPSIS", "Classement compensatoire",
          "Choisir l'alternative la plus proche de la solution idéale positive et la plus éloignée de la solution "
          "idéale négative.")

with st.expander("Théorie", expanded=True):
    st.latex(r"r_{ij}=\frac{x_{ij}}{\sqrt{\sum_i x_{ij}^2}},\qquad v_{ij}=w_j\,r_{ij}")
    st.markdown("$I^+$ : meilleure valeur $v_j^*$ de chaque critère (max si +, min si −) ; $I^-$ : la plus mauvaise $v_j'$.")
    st.latex(r"S_i^+=\sqrt{\sum_j (v_j^*-v_{ij})^2},\qquad S_i^-=\sqrt{\sum_j (v_j'-v_{ij})^2},\qquad "
             r"RC_i^*=\frac{S_i^-}{S_i^-+S_i^+}")
    st.markdown("La meilleure alternative a le $RC_i^*$ le plus élevé.")

ident = choisir_dataset("topsis", options_matrices("C5_topsis_voitures", "D1_selection_modele_ml", "D2_fournisseurs_5offres",
                                                   "C12_portefeuille_vukovic2020"))
M = editeur_matrice(f"topsis_{ident}", charger_matrice(ident), nom_fichier="topsis.csv")
if M is None:
    st.stop()
w = poids_valides(M, f"topsis_{ident}")
if w is None:
    st.stop()
try:
    r = topsis(M.valeurs, M.sens, w)
except ErreurMethode as exc:
    st.error(str(exc))
    st.stop()

with af.etape("Matrices normalisée et pondérée"):
    af.matrice(r.R, M.alternatives, M.criteres)
    af.matrice(r.V, M.alternatives, M.criteres)
st.dataframe(pd.DataFrame(np.round([r.I_plus, r.I_moins], 4), index=["I⁺", "I⁻"], columns=M.criteres))
af.classement(M.alternatives, {"S⁺": r.S_plus, "S⁻": r.S_moins, "RC*": r.RC}, r.rangs)
gr.afficher(gr.barres(M.alternatives, r.RC, "Coefficient de proximité RC*", "RC*"))
st.success(f"**{M.alternatives[int(np.argmax(r.RC))]}** a le RC* le plus élevé ({r.RC.max():.4f}) : c'est la meilleure alternative.")
