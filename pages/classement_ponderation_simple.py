import numpy as np
import pandas as pd
import streamlit as st

from core.classement.ponderation_simple import waspas
from core.commun import ErreurMethode
from ui import affichage as af
from ui import graphiques as gr
from ui.donnees import charger_matrice, choisir_dataset, editeur_matrice, options_matrices, poids_valides

af.entete("WSM, WPM et WASPAS", "Classement compensatoire")

with st.expander("Théorie", expanded=True):
    st.markdown("Normalisation linéaire selon le sens du critère, puis agrégation :")
    st.latex(r"r_{ij}=\frac{x_{ij}}{\max_i x_{ij}}\ (+),\qquad r_{ij}=\frac{\min_i x_{ij}}{x_{ij}}\ (-)")
    st.latex(r"\text{WSM : } Q_i^{(1)}=\sum_j w_j r_{ij}\qquad \text{WPM : } Q_i^{(2)}=\prod_j r_{ij}^{\,w_j}\qquad "
             r"\text{WASPAS : } Q_i=\lambda Q_i^{(1)}+(1-\lambda)Q_i^{(2)}")
    st.markdown("La meilleure alternative a le score le plus élevé ; λ = 0,5 donne $Q_i = \\frac12(Q_i^{(1)} + Q_i^{(2)})$.")

ident = choisir_dataset("wsm", options_matrices("C4_waspas_machines", "D1_selection_modele_ml", "D2_fournisseurs_5offres"))
M = editeur_matrice(f"wsm_{ident}", charger_matrice(ident), nom_fichier="ponderation_simple.csv")
if M is None:
    st.stop()
w = poids_valides(M, f"wsm_{ident}")
if w is None:
    st.stop()
lam = st.slider("λ", 0.0, 1.0, 0.5, 0.05, help="λ = 1 : WSM ; λ = 0 : WPM.")
try:
    r = waspas(M.valeurs, M.sens, w, lam)
except ErreurMethode as exc:
    st.error(f"La méthode ne s'applique pas : {exc}.")
    st.stop()

with af.etape("Matrice normalisée rᵢⱼ"):
    af.matrice(r.R, M.alternatives, M.criteres)
st.dataframe(pd.DataFrame({"Q⁽¹⁾ WSM": np.round(r.Q1, 4), "Q⁽²⁾ WPM": np.round(r.Q2, 4), "Q WASPAS": np.round(r.Q, 4),
                           "Rang": r.rangs_Q}, index=M.alternatives).sort_values("Rang"))
gr.afficher(gr.barres_groupees(M.alternatives, {"WSM": r.Q1, "WPM": r.Q2, f"WASPAS (λ = {lam:.2f})": r.Q}, None))
st.success(f"Meilleure alternative selon WASPAS : **{M.alternatives[int(np.argmax(r.Q))]}** (Q = {r.Q.max():.4f}).")
