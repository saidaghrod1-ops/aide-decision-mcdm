import numpy as np
import pandas as pd
import streamlit as st

from core.commun import ErreurMethode
from core.ponderation.critic import poids_critic
from ui import affichage as af
from ui import graphiques as gr
from ui.donnees import charger_matrice, choisir_dataset, editeur_matrice, options_matrices
from ui.etat import publier_poids

af.entete("Méthode CRITIC", "Pondération objective")

with st.expander("Théorie", expanded=True):
    st.latex(r"r_{ij}=\frac{x_{ij}-\min x_j}{\max x_j-\min x_j}\ (+),\qquad r_{ij}=\frac{\max x_j-x_{ij}}{\max x_j-\min x_j}\ (-)")
    st.latex(r"\sigma_j=\sqrt{\frac{1}{m-1}\sum_i (r_{ij}-\bar r_j)^2},\qquad C_j=\sigma_j\sum_{k}(1-\lvert\rho_{jk}\rvert),"
             r"\qquad w_j=\frac{C_j}{\sum_j C_j}")
    st.markdown("$\\rho_{jk}$ : corrélation entre les critères j et k. Plus un critère est **dispersé** et **peu corrélé** "
                "aux autres, plus il est important.")

ident = choisir_dataset("critic", options_matrices("C3_critic_machines_decoupe", "D1_selection_modele_ml",
                                                   "D2_fournisseurs_5offres"))
M = editeur_matrice(f"critic_{ident}", charger_matrice(ident), avec_poids=False, nom_fichier="critic.csv")
if M is None:
    st.stop()
try:
    r = poids_critic(M.valeurs, M.sens)
except ErreurMethode as exc:
    st.error(str(exc))
    st.stop()

with af.etape("Calcul : matrice normalisée rᵢⱼ"):
    af.matrice(r.R, M.alternatives, M.criteres)
gr.afficher(gr.heatmap(r.rho, M.criteres, M.criteres, "Corrélations ρⱼₖ", divergente=True, zmin=-1, zmax=1))
st.dataframe(pd.DataFrame({"σⱼ": np.round(r.sigma, 4), "Σₖ (1 − |ρⱼₖ|)": np.round(r.conflit, 4), "Cⱼ": np.round(r.C, 4),
                           "Poids wⱼ": np.round(r.w, 4)}, index=M.criteres).T)
gr.afficher(gr.barres(M.criteres, r.w, "Poids", "wⱼ"))
j = int(np.argmax(r.w))
st.success(f"Critère le plus lourd : **{M.criteres[j]}** (w = {r.w[j]:.4f}).")
if st.button("Utiliser ces poids dans la page « Chaînage »", icon=":material/link:"):
    publier_poids(f"CRITIC — {ident}", M.criteres, r.w)
    st.toast("Poids envoyés vers la page de chaînage.")
