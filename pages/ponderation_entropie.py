import numpy as np
import streamlit as st

from core.commun import NORMALISATIONS, ErreurMethode
from core.ponderation.entropie import poids_entropie
from ui import affichage as af
from ui import graphiques as gr
from ui.donnees import charger_matrice, choisir_dataset, editeur_matrice, options_matrices
from ui.etat import publier_poids

af.entete("Méthode de l'entropie", "Pondération objective")

with st.expander("Théorie", expanded=True):
    st.markdown("L'entropie mesure le désordre de la colonne d'un critère : **entropie élevée** (valeurs uniformes) → "
                "critère peu discriminant → **poids faible** ; **entropie faible** (valeurs dispersées) → **poids élevé**.")
    st.latex(r"p_{ij}=\frac{d_{ij}}{\sum_i d_{ij}},\qquad E_j=-\frac{1}{\ln m}\sum_i p_{ij}\ln p_{ij},\qquad "
             r"w_j=\frac{1-E_j}{n-\sum_k E_k}")
    st.caption("D est la matrice normalisée (normalisation linéaire par défaut) ; convention 0 · ln 0 = 0.")

ident = choisir_dataset("entropie", options_matrices("C3_critic_machines_decoupe", "D1_selection_modele_ml",
                                                     "D2_fournisseurs_5offres"))
M = editeur_matrice(f"entropie_{ident}", charger_matrice(ident), avec_poids=False, nom_fichier="entropie.csv")
if M is None:
    st.stop()
normalisation = st.radio("Normalisation", list(NORMALISATIONS), format_func=NORMALISATIONS.get, horizontal=True)
try:
    r = poids_entropie(M.valeurs, M.sens, normalisation)
except ErreurMethode as exc:
    st.error(f"La méthode ne s'applique pas : {exc}.")
    st.stop()

with af.etape("Calcul : matrice normalisée D et probabilités pᵢⱼ"):
    af.matrice(r.D, M.alternatives, M.criteres)
    af.matrice(r.P, M.alternatives, M.criteres)
af.vecteur(r.E, M.criteres, "Entropie Eⱼ")
af.vecteur(r.w, M.criteres, "Poids wⱼ")
gr.afficher(gr.barres(M.criteres, r.w, "Poids", "wⱼ"))
j = int(np.argmax(r.w))
st.success(f"**{M.criteres[j]}** a l'entropie la plus faible ({r.E[j]:.4f}) : c'est le critère le plus discriminant "
           f"(w = {r.w[j]:.4f}).")
if st.button("Utiliser ces poids dans la page « Chaînage »", icon=":material/link:"):
    publier_poids(f"Entropie — {ident}", M.criteres, r.w)
    st.toast("Poids envoyés vers la page de chaînage.")
