import numpy as np
import pandas as pd
import streamlit as st

from core.commun import ErreurMethode
from core.ia.classification import CLASSES, entrainer, etiqueter, scores_wsm_reference, seuils_quartiles
from ui import affichage as af
from ui import graphiques as gr
from ui.donnees import charger_matrice, editeur_matrice
from ui.ia import choisir_poids

af.entete("De MCDM à la classification supervisée", "MCDM + IA")

with st.expander("Démarche", expanded=True):
    st.markdown("1. Pondération par MCDM (AHP, CRITIC, entropie).  \n"
                "2. Score de chaque alternative par une méthode additive : $S_i = \\sum_j w_j r_{ij}$.  \n"
                "3. Étiquettes A, B, C, D selon $S_i$ (quartiles) : base « critères → classe ».  \n"
                "4. Apprentissage d'un modèle de ML (Random Forest).  \n"
                "5. Prédiction de la classe d'une nouvelle alternative.")

st.caption("Jeu créé : historique de 300 fournisseurs (qualité, prix, délai, RSE).")
M = editeur_matrice("ia_classif", charger_matrice("D6_fournisseurs_historique"), avec_poids=False, nom_fichier="fournisseurs.csv")
if M is None:
    st.stop()
w = choisir_poids("ia_classif_poids", M.valeurs, M.sens, M.criteres)
if w is None:
    st.stop()
try:
    S = scores_wsm_reference(M.valeurs, M.sens, w)
except ErreurMethode as exc:
    st.error(str(exc))
    st.stop()
seuils = seuils_quartiles(S)
classes = etiqueter(S, seuils)
st.markdown(f"Poids : {af.format_poids(M.criteres, w, 3)}  \nSeuils des classes : A ≥ {seuils[0]:.3f} ; "
            f"B ≥ {seuils[1]:.3f} ; C ≥ {seuils[2]:.3f} ; D sinon.")


@st.cache_resource(show_spinner="Apprentissage du modèle…")
def _apprendre(X, etiquettes):
    return entrainer(X, etiquettes, n_arbres=150)


res = _apprendre(M.valeurs, tuple(classes))
c1, c2 = st.columns([1, 2])
c1.metric("Exactitude sur le jeu de test", f"{res.exactitude:.3f}")
with c2:
    gr.afficher(gr.heatmap(res.confusion, [f"prédit {c}" for c in CLASSES], [f"réel {c}" for c in CLASSES],
                           "Matrice de confusion", decimales=0))

st.subheader("Nouvelle alternative")
cols = st.columns(M.n)
nouvelle = np.array([col.number_input(c, value=float(np.median(M.valeurs[:, j])), key=f"ia_new_{j}")
                     for j, (col, c) in enumerate(zip(cols, M.criteres))])
try:
    s_nouv = float(scores_wsm_reference(nouvelle[None, :], M.sens, w, reference=M.valeurs)[0])
except ErreurMethode as exc:
    st.error(str(exc))
    st.stop()
c1, c2 = st.columns(2)
c1.metric("Classe prédite (Random Forest)", res.modele.predict(nouvelle[None, :])[0])
c2.metric("Classe par le calcul MCDM", etiqueter([s_nouv], seuils)[0], f"S = {s_nouv:.4f}", delta_color="off")
