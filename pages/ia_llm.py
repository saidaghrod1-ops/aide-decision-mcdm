import numpy as np
import pandas as pd
import streamlit as st

from core.classement.topsis import topsis
from core.commun import ErreurMethode
from ui import affichage as af
from ui import graphiques as gr
from ui.donnees import charger_tableau, editeur_tableau

af.entete("LLM extracteur sémantique de scores", "MCDM & IA générative")

with st.expander("Principe", expanded=True):
    st.markdown("Un LLM fine-tuné sur des annotations d'experts lit un texte $T$ (CV, rapport…) et produit un score par "
                "critère, qui devient une ligne de la matrice de décision :")
    st.latex(r"f_{LLM}(T, C) \rightarrow s = (s_1, \dots, s_n) \in [0, 10]^n,\qquad \text{Décision} = MCDM(s)")
    st.markdown("Pour un critère coût (prétentions salariales), le score est **inversé** par la rubrique : "
                "la colonne est alors un critère **+**.")
    st.caption("Ici, les scores sont simulés (aucun appel à un LLM) ; le candidat 1 reprend l'exemple de CV du cours.")

corpus = editeur_tableau("llm_corpus", charger_tableau("D8_llm_cv_corpus", "tableau"), nom_fichier="cv_scores.csv")
criteres = ["Expérience technique", "Communication / leadership", "Adéquation salariale (coût, score inversé)"]
try:
    X = corpus[criteres].apply(pd.to_numeric).to_numpy(float)
    candidats = [str(c) for c in corpus["candidat"]]
    if np.isnan(X).any():
        raise ValueError
except (ValueError, KeyError):
    st.error("Chaque candidat doit avoir trois scores.")
    st.stop()

c1, c2, c3 = st.columns(3)
w = np.array([c1.number_input("w expérience", 0.0, 1.0, 0.4, 0.05), c2.number_input("w communication", 0.0, 1.0, 0.3, 0.05),
              c3.number_input("w salaire", 0.0, 1.0, 0.3, 0.05)])
if abs(w.sum() - 1) > 1e-3:
    st.error(f"La somme des poids vaut {w.sum():.2f} au lieu de 1.")
    st.stop()
double = st.checkbox("Erreur à éviter : déclarer aussi l'adéquation salariale comme critère coût (double inversion)")
try:
    r = topsis(np.maximum(X, 1e-9), np.array([1, 1, -1 if double else 1]), w)
    juste = topsis(np.maximum(X, 1e-9), np.ones(3, int), w)
except ErreurMethode as exc:
    st.error(str(exc))
    st.stop()
af.classement(candidats, {"RC* (TOPSIS)": r.RC}, r.rangs)
gr.afficher(gr.barres(candidats, r.RC, "Score TOPSIS des candidats", "RC*"))
if double:
    st.warning(f"Avec la double inversion, le meilleur candidat devient **{candidats[int(np.argmax(r.RC))]}** au lieu de "
               f"**{candidats[int(np.argmax(juste.RC))]}** : des prétentions élevées seraient récompensées.")
else:
    st.success(f"Meilleur candidat : **{candidats[int(np.argmax(r.RC))]}**.")
