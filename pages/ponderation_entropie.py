import numpy as np
import pandas as pd
import streamlit as st

from core.commun import NORMALISATIONS, ErreurMethode
from core.ponderation.entropie import poids_entropie
from ui import affichage as af
from ui import graphiques as gr
from ui.donnees import charger_matrice, choisir_dataset, editeur_matrice
from ui.etat import publier_poids

af.entete("Méthode de l'entropie", "Pondération objective", "44–45")

with st.expander("Théorie (cours p. 44–45)", expanded=False):
    st.markdown("""
L'entropie mesure le degré d'incertitude (de désordre) de la colonne d'un critère.
**Entropie élevée** (proche de 1) : valeurs uniformément réparties, le critère distingue mal les alternatives → **poids faible**.
**Entropie faible** : valeurs dispersées, forte capacité de discrimination → **poids élevé**.
Pour une matrice de décision normalisée $D = (d_{ij})$ à $m$ alternatives et $n$ critères :
""")
    st.latex(r"p_{ij} = \frac{d_{ij}}{\sum_{i=1}^{m} d_{ij}},\qquad E_j = -k\sum_{i=1}^{m} p_{ij}\ln p_{ij},\qquad k=\frac{1}{\ln m}")
    st.latex(r"w_j = \frac{1-E_j}{n-\sum_{k=1}^{n}E_k}")
    st.markdown("$1 - E_j$ est le **degré de diversification** du critère j : la quantité d'information utile qu'il apporte.")
    af.encadre_ecart("E-10", "le cours ne précise pas la normalisation de D. Par défaut, la normalisation linéaire du cours "
                             "(p. 55) est utilisée ; les deux autres normalisations du cours sont proposées. "
                             "Convention : 0 · ln 0 = 0.")

ident = choisir_dataset("entropie", {
    "Exercice CRITIC : machines de découpage (p. 51)": "C3_critic_machines_decoupe",
    "Sélection d'un modèle de ML (jeu créé, p. 14)": "D1_selection_modele_ml",
    "Choisir un fournisseur parmi 5 offres (jeu créé)": "D2_fournisseurs_5offres",
    "Exercice WASPAS : machines (p. 58)": "C4_waspas_machines",
    "Portefeuille Vuković et al. (p. 18)": "C12_portefeuille_vukovic2020",
})
M = editeur_matrice(f"entropie_{ident}", charger_matrice(ident), avec_poids=False, nom_fichier="entropie.csv")
if M is None:
    st.stop()
normalisation = st.radio("Normalisation de D", list(NORMALISATIONS), format_func=NORMALISATIONS.get, horizontal=True)
try:
    r = poids_entropie(M.valeurs, M.sens, normalisation)
except ErreurMethode as exc:
    st.error(f"La méthode ne s'applique pas : {exc}. Essayez la normalisation min-max (p. 47), "
             "qui ramène les valeurs dans [0, 1].")
    st.stop()

alt, crit = M.alternatives, M.criteres
with af.etape("Étape 1 — Matrice normalisée D"):
    af.matrice(r.D, alt, crit)
with af.etape("Étape 2 — pᵢⱼ = dᵢⱼ / Σᵢ dᵢⱼ"):
    af.matrice(r.P, alt, crit)
with af.etape("Étape 3 — pᵢⱼ ln pᵢⱼ et entropie Eⱼ", ouverte=True):
    af.matrice(r.PlnP, alt, crit)
    st.markdown(f"k = 1 / ln({M.m}) = {r.k:.4f}")
    af.vecteur(r.E, crit, "Eⱼ")
with af.etape("Étape 4 — Degré de diversification et poids", ouverte=True):
    st.dataframe(pd.DataFrame({"1 − Eⱼ": np.round(r.diversification, 4), "wⱼ": np.round(r.w, 4)}, index=crit).T)
    st.latex(rf"n - \sum_k E_k = {M.n} - {r.E.sum():.4f} = {M.n - r.E.sum():.4f}")

c1, c2 = st.columns(2)
with c1:
    gr.afficher(gr.barres(crit, r.E, "Entropie Eⱼ", "Eⱼ"))
with c2:
    gr.afficher(gr.barres(crit, r.w, "Poids wⱼ", "wⱼ"))
j = int(np.argmax(r.w))
st.success(f"**{crit[j]}** a l'entropie la plus faible ({r.E[j]:.4f}) : c'est le critère le plus discriminant, "
           f"donc le plus lourd (w = {r.w[j]:.4f}).")
st.caption("Limite rappelée p. 22 : les poids objectifs ignorent l'importance stratégique réelle des critères.")

with st.expander("Sensibilité au choix de la normalisation"):
    lignes = {}
    for cle, nom in NORMALISATIONS.items():
        try:
            lignes[nom] = np.round(poids_entropie(M.valeurs, M.sens, cle).w, 4)
        except ErreurMethode as exc:
            st.caption(f"{nom} : non applicable ({exc}).")
    if lignes:
        st.dataframe(pd.DataFrame(lignes, index=crit).T)

if st.button("Utiliser ces poids dans la page « Chaînage »", icon=":material/link:"):
    publier_poids(f"Entropie — {ident}", M.criteres, r.w)
    st.toast("Poids de l'entropie envoyés vers la page de chaînage.")
