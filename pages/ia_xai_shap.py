import numpy as np
import pandas as pd
import streamlit as st

from core.commun import ErreurMethode, rangs_moyens
from core.ia.classification import scores_wsm_reference
from core.ia.shapley import entrainer_regresseur, mise_a_jour_poids, poids_shap, shap_exact
from core.robustesse.spearman import spearman
from ui import affichage as af
from ui import graphiques as gr
from ui.donnees import charger_matrice, editeur_matrice
from ui.ia import choisir_poids

af.entete("XAI (SHAP) comme approche de pondération", "MCDM + IA")

with st.expander("Démarche", expanded=True):
    st.markdown("1. Poids initiaux par MCDM (AHP…). 2. Scores $S_i = \\sum_j w_j r_{ij}$. "
                "3. Modèle de ML $\\hat S = f(x)$. 4. Explication SHAP. 5. Mise à jour des poids :")
    st.latex(r"\hat S = \phi_0 + \sum_j \phi_j,\qquad w_j^{new} = \alpha\, w_j^{AHP} + (1-\alpha)\,\frac{\lvert\phi_j\rvert}{\sum_j \lvert\phi_j\rvert}")
    st.caption("|φⱼ| : moyenne des contributions absolues sur les alternatives ; valeurs SHAP calculées exactement.")

st.caption("Jeu créé : historique de 300 fournisseurs.")
M = editeur_matrice("ia_shap", charger_matrice("D6_fournisseurs_historique"), avec_poids=False, nom_fichier="fournisseurs.csv")
if M is None:
    st.stop()
w_init = choisir_poids("ia_shap_poids", M.valeurs, M.sens, M.criteres)
if w_init is None:
    st.stop()
try:
    S = scores_wsm_reference(M.valeurs, M.sens, w_init)
except ErreurMethode as exc:
    st.error(str(exc))
    st.stop()


@st.cache_data(show_spinner="Apprentissage du modèle et calcul des valeurs SHAP…")
def _shap(X, S):
    modele, r2 = entrainer_regresseur(X, S, n_arbres=150)
    rng = np.random.default_rng(0)
    idx = rng.choice(len(X), size=min(100, len(X)), replace=False)
    fond = X[rng.choice(len(X), size=min(40, len(X)), replace=False)]
    phi, phi0 = shap_exact(modele.predict, X[idx], fond)
    return r2, phi, phi0


r2, phi, phi0 = _shap(M.valeurs, S)
alpha = st.slider("α", 0.0, 1.0, 0.5, 0.05)
w_shap = poids_shap(phi)
w_new = mise_a_jour_poids(w_init, w_shap, alpha)
st.dataframe(pd.DataFrame({"w initiaux": np.round(w_init, 4), "|φⱼ| / Σ|φ|": np.round(w_shap, 4),
                           f"w new (α = {alpha:.2f})": np.round(w_new, 4)}, index=M.criteres).T)
gr.afficher(gr.barres_groupees(M.criteres, {"w initiaux": w_init, "Poids SHAP": w_shap, "w new": w_new}, None, "wⱼ"))
S_new = scores_wsm_reference(M.valeurs, M.sens, w_new)
c1, c2 = st.columns(2)
c1.metric("R² du modèle Ŝ = f(x)", f"{r2:.3f}")
c2.metric("Spearman entre les classements (w initiaux / w new)", f"{spearman(rangs_moyens(S), rangs_moyens(S_new)):.4f}")
st.caption("|φⱼ| dépend du poids du critère mais aussi de sa dispersion dans les données.")
