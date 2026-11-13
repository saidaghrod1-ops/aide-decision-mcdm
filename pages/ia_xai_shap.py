import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from core.commun import ErreurMethode, rangs, rangs_moyens
from core.ia.classification import scores_wsm_reference
from core.ia.shapley import entrainer_regresseur, importance_globale, mise_a_jour_poids, poids_shap, shap_exact
from core.robustesse.spearman import spearman
from ui import affichage as af
from ui import graphiques as gr
from ui.donnees import charger_matrice, choisir_dataset, editeur_matrice
from ui.ia import choisir_poids

af.entete("XAI (SHAP) comme approche de pondération", "MCDM + IA — scénario 2", "92")

with st.expander("Démarche (cours p. 92)", expanded=True):
    st.markdown("1. Pondération par MCDM (AHP, CRITIC, entropie…). 2. Scores $S_i$ par une méthode MCDM additive.  \n"
                "3. Apprentissage d'un modèle de ML $\\hat S = f(x)$ (critères → score MCDM ; ici Random Forest).  \n"
                "4. Explication avec SHAP : contribution de chaque critère, importance locale et globale :")
    st.latex(r"\hat S = \phi_0 + \sum_j \phi_j")
    st.markdown("5. Mise à jour des poids :")
    st.latex(r"w_j^{new} = \alpha\, w_j^{AHP} + (1-\alpha)\,\frac{\lvert\phi_j\rvert}{\sum_j \lvert\phi_j\rvert}")
    st.caption("Les valeurs SHAP sont calculées exactement (énumération des 2ⁿ coalitions de critères, espérance sur un "
               "échantillon de fond). |φⱼ| global = moyenne des |φᵢⱼ| sur les alternatives expliquées (E-16).")

ident = choisir_dataset("ia_shap", {"Historique de 300 fournisseurs (jeu créé)": "D6_fournisseurs_historique"})
M = editeur_matrice(f"ia_shap_{ident}", charger_matrice(ident), avec_poids=False, nom_fichier="fournisseurs.csv")
if M is None:
    st.stop()

st.header("1–2. Poids initiaux et scores MCDM")
w_init = choisir_poids("ia_shap_poids", M.valeurs, M.sens, M.criteres, defaut="AHP")
if w_init is None:
    st.stop()
try:
    S = scores_wsm_reference(M.valeurs, M.sens, w_init)
except ErreurMethode as exc:
    st.error(str(exc))
    st.stop()

st.header("3. Modèle Ŝ = f(x)")
n_arbres = st.slider("Nombre d'arbres du Random Forest", 20, 300, 150, 10)


@st.cache_resource(show_spinner="Apprentissage du modèle…")
def _regresseur(X, S, n_arbres):
    return entrainer_regresseur(X, S, n_arbres)


modele, r2 = _regresseur(M.valeurs, S, n_arbres)
st.metric("R² sur les alternatives de test", f"{r2:.3f}")

st.header("4. Explication SHAP")
c1, c2 = st.columns(2)
k = c1.slider("Nombre d'alternatives expliquées", 20, min(300, M.m), min(100, M.m), 10)
b = c2.slider("Taille de l'échantillon de fond", 10, min(100, M.m), min(40, M.m), 10)


@st.cache_data(show_spinner="Calcul exact des valeurs SHAP…")
def _shap(X, S, n_arbres, k, b):
    mod, _ = _regresseur(X, S, n_arbres)
    rng = np.random.default_rng(0)
    idx_k = rng.choice(len(X), size=k, replace=False)
    idx_b = rng.choice(len(X), size=b, replace=False)
    phi, phi0 = shap_exact(mod.predict, X[idx_k], X[idx_b])
    return idx_k, phi, phi0


idx_k, phi, phi0 = _shap(M.valeurs, S, n_arbres, k, b)
pred = modele.predict(M.valeurs[idx_k])
st.caption(f"φ₀ = E[f(x)] = {phi0:.4f} ; écart maximal |φ₀ + Σⱼ φⱼ − Ŝ| = {np.max(np.abs(phi0 + phi.sum(1) - pred)):.2e} "
           "(propriété d'efficacité).")
fig = go.Figure()
for j, c in enumerate(M.criteres):
    fig.add_trace(go.Box(y=phi[:, j], name=c, boxpoints="all", jitter=0.4, pointpos=0, marker=dict(size=4, color=gr.couleur(j)),
                         line=dict(color=gr.couleur(j)), hovertemplate=c + " : φ = %{y:.4f}<extra></extra>"))
fig.update_layout(title="Importance locale : φᵢⱼ de chaque alternative expliquée", showlegend=False,
                  paper_bgcolor=gr.SURFACE, plot_bgcolor=gr.SURFACE, height=420)
gr.afficher(fig)
choix = st.selectbox("Explication d'une alternative", range(len(idx_k)), format_func=lambda i: M.alternatives[idx_k[i]])
gr.afficher(gr.barres(M.criteres, phi[choix], f"Contributions SHAP pour {M.alternatives[idx_k[choix]]} "
                      f"(Ŝ = {pred[choix]:.4f}, φ₀ = {phi0:.4f})", "φⱼ"))

st.header("5. Mise à jour des poids")
w_shap = poids_shap(phi)
alpha = st.slider("α (part des poids initiaux)", 0.0, 1.0, 0.5, 0.05)
w_new = mise_a_jour_poids(w_init, w_shap, alpha)
st.dataframe(pd.DataFrame({"Importance globale |φⱼ|": np.round(importance_globale(phi), 4), "w initiaux": np.round(w_init, 4),
                           "|φⱼ| / Σ|φ|": np.round(w_shap, 4), f"w new (α = {alpha:.2f})": np.round(w_new, 4)},
                          index=M.criteres).T)
gr.afficher(gr.barres_groupees(M.criteres, {"w initiaux": w_init, "Poids SHAP": w_shap, "w new": w_new}, "Poids", "wⱼ"))

S_new = scores_wsm_reference(M.valeurs, M.sens, w_new)
rho = spearman(rangs_moyens(S), rangs_moyens(S_new))
c1, c2 = st.columns(2)
c1.metric("ρ de Spearman entre les deux classements", f"{rho:.4f}")
c2.metric("Meilleure alternative (w initiaux → w new)",
          f"{M.alternatives[int(np.argmax(S))]} → {M.alternatives[int(np.argmax(S_new))]}")
top = pd.DataFrame({"Rang avec w initiaux": rangs(S), "Rang avec w new": rangs(S_new)}, index=M.alternatives)
st.dataframe(top.sort_values("Rang avec w new").head(10))
R = np.where(M.sens > 0, M.valeurs / M.valeurs.max(0), M.valeurs.min(0) / M.valeurs)
dispersion = np.abs(R - R.mean(0)).mean(0)
st.info("Lecture : |φⱼ| mesure l'effet du critère j sur le score **dans ces données**. Il dépend du poids initial mais "
        "aussi de la dispersion du critère : un critère très pondéré mais peu dispersé reçoit un |φⱼ| faible. "
        "Écart moyen des rᵢⱼ normalisés : " + af.format_poids(M.criteres, dispersion, 3) + ".")
