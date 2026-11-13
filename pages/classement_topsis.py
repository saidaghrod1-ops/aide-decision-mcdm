import numpy as np
import pandas as pd
import streamlit as st

from core.classement.topsis import topsis
from core.commun import ErreurMethode
from core.robustesse.sensibilite import balayage_poids, changements_de_premier
from ui import affichage as af
from ui import graphiques as gr
from ui.donnees import charger_matrice, choisir_dataset, editeur_matrice, options_matrices, poids_valides

af.entete("TOPSIS", "Classement compensatoire", "60–65",
          "TOPSIS (Hwang et Yoon, 1981) choisit l'alternative la plus proche de la solution idéale positive "
          "et la plus éloignée de la solution idéale négative.")

with st.expander("Théorie (cours p. 60–64)", expanded=False):
    st.markdown("**Étape 1** — Normaliser la matrice de décision :")
    st.latex(r"r_{ij}=\frac{x_{ij}}{\sqrt{\sum_i x_{ij}^2}}")
    st.markdown("**Étape 2** — Matrice normalisée et pondérée : $v_{ij} = w_j \\times r_{ij}$.  \n"
                "**Étape 3** — Solutions idéales, avec $J$ et $J'$ les ensembles des critères positifs et négatifs :")
    st.latex(r"v_j^*=\begin{cases}\max_i v_{ij} & j\in J\\ \min_i v_{ij} & j\in J'\end{cases},\qquad "
             r"v_j'=\begin{cases}\min_i v_{ij} & j\in J\\ \max_i v_{ij} & j\in J'\end{cases}")
    st.markdown("**Étape 4** — Mesures de séparation :")
    st.latex(r"S_i^+=\sqrt{\sum_j (v_j^*-v_{ij})^2},\qquad S_i^-=\sqrt{\sum_j (v_j'-v_{ij})^2}")
    st.markdown("**Étape 5** — Coefficient de proximité ; la meilleure alternative a le $RC_i^*$ le plus élevé :")
    st.latex(r"RC_i^*=\frac{S_i^-}{S_i^-+S_i^+},\qquad 0\le RC_i^*\le 1")

ident = choisir_dataset("topsis", options_matrices("C5_topsis_voitures"))
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

alt, crit = M.alternatives, M.criteres
with af.etape("Étape 1 — Matrice normalisée rᵢⱼ"):
    af.matrice(r.R, alt, crit)
with af.etape("Étape 2 — Matrice normalisée pondérée vᵢⱼ = wⱼ rᵢⱼ"):
    af.matrice(r.V, alt, crit)
with af.etape("Étape 3 — Solutions idéales I⁺ et I⁻", ouverte=True):
    st.dataframe(pd.DataFrame(np.round([r.I_plus, r.I_moins], 4), index=["I⁺ (v*)", "I⁻ (v')"], columns=crit))
with af.etape("Étapes 4 et 5 — Séparations et coefficient de proximité", ouverte=True):
    st.dataframe(pd.DataFrame({"S⁺": np.round(r.S_plus, 4), "S⁻": np.round(r.S_moins, 4), "RC*": np.round(r.RC, 4),
                               "Rang": r.rangs}, index=alt))

af.classement(alt, {"RC*": r.RC}, r.rangs)
c1, c2 = st.columns(2)
with c1:
    gr.afficher(gr.barres(alt, r.RC, "Coefficient de proximité RCᵢ*", "RC*"))
with c2:
    gr.afficher(gr.nuage(r.S_plus, r.S_moins, alt, "Séparations : idéal en haut à gauche", "S⁺ (distance à I⁺)",
                         "S⁻ (distance à I⁻)", couleurs=r.RC, echelle_couleur="RC*"))
meilleure = alt[int(np.argmax(r.RC))]
st.success(f"**{meilleure}** a le RC* le plus élevé ({r.RC.max():.4f}) : c'est l'alternative la plus proche de la "
           "solution idéale positive et la plus éloignée de la solution idéale négative.")

st.header("Sensibilité aux poids")
k = st.selectbox("Faire varier le poids du critère", range(len(crit)), format_func=lambda j: crit[j])
ts, S, Rg = balayage_poids(w, k, lambda v: topsis(M.valeurs, M.sens, v).RC)
gr.afficher(gr.courbes(ts, {a: S[:, i] for i, a in enumerate(alt)}, f"RC* selon le poids de « {crit[k]} »",
                       f"poids de {crit[k]}", "RC*", marqueur_x=float(w[k])))
changements = changements_de_premier(ts, Rg)
st.info("La meilleure alternative change : " + " ; ".join(f"vers w = {t:.2f}, {alt[a]} → {alt[b]}" for t, a, b in changements)
        if changements else "La meilleure alternative reste la même quel que soit ce poids.")
