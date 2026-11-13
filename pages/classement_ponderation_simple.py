import numpy as np
import pandas as pd
import streamlit as st

from core.classement.ponderation_simple import courbe_lambda, waspas
from core.commun import ErreurMethode
from core.robustesse.sensibilite import balayage_poids
from ui import affichage as af
from ui import graphiques as gr
from ui.donnees import charger_matrice, choisir_dataset, editeur_matrice, options_matrices, poids_valides

af.entete("WSM, WPM et WASPAS", "Classement compensatoire — méthodes de pondération simple", "53–59")

with st.expander("Théorie (cours p. 53–59)", expanded=False):
    st.markdown("Soit une matrice de décision $(x_{ij})$ et des poids $w_j$ avec $\\sum_j w_j = 1$.  \n"
                "**1 —** Solutions idéales $A^+ = \\{x_1^+, \\dots, x_n^+\\}$ et $A^- = \\{x_1^-, \\dots, x_n^-\\}$ : "
                "meilleure et plus mauvaise valeur de chaque critère selon son sens.  \n"
                "**2 —** Normaliser la matrice de décision (p. 55) :")
    st.latex(r"r_{ij}=\frac{x_{ij}}{x_j^+}=\frac{x_{ij}}{\max x_{ij}}\ \text{(critère +)},\qquad "
             r"r_{ij}=\frac{x_j^+}{x_{ij}}=\frac{\min x_{ij}}{x_{ij}}\ \text{(critère −)}")
    st.markdown("**3.1 — WSM** et **3.2 — WPM** : l'alternative de plus grande valeur est la meilleure.")
    st.latex(r"Q_i^{(1)}=\sum_{j=1}^{n} w_j\, r_{ij},\qquad Q_i^{(2)}=\prod_{j=1}^{n} (r_{ij})^{w_j}")
    st.markdown("**3.3 — WASPAS** (Zavadskas et al., 2012) : critère généralisé conjoint, classé par ordre décroissant.")
    st.latex(r"Q_i=\tfrac12\left(Q_i^{(1)}+Q_i^{(2)}\right),\qquad Q_i=\lambda\,Q_i^{(1)}+(1-\lambda)\,Q_i^{(2)},\ \lambda\in[0,1]")

ident = choisir_dataset("wsm", options_matrices("C4_waspas_machines"))
M = editeur_matrice(f"wsm_{ident}", charger_matrice(ident), nom_fichier="ponderation_simple.csv")
if M is None:
    st.stop()
w = poids_valides(M, f"wsm_{ident}")
if w is None:
    st.stop()
lam = st.slider("λ (WASPAS)", 0.0, 1.0, 0.5, 0.05, help="λ = 1 : WSM ; λ = 0 : WPM ; λ = 0,5 : forme de base.")
try:
    r = waspas(M.valeurs, M.sens, w, lam)
except ErreurMethode as exc:
    st.error(f"La méthode ne s'applique pas : {exc}.")
    st.stop()

alt, crit = M.alternatives, M.criteres
with af.etape("Étape 1 — Solutions idéales positive et négative"):
    st.dataframe(pd.DataFrame({"A⁺ (meilleure valeur)": r.ideal_positif, "A⁻ (plus mauvaise valeur)": r.ideal_negatif,
                               "Sens": ["+" if s > 0 else "−" for s in M.sens]}, index=crit).T)
with af.etape("Étape 2 — Matrice normalisée rᵢⱼ", ouverte=True):
    af.matrice(r.R, alt, crit)
with af.etape("Étape 3.1 — WSM : termes wⱼ rᵢⱼ et Qᵢ⁽¹⁾"):
    af.matrice(np.column_stack([r.termes_somme, r.Q1]), alt, list(crit) + ["Q⁽¹⁾"])
with af.etape("Étape 3.2 — WPM : facteurs rᵢⱼ^wⱼ et Qᵢ⁽²⁾"):
    af.matrice(np.column_stack([r.facteurs_produit, r.Q2]), alt, list(crit) + ["Q⁽²⁾"])
with af.etape(f"Étape 3.3 — WASPAS : Qᵢ = {lam:.2f} Q⁽¹⁾ + {1 - lam:.2f} Q⁽²⁾", ouverte=True):
    st.dataframe(pd.DataFrame({"Q⁽¹⁾ (WSM)": np.round(r.Q1, 4), "Rang WSM": r.rangs_Q1, "Q⁽²⁾ (WPM)": np.round(r.Q2, 4),
                               "Rang WPM": r.rangs_Q2, "Q (WASPAS)": np.round(r.Q, 4), "Rang WASPAS": r.rangs_Q},
                              index=alt))

af.classement(alt, {"Q (WASPAS)": r.Q}, r.rangs_Q)
gr.afficher(gr.barres_groupees(alt, {"Q⁽¹⁾ WSM": r.Q1, "Q⁽²⁾ WPM": r.Q2, f"Q WASPAS (λ = {lam:.2f})": r.Q},
                               "Scores des trois méthodes"))
meilleure = alt[int(np.argmax(r.Q))]
accord = len({int(np.argmax(r.Q1)), int(np.argmax(r.Q2)), int(np.argmax(r.Q))}) == 1
st.success(f"**{meilleure}** a le Qᵢ le plus élevé : c'est la meilleure alternative selon WASPAS (λ = {lam:.2f})."
           + (" WSM et WPM désignent la même alternative." if accord else
              " WSM et WPM ne désignent pas la même alternative : le choix dépend de λ."))

st.header("Sensibilité")
lams, Q = courbe_lambda(M.valeurs, M.sens, w)
gr.afficher(gr.courbes(lams, {a: Q[:, i] for i, a in enumerate(alt)}, "Qᵢ selon λ (λ = 0 : WPM, λ = 1 : WSM)",
                       "λ", "Qᵢ", marqueur_x=lam))
k = st.selectbox("Faire varier le poids du critère", range(len(crit)), format_func=lambda j: crit[j])
ts, S, _ = balayage_poids(w, k, lambda v: waspas(M.valeurs, M.sens, v, lam).Q)
gr.afficher(gr.courbes(ts, {a: S[:, i] for i, a in enumerate(alt)}, f"Q (WASPAS) selon le poids de « {crit[k]} »",
                       f"poids de {crit[k]}", "Qᵢ", marqueur_x=float(w[k])))
