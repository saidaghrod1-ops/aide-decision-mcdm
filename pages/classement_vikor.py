import numpy as np
import pandas as pd
import streamlit as st

from core.classement.vikor import vikor
from core.commun import ErreurMethode
from core.robustesse.sensibilite import balayage_poids
from ui import affichage as af
from ui import graphiques as gr
from ui.donnees import charger_matrice, choisir_dataset, editeur_matrice, options_matrices, poids_valides

af.entete("VIKOR", "Classement compensatoire — méthode de compromis", "66–72")

with st.expander("Théorie (cours p. 66–71)", expanded=False):
    st.markdown("**Étape 1** — Meilleures et plus mauvaises valeurs de chaque critère : $x_j^* = \\max_i x_{ij}$ et "
                "$x_j^- = \\min_i x_{ij}$ pour un critère positif (l'inverse pour un critère négatif).  \n"
                "**Étape 2** — Mesure d'utilité de groupe $S_i$ et de regret individuel $R_i$ :")
    st.latex(r"S_i=\sum_{j=1}^{n} w_j\frac{x_j^*-x_{ij}}{x_j^*-x_j^-},\qquad R_i=\max_j\left[w_j\frac{x_j^*-x_{ij}}{x_j^*-x_j^-}\right]")
    st.markdown("**Étape 3** — Indice de compromis ($v$ : poids de la stratégie d'utilité de groupe, généralement 0,5) :")
    st.latex(r"Q_i=v\frac{S_i-S^*}{S^--S^*}+(1-v)\frac{R_i-R^*}{R^--R^*}")
    st.markdown("**Étape 4** — Trois classements selon S, R et Q : plus les valeurs sont **petites**, meilleure est "
                "l'alternative.  \n**Étape 5** — $A'$ (plus petit Q) est la solution de compromis si :  \n"
                "• *avantage acceptable* : $Q(A'') - Q(A') \\ge \\frac{1}{m-1}$ ($A''$ : deuxième selon Q) ;  \n"
                "• *stabilité acceptable* : $A'$ est aussi la meilleure en termes de S et de R.  \n"
                "Sinon : $A'$ et $A''$ si seule la stabilité échoue ; $A', A'', \\dots, A^k$ avec "
                "$Q(A^k) - Q(A') < \\frac{1}{m-1}$ si l'avantage échoue (p. 71).")
    af.encadre_ecart("E-5", "le cours écrit « classer par ordre décroissant » de S, R et Q alors que les plus petites "
                            "valeurs sont les meilleures : les classements sont faits par ordre croissant.")
    af.encadre_ecart("E-6", "dans la définition de x_j*, le max porte sur les alternatives i = 1, …, m (et non sur j).")

ident = choisir_dataset("vikor", options_matrices("C6_vikor_machines"))
M = editeur_matrice(f"vikor_{ident}", charger_matrice(ident), nom_fichier="vikor.csv")
if M is None:
    st.stop()
w = poids_valides(M, f"vikor_{ident}")
if w is None:
    st.stop()
v = st.slider("v (poids de la stratégie d'utilité de groupe)", 0.0, 1.0, 0.5, 0.05)
try:
    r = vikor(M.valeurs, M.sens, w, v)
except ErreurMethode as exc:
    st.error(str(exc))
    st.stop()

alt, crit = M.alternatives, M.criteres
with af.etape("Étape 1 — Meilleures et plus mauvaises valeurs"):
    st.dataframe(pd.DataFrame([r.meilleures, r.pires], index=["x* (meilleure)", "x⁻ (plus mauvaise)"], columns=crit))
with af.etape("Étape 2 — Termes wⱼ (xⱼ* − xᵢⱼ)/(xⱼ* − xⱼ⁻), Sᵢ et Rᵢ", ouverte=True):
    af.matrice(np.column_stack([r.termes, r.S, r.R]), alt, list(crit) + ["S", "R"])
with af.etape("Étape 3 — Indice Qᵢ", ouverte=True):
    st.markdown(f"S\\* = {r.S_etoile:.4f}, S⁻ = {r.S_moins:.4f}, R\\* = {r.R_etoile:.4f}, R⁻ = {r.R_moins:.4f}, v = {v:.2f}")
    af.vecteur(r.Q, alt, "Qᵢ")
with af.etape("Étape 4 — Trois classements (ordre croissant)", ouverte=True):
    st.dataframe(pd.DataFrame({"S": np.round(r.S, 4), "Rang S": r.rangs_S, "R": np.round(r.R, 4), "Rang R": r.rangs_R,
                               "Q": np.round(r.Q, 4), "Rang Q": r.rangs_Q}, index=alt))

st.header("Étape 5 — Solution de compromis")
a1, a2 = int(r.ordre_Q[0]), int(r.ordre_Q[1])
c1, c2 = st.columns(2)
c1.metric("Avantage Q(A'') − Q(A')", f"{r.avantage:.4f}", f"seuil 1/(m − 1) = {r.DQ:.4f}", delta_color="off")
c2.metric("A' meilleure aussi en S et en R ?", "oui" if r.condition_stabilite else "non")
st.markdown(f"- **Avantage acceptable** : {'vérifié' if r.condition_avantage else 'non vérifié'} "
            f"(A' = {alt[a1]}, A'' = {alt[a2]}).\n"
            f"- **Stabilité acceptable** : {'vérifiée' if r.condition_stabilite else 'non vérifiée'}.")
noms = ", ".join(alt[k] for k in r.compromis)
(st.success if len(r.compromis) == 1 else st.warning)(
    f"Solution{'s' if len(r.compromis) > 1 else ''} de compromis : **{noms}**. {r.explication}")

c1, c2 = st.columns(2)
with c1:
    gr.afficher(gr.barres(alt, r.Q, "Indice Qᵢ (plus petit = meilleur)", "Qᵢ",
                          ligne=(float(r.Q[a1] + r.DQ), "Q(A') + 1/(m − 1)")))
with c2:
    gr.afficher(gr.nuage(r.S, r.R, alt, "Utilité de groupe S et regret individuel R", "S (plus petit = meilleur)",
                         "R (plus petit = meilleur)", couleurs=r.Q, echelle_couleur="Q"))
st.caption("La solution proposée par VIKOR maximise la satisfaction de groupe (S) et minimise le regret individuel (R) : "
           "c'est une base de négociation entre décideurs (p. 71).")

st.header("Sensibilité")
vs = np.linspace(0, 1, 101)
Qs = np.array([vikor(M.valeurs, M.sens, w, x).Q for x in vs])
gr.afficher(gr.courbes(vs, {a: Qs[:, i] for i, a in enumerate(alt)}, "Qᵢ selon v", "v", "Qᵢ", marqueur_x=v))
compromis_v = {x: vikor(M.valeurs, M.sens, w, x).compromis for x in (0.0, 0.25, 0.5, 0.75, 1.0)}
st.dataframe(pd.DataFrame({"v": list(compromis_v), "Compromis": [", ".join(alt[k] for k in c) for c in compromis_v.values()]}),
             hide_index=True)
k = st.selectbox("Faire varier le poids du critère", range(len(crit)), format_func=lambda j: crit[j])
ts, S, _ = balayage_poids(w, k, lambda x: vikor(M.valeurs, M.sens, x, v).Q, plus_grand_meilleur=False)
gr.afficher(gr.courbes(ts, {a: S[:, i] for i, a in enumerate(alt)}, f"Qᵢ selon le poids de « {crit[k]} »",
                       f"poids de {crit[k]}", "Qᵢ", marqueur_x=float(w[k])))
