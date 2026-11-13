import numpy as np
import pandas as pd
import streamlit as st

from core.classement.vikor import vikor
from core.commun import ErreurMethode
from ui import affichage as af
from ui import graphiques as gr
from ui.donnees import charger_matrice, choisir_dataset, editeur_matrice, options_matrices, poids_valides

af.entete("VIKOR", "Classement de compromis")

with st.expander("Théorie", expanded=True):
    st.markdown("$x_j^*$ et $x_j^-$ : meilleure et plus mauvaise valeur du critère j.")
    st.latex(r"S_i=\sum_j w_j\frac{x_j^*-x_{ij}}{x_j^*-x_j^-},\qquad R_i=\max_j\Big[w_j\frac{x_j^*-x_{ij}}{x_j^*-x_j^-}\Big],\qquad "
             r"Q_i=v\frac{S_i-S^*}{S^--S^*}+(1-v)\frac{R_i-R^*}{R^--R^*}")
    st.markdown("Plus S, R et Q sont **petits**, meilleure est l'alternative ($v$ = 0,5 en général). $A'$ (plus petit Q) "
                "est la solution de compromis si :  \n"
                "• **avantage acceptable** : $Q(A'') - Q(A') \\ge \\frac{1}{m-1}$ ;  \n"
                "• **stabilité acceptable** : $A'$ est aussi la meilleure en S et en R.  \n"
                "Sinon, VIKOR propose un ensemble de solutions de compromis.")

ident = choisir_dataset("vikor", options_matrices("C6_vikor_machines", "D1_selection_modele_ml", "D2_fournisseurs_5offres"))
M = editeur_matrice(f"vikor_{ident}", charger_matrice(ident), nom_fichier="vikor.csv")
if M is None:
    st.stop()
w = poids_valides(M, f"vikor_{ident}")
if w is None:
    st.stop()
v = st.slider("v", 0.0, 1.0, 0.5, 0.05)
try:
    r = vikor(M.valeurs, M.sens, w, v)
except ErreurMethode as exc:
    st.error(str(exc))
    st.stop()

with af.etape("Meilleures et plus mauvaises valeurs, termes de Sᵢ et Rᵢ"):
    st.dataframe(pd.DataFrame([r.meilleures, r.pires], index=["x*", "x⁻"], columns=M.criteres))
    af.matrice(r.termes, M.alternatives, M.criteres)
st.dataframe(pd.DataFrame({"S": np.round(r.S, 4), "R": np.round(r.R, 4), "Q": np.round(r.Q, 4),
                           "Rang S": r.rangs_S, "Rang R": r.rangs_R, "Rang Q": r.rangs_Q}, index=M.alternatives).sort_values("Q"))
a1, a2 = int(r.ordre_Q[0]), int(r.ordre_Q[1])
st.markdown(f"- Avantage acceptable : Q(A'') − Q(A') = {r.avantage:.4f} {'≥' if r.condition_avantage else '<'} "
            f"1/(m − 1) = {r.DQ:.4f} → **{'vérifié' if r.condition_avantage else 'non vérifié'}**\n"
            f"- Stabilité acceptable : {M.alternatives[a1]} est {'aussi' if r.condition_stabilite else 'pas'} la meilleure "
            f"en S et en R → **{'vérifiée' if r.condition_stabilite else 'non vérifiée'}**")
gr.afficher(gr.barres(M.alternatives, r.Q, "Indice Q (plus petit = meilleur)", "Q"))
noms = ", ".join(M.alternatives[k] for k in r.compromis)
(st.success if len(r.compromis) == 1 else st.warning)(
    f"Solution{'s' if len(r.compromis) > 1 else ''} de compromis : **{noms}**. {r.explication}")
