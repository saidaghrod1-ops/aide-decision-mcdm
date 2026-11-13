import numpy as np
import pandas as pd
import streamlit as st

from core.commun import ErreurMethode
from core.mcdm_flou.topsis_flou import depuis_tableaux, topsis_flou
from ui import affichage as af
from ui import graphiques as gr
from ui.donnees import charger_tableau, editeur_tableau

af.entete("TOPSIS floue (Chen, 2000)", "MCDM flou — décision de groupe")

with st.expander("Méthode", expanded=True):
    st.markdown("Les décideurs évaluent les alternatives et l'importance des critères avec des variables linguistiques "
                "(nombres flous triangulaires).")
    st.latex(r"\tilde x_{ij}=\frac1K(\tilde x_{ij1}\oplus\cdots\oplus\tilde x_{ijK}),\quad \tilde w_j=\frac1K(\tilde w_{j1}\oplus\cdots\oplus\tilde w_{jK}),\quad "
             r"\tilde r_{ij}=\Big(\frac{a_{ij}}{c_j^*},\frac{b_{ij}}{c_j^*},\frac{c_{ij}}{c_j^*}\Big),\quad \tilde v_{ij}=\tilde r_{ij}\otimes\tilde w_j")
    st.latex(r"v^*=(1,1,1),\ v^-=(0,0,0),\quad d(\tilde v,v)=\sqrt{\tfrac13\sum(\cdot)^2},\quad "
             r"D_i^{*}=\sum_j d(\tilde v_{ij},v^*),\ D_i^-=\sum_j d(\tilde v_{ij},v^-),\quad CC_i=\frac{D_i^-}{D_i^-+D_i^*}")

ident = "C10_chen2000_recrutement"
st.caption("Exercice du cours : recruter un ingénieur parmi 3 candidats, évalués par 3 décideurs sur 5 critères.")
with st.expander("Échelles linguistiques, évaluations et poids des décideurs (modifiables)"):
    ech_alt = editeur_tableau("chen_ech_alt", charger_tableau(ident, "echelle_alternatives"), nom_fichier="echelle_alternatives.csv")
    ech_imp = editeur_tableau("chen_ech_imp", charger_tableau(ident, "echelle_importance"), nom_fichier="echelle_importance.csv")
    evals = editeur_tableau("chen_evals", charger_tableau(ident, "evaluations"),
                            colonnes_attendues=["alternative", "decideur", "critere", "terme"], nom_fichier="evaluations.csv")
    poids = editeur_tableau("chen_poids", charger_tableau(ident, "poids"),
                            colonnes_attendues=["decideur", "critere", "terme"], nom_fichier="poids.csv")
try:
    echelle = lambda t: {str(r["code"]): (float(r["a"]), float(r["b"]), float(r["c"])) for _, r in t.iterrows()}
    alternatives, _, criteres, E, P = depuis_tableaux(echelle(ech_alt), echelle(ech_imp), evals.astype(str).values.tolist(),
                                                      poids.astype(str).values.tolist())
    r = topsis_flou(E, P, np.ones(len(criteres), int))
except (ErreurMethode, ValueError, KeyError) as exc:
    st.error(f"Données invalides : {exc}")
    st.stop()

with af.etape("1. Matrice de décision collective", ouverte=True):
    af.matrice_triplets(r.X, alternatives, criteres, 2)
with af.etape("2. Matrice collective normalisée"):
    af.matrice_triplets(r.R, alternatives, criteres, 3)
with af.etape("3. Poids collectifs des critères", ouverte=True):
    st.dataframe(pd.DataFrame({"w̃ⱼ": af.triplets(r.W, 3)}, index=criteres).T)
with af.etape("4. Matrice normalisée pondérée"):
    af.matrice_triplets(r.V, alternatives, criteres, 3)
st.subheader("5. Résolution")
af.classement(alternatives, {"D*": r.D_plus, "D⁻": r.D_moins, "CC": r.CC}, r.rangs)
gr.afficher(gr.barres(alternatives, r.CC, "Coefficient de proximité CC", "CC"))
st.success(f"Meilleur candidat : **{alternatives[int(np.argmax(r.CC))]}** (CC = {r.CC.max():.4f}).")
