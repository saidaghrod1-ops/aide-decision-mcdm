import numpy as np
import pandas as pd
import streamlit as st

from core.commun import ErreurMethode
from core.donnees import dataset, lire_sens
from core.mcdm_flou.topsis_flou import depuis_tableaux, topsis_flou
from ui import affichage as af
from ui import graphiques as gr
from ui.donnees import charger_tableau, editeur_tableau

af.entete("TOPSIS floue (Chen, 2000)", "MCDM flou — décision de groupe", "135–142")

with st.expander("Méthode (cours p. 135–139)", expanded=False):
    st.markdown("Les variables linguistiques (nombres flous triangulaires) servent à évaluer les alternatives **et** "
                "l'importance des critères. 1) Chaque décideur évalue chaque alternative ; 2) chaque décideur évalue "
                "l'importance de chaque critère ; 3) évaluations collectives (K décideurs) :")
    st.latex(r"\tilde x_{ij}=\frac1K(\tilde x_{ij1}\oplus\cdots\oplus\tilde x_{ijK}),\qquad \tilde w_j=\frac1K(\tilde w_{j1}\oplus\cdots\oplus\tilde w_{jK})")
    st.markdown("4) Matrice normalisée :")
    st.latex(r"\tilde r_{ij}=\Big(\frac{a_{ij}}{c_j^*},\frac{b_{ij}}{c_j^*},\frac{c_{ij}}{c_j^*}\Big),\ c_j^*=\max_i c_{ij}\ (+)\qquad "
             r"\tilde r_{ij}=\Big(\frac{a_j^-}{c_{ij}},\frac{a_j^-}{b_{ij}},\frac{a_j^-}{a_{ij}}\Big),\ a_j^-=\min_i a_{ij}\ (-)")
    st.markdown("5) $\\tilde v_{ij} = \\tilde r_{ij} \\otimes \\tilde w_j$ ; 6) $v_j^* = (1, 1, 1)$, $v_j^- = (0, 0, 0)$ ; 7) distances :")
    st.latex(r"D_i^*=\sum_j d(\tilde v_{ij},v_j^*),\quad D_i^-=\sum_j d(\tilde v_{ij},v_j^-),\quad "
             r"d(\tilde v,v^*)=\sqrt{\tfrac13\big[(v^1-1)^2+(v^2-1)^2+(v^3-1)^2\big]}")
    st.markdown("8) $CC_i = D_i^- / (D_i^- + D_i^*)$ ; 9) classer par $CC_i$ décroissant.")
    af.encadre_ecart("E-4", "pour un critère négatif, le cours écrit (a⁻/a, a⁻/b, a⁻/c), qui donne un triplet décroissant "
                            "(nombre flou invalide). L'ordre de Chen (2000), (a⁻/c, a⁻/b, a⁻/a), est appliqué. "
                            "L'exercice, qui n'a que des critères positifs, n'est pas concerné.")

ident = "C10_chen2000_recrutement"
d = dataset(ident)
st.caption("Exercice (p. 140–142) : recruter un ingénieur d'analyse de systèmes parmi 3 candidats, évalués par 3 décideurs "
           "selon 5 critères (stabilité émotionnelle, communication orale, personnalité, expérience passée, confiance en soi).")
with st.expander("Tableau 1 — Variables linguistiques (modifiables)", expanded=False):
    c1, c2 = st.columns(2)
    with c1:
        ech_alt = editeur_tableau("chen_ech_alt", charger_tableau(ident, "echelle_alternatives"), titre="Évaluation des alternatives",
                                  nom_fichier="echelle_alternatives.csv", num_rows="dynamic")
    with c2:
        ech_imp = editeur_tableau("chen_ech_imp", charger_tableau(ident, "echelle_importance"), titre="Importance des critères",
                                  nom_fichier="echelle_importance.csv", num_rows="dynamic")
with st.expander("Tableaux 2 et 3 — Évaluations et poids donnés par les décideurs", expanded=False):
    evals = editeur_tableau("chen_evals", charger_tableau(ident, "evaluations"), titre="Tableau 2 — Évaluations",
                            colonnes_attendues=["alternative", "decideur", "critere", "terme"], nom_fichier="evaluations.csv")
    poids = editeur_tableau("chen_poids", charger_tableau(ident, "poids"), titre="Tableau 3 — Poids",
                            colonnes_attendues=["decideur", "critere", "terme"], nom_fichier="poids.csv")
try:
    echelle = lambda t: {str(r["code"]): (float(r["a"]), float(r["b"]), float(r["c"])) for _, r in t.iterrows()}
    alternatives, decideurs, criteres, E, P = depuis_tableaux(echelle(ech_alt), echelle(ech_imp),
                                                              evals.astype(str).values.tolist(), poids.astype(str).values.tolist())
except (ErreurMethode, ValueError, KeyError) as exc:
    st.error(f"Données invalides : {exc}")
    st.stop()
sens_defaut = pd.DataFrame({"critere": criteres, "sens": [d.get("sens", {}).get(c, "+") for c in criteres]})
sens_table = st.data_editor(sens_defaut, hide_index=True, disabled=["critere"], key="chen_sens",
                            column_config={"sens": st.column_config.SelectboxColumn("sens", options=["+", "-"])})
try:
    sens = np.array([lire_sens(s) for s in sens_table["sens"]])
    r = topsis_flou(E, P, sens)
except ErreurMethode as exc:
    st.error(str(exc))
    st.stop()

with af.etape("Question 1 — Matrice de décision collective", ouverte=True):
    af.matrice_triplets(r.X, alternatives, criteres, 3)
with af.etape("Question 2 — Matrice collective normalisée"):
    af.matrice_triplets(r.R, alternatives, criteres, 3)
with af.etape("Question 3 — Poids collectifs des critères", ouverte=True):
    st.dataframe(pd.DataFrame({"w̃ⱼ": af.triplets(r.W, 3)}, index=criteres).T)
with af.etape("Question 4 — Matrice normalisée pondérée"):
    af.matrice_triplets(r.V, alternatives, criteres, 3)
with af.etape("Question 5 — Distances aux solutions idéales et coefficient de proximité", ouverte=True):
    st.markdown("**d(ṽᵢⱼ, v\\*)**")
    af.matrice(r.d_plus, alternatives, criteres)
    st.markdown("**d(ṽᵢⱼ, v⁻)**")
    af.matrice(r.d_moins, alternatives, criteres)
af.classement(alternatives, {"D*": r.D_plus, "D⁻": r.D_moins, "CC": r.CC}, r.rangs)
gr.afficher(gr.barres(alternatives, r.CC, "Coefficient de proximité CCᵢ", "CC"))
st.success(f"**{alternatives[int(np.argmax(r.CC))]}** a le coefficient de proximité le plus élevé "
           f"(CC = {r.CC.max():.4f}) : c'est le meilleur candidat.")
crit_vu = st.selectbox("Nombres flous pondérés ṽᵢⱼ du critère", range(len(criteres)), format_func=lambda j: criteres[j])
gr.afficher(gr.nombres_flous({a: r.V[i, crit_vu] for i, a in enumerate(alternatives)}, f"ṽᵢⱼ — {criteres[crit_vu]}", "valeur"))
