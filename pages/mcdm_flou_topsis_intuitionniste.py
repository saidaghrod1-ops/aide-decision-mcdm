import numpy as np
import pandas as pd
import streamlit as st

from core.commun import ErreurMethode
from core.donnees import lire_sens
from core.flou.orthopair import frontiere
from core.mcdm_flou.topsis_intuitionniste import depuis_tableaux, topsis_intuitionniste
from ui import affichage as af
from ui import graphiques as gr
from ui.donnees import charger_tableau, editeur_tableau

af.entete("TOPSIS floue intuitionniste", "MCDM flou — décision de groupe", "143–150")

with st.expander("Méthode (cours p. 143–150)", expanded=False):
    st.markdown("Notation de la section : $n$ alternatives $A_i$, $m$ critères $C_j$, $l$ décideurs $D_k$ de poids "
                "$\\sigma_k$ ($\\sum_k \\sigma_k = 1$). Chaque décideur exprime $\\alpha_{ij}^k = (\\mu_{ij}^k, \\nu_{ij}^k)$.  \n"
                "**3) Agrégation des décideurs** (IFWA) et **4) poids des critères** :")
    st.latex(r"\alpha_{ij}=\Big(1-\prod_{k=1}^{l}(1-\mu_{ij}^k)^{\sigma_k},\ \prod_{k=1}^{l}(\nu_{ij}^k)^{\sigma_k}\Big),\qquad "
             r"w_j=\Big(1-\prod_{k}(1-\mu_j^k)^{\sigma_k},\ \prod_k(\nu_j^k)^{\sigma_k}\Big)")
    st.markdown("**5) Matrice pondérée** $a'_{ij} = a_{ij} \\otimes w_j$. **6) Solutions idéales** : pour $J_1$ (critères +), "
                "$\\mu$ max et $\\nu$ min pour $I^+$ ; pour $J_2$ (critères −), $\\mu$ min et $\\nu$ max ; $I^-$ à l'inverse.  \n"
                "**7) Séparations** et **8) coefficient de proximité** :")
    st.latex(r"S_i^{\pm}=\frac{1}{2m}\sum_{j=1}^{m}\sqrt{(\mu_{ij}'-\mu_j^{\pm})^2+(\nu_{ij}'-\nu_j^{\pm})^2+(\pi_{ij}'-\pi_j^{\pm})^2},"
             r"\qquad C_i=\frac{S_i^-}{S_i^-+S_i^+}")
    af.encadre_ecart("E-8, E-9, E-22 à E-24", "formule de séparation du cours appliquée telle quelle (racine par critère) ; "
                     "0 ≤ Cᵢ ≤ 1 (le cours écrit « 1 ≤ Cᵢ ≤ 1 ») ; notation n/m de la section conservée ; indices de "
                     "l'IFWA des poids corrigés (μⱼᵏ) ; ⊗ = produit de la p. 115 : (μₐμ_w, νₐ + ν_w − νₐν_w).")

ident = "D4_electrification_village"
st.caption("Jeu créé (aucun exemple dans le cours) : électrifier un village isolé (p. 4). Trois décideurs évaluent quatre "
           "options sur quatre critères ; échelles linguistiques construites pour l'exemple, modifiables.")
with st.expander("Échelles linguistiques → nombres intuitionnistes", expanded=False):
    c1, c2 = st.columns(2)
    with c1:
        ech_e = editeur_tableau("ift_ech_e", charger_tableau(ident, "echelle_evaluations"), titre="Évaluations",
                                nom_fichier="echelle_evaluations.csv")
    with c2:
        ech_w = editeur_tableau("ift_ech_w", charger_tableau(ident, "echelle_importance"), titre="Importance",
                                nom_fichier="echelle_importance.csv")
with st.expander("Évaluations, poids des critères, poids des décideurs et sens", expanded=False):
    evals = editeur_tableau("ift_evals", charger_tableau(ident, "evaluations"), titre="Évaluations des décideurs",
                            colonnes_attendues=["alternative", "decideur", "critere", "terme"], nom_fichier="evaluations.csv")
    poids = editeur_tableau("ift_poids", charger_tableau(ident, "poids_criteres"), titre="Importance des critères",
                            colonnes_attendues=["decideur", "critere", "terme"], nom_fichier="poids_criteres.csv")
    sigma_t = editeur_tableau("ift_sigma", charger_tableau(ident, "poids_decideurs"), titre="Poids σ des décideurs",
                              colonnes_attendues=["decideur", "sigma"], nom_fichier="poids_decideurs.csv")
    sens_t = editeur_tableau("ift_sens", charger_tableau(ident, "sens"), titre="Sens des critères",
                             colonnes_attendues=["critere", "sens"], nom_fichier="sens.csv", num_rows="fixed",
                             column_config={"sens": st.column_config.SelectboxColumn("sens", options=["+", "-"])})
try:
    echelle = lambda t: {str(r["code"]): (float(r["mu"]), float(r["nu"])) for _, r in t.iterrows()}
    sigma = {str(r["decideur"]): float(r["sigma"]) for _, r in sigma_t.iterrows()}
    alternatives, decideurs, criteres, mu_e, nu_e, mu_w, nu_w, sig = depuis_tableaux(
        echelle(ech_e), echelle(ech_w), evals.astype(str).values.tolist(), poids.astype(str).values.tolist(), sigma)
    sens_dict = {str(r["critere"]): lire_sens(r["sens"]) for _, r in sens_t.iterrows()}
    sens = np.array([sens_dict.get(c, 1) for c in criteres])
    r = topsis_intuitionniste(mu_e, nu_e, mu_w, nu_w, sig, sens)
except (ErreurMethode, ValueError, KeyError) as exc:
    st.error(f"Données invalides : {exc}")
    st.stop()

paires = lambda mu, nu: np.vectorize(lambda a, b: f"({a:.3f} ; {b:.3f})")(mu, nu)
with af.etape("3) Matrice agrégée D (IFWA des décideurs)", ouverte=True):
    st.dataframe(pd.DataFrame(paires(r.mu, r.nu), index=alternatives, columns=criteres))
with af.etape("4) Poids agrégés des critères", ouverte=True):
    st.dataframe(pd.DataFrame(paires(r.w_mu, r.w_nu)[None, :], index=["wⱼ = (μ ; ν)"], columns=criteres))
with af.etape("5) Matrice pondérée D' = D ⊗ W"):
    st.dataframe(pd.DataFrame(paires(r.mu_p, r.nu_p), index=alternatives, columns=criteres))
with af.etape("6) Solutions idéales I⁺ et I⁻ (μ ; ν ; π)", ouverte=True):
    fmt3 = lambda t: f"({t[0]:.3f} ; {t[1]:.3f} ; {t[2]:.3f})"
    st.dataframe(pd.DataFrame([[fmt3(t) for t in r.I_plus], [fmt3(t) for t in r.I_moins]], index=["I⁺", "I⁻"], columns=criteres))
with af.etape("7–8) Séparations et coefficients de proximité", ouverte=True):
    st.dataframe(pd.DataFrame({"S⁺": np.round(r.S_plus, 4), "S⁻": np.round(r.S_moins, 4), "C": np.round(r.C, 4)}, index=alternatives))
af.classement(alternatives, {"C": r.C}, r.rangs)
c1, c2 = st.columns(2)
with c1:
    gr.afficher(gr.barres(alternatives, r.C, "Coefficient de proximité Cᵢ", "C"))
with c2:
    j = st.selectbox("Critère affiché dans le plan (μ, ν)", range(len(criteres)), format_func=lambda k: criteres[k])
    gr.afficher(gr.plan_mu_nu({**{a: (r.mu_p[i, j], r.nu_p[i, j]) for i, a in enumerate(alternatives)},
                               "I⁺": (r.I_plus[j, 0], r.I_plus[j, 1]), "I⁻": (r.I_moins[j, 0], r.I_moins[j, 1])},
                              {"μ + ν = 1": frontiere(1)}, f"a'ᵢⱼ — {criteres[j]}"))
st.success(f"**{alternatives[int(np.argmax(r.C))]}** a le coefficient de proximité le plus élevé (C = {r.C.max():.4f}) : "
           "c'est l'option retenue par le groupe.")
