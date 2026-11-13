import numpy as np
import pandas as pd
import streamlit as st

from core.commun import ErreurMethode
from core.donnees import lire_sens
from core.mcdm_flou.topsis_intuitionniste import depuis_tableaux, topsis_intuitionniste
from ui import affichage as af
from ui import graphiques as gr
from ui.donnees import charger_tableau, editeur_tableau

af.entete("TOPSIS floue intuitionniste", "MCDM flou — décision de groupe")

with st.expander("Méthode", expanded=True):
    st.markdown("Chaque décideur $D_k$ (poids $\\sigma_k$) donne des nombres intuitionnistes $(\\mu, \\nu)$ ; "
                "ils sont agrégés par l'opérateur IFWA, puis pondérés par les poids des critères.")
    st.latex(r"a_{ij}=\Big(1-\prod_k(1-\mu_{ij}^k)^{\sigma_k},\ \prod_k(\nu_{ij}^k)^{\sigma_k}\Big),\qquad a'_{ij}=a_{ij}\otimes w_j")
    st.markdown("$I^+$ : $\\mu$ max et $\\nu$ min pour un critère + (l'inverse pour un critère −) ; $I^-$ à l'opposé.")
    st.latex(r"S_i^{\pm}=\frac{1}{2m}\sum_{j}\sqrt{(\mu'_{ij}-\mu_j^{\pm})^2+(\nu'_{ij}-\nu_j^{\pm})^2+(\pi'_{ij}-\pi_j^{\pm})^2},"
             r"\qquad C_i=\frac{S_i^-}{S_i^-+S_i^+}")

ident = "D4_electrification_village"
st.caption("Jeu créé : électrifier un village isolé. Trois décideurs (ingénieur, économiste, élu local) évaluent quatre "
           "options sur quatre critères.")
with st.expander("Échelles, évaluations et poids (modifiables)"):
    ech_e = editeur_tableau("ift_ech_e", charger_tableau(ident, "echelle_evaluations"), nom_fichier="echelle_evaluations.csv")
    ech_w = editeur_tableau("ift_ech_w", charger_tableau(ident, "echelle_importance"), nom_fichier="echelle_importance.csv")
    evals = editeur_tableau("ift_evals", charger_tableau(ident, "evaluations"),
                            colonnes_attendues=["alternative", "decideur", "critere", "terme"], nom_fichier="evaluations.csv")
    poids = editeur_tableau("ift_poids", charger_tableau(ident, "poids_criteres"),
                            colonnes_attendues=["decideur", "critere", "terme"], nom_fichier="poids_criteres.csv")
    sigma_t = editeur_tableau("ift_sigma", charger_tableau(ident, "poids_decideurs"), nom_fichier="poids_decideurs.csv")
    sens_t = editeur_tableau("ift_sens", charger_tableau(ident, "sens"), nom_fichier="sens.csv", num_rows="fixed",
                             column_config={"sens": st.column_config.SelectboxColumn("sens", options=["+", "-"])})
try:
    echelle = lambda t: {str(r["code"]): (float(r["mu"]), float(r["nu"])) for _, r in t.iterrows()}
    sigma = {str(r["decideur"]): float(r["sigma"]) for _, r in sigma_t.iterrows()}
    alternatives, _, criteres, mu_e, nu_e, mu_w, nu_w, sig = depuis_tableaux(
        echelle(ech_e), echelle(ech_w), evals.astype(str).values.tolist(), poids.astype(str).values.tolist(), sigma)
    sens_dict = {str(r["critere"]): lire_sens(r["sens"]) for _, r in sens_t.iterrows()}
    r = topsis_intuitionniste(mu_e, nu_e, mu_w, nu_w, sig, np.array([sens_dict.get(c, 1) for c in criteres]))
except (ErreurMethode, ValueError, KeyError) as exc:
    st.error(f"Données invalides : {exc}")
    st.stop()

paires = lambda mu, nu: np.vectorize(lambda a, b: f"({a:.3f} ; {b:.3f})")(mu, nu)
with af.etape("Matrice agrégée (IFWA) et poids des critères", ouverte=True):
    st.dataframe(pd.DataFrame(paires(r.mu, r.nu), index=alternatives, columns=criteres))
    st.dataframe(pd.DataFrame(paires(r.w_mu, r.w_nu)[None, :], index=["wⱼ"], columns=criteres))
with af.etape("Matrice pondérée et solutions idéales"):
    st.dataframe(pd.DataFrame(paires(r.mu_p, r.nu_p), index=alternatives, columns=criteres))
    fmt3 = lambda t: f"({t[0]:.3f} ; {t[1]:.3f} ; {t[2]:.3f})"
    st.dataframe(pd.DataFrame([[fmt3(t) for t in r.I_plus], [fmt3(t) for t in r.I_moins]], index=["I⁺", "I⁻"], columns=criteres))
af.classement(alternatives, {"S⁺": r.S_plus, "S⁻": r.S_moins, "C": r.C}, r.rangs)
gr.afficher(gr.barres(alternatives, r.C, "Coefficient de proximité C", "C"))
st.success(f"Option retenue par le groupe : **{alternatives[int(np.argmax(r.C))]}** (C = {r.C.max():.4f}).")
