import numpy as np
import pandas as pd
import streamlit as st

from core.commun import ErreurMethode
from core.ponderation.critic import poids_critic
from ui import affichage as af
from ui import graphiques as gr
from ui.donnees import charger_matrice, choisir_dataset, editeur_matrice
from ui.etat import publier_poids

af.entete("Méthode CRITIC", "Pondération objective", "46–51")

with st.expander("Théorie (cours p. 46–50)", expanded=False):
    st.markdown("1) Normaliser la matrice de décision selon le sens des critères (p. 47) :")
    st.latex(r"r_{ij}=\frac{x_{ij}-\min(x_j)}{\max(x_j)-\min(x_j)}\ \text{(critère +)},\qquad "
             r"r_{ij}=\frac{\max(x_j)-x_{ij}}{\max(x_j)-\min(x_j)}\ \text{(critère −)}")
    st.markdown("2) Coefficient de corrélation entre les attributs (p. 48) :")
    st.latex(r"\rho_{jk}=\frac{\sum_i (r_{ij}-\bar r_j)(r_{ik}-\bar r_k)}{\sqrt{\sum_i (r_{ij}-\bar r_j)^2\sum_i (r_{ik}-\bar r_k)^2}},"
             r"\qquad \bar r_j=\frac1m\sum_i r_{ij}")
    st.markdown("3) Écart type et indice C (p. 49) :")
    st.latex(r"\sigma_j=\sqrt{\frac{1}{m-1}\sum_i (r_{ij}-\bar r_j)^2},\qquad C_j=\sigma_j\sum_{k=1}^{n}(1-\lvert\rho_{jk}\rvert)")
    st.markdown("4) Poids : $w_j = C_j / \\sum_j C_j$. Plus un critère est dispersé et moins il est corrélé aux autres, "
                "plus il est important : $\\sum_k (1-\\lvert\\rho_{jk}\\rvert)$ mesure son indépendance.")

ident = choisir_dataset("critic", {
    "Exercice : machines de découpage (p. 51)": "C3_critic_machines_decoupe",
    "Sélection d'un modèle de ML (jeu créé, p. 14)": "D1_selection_modele_ml",
    "Choisir un fournisseur parmi 5 offres (jeu créé)": "D2_fournisseurs_5offres",
    "Portefeuille Vuković et al. (p. 18)": "C12_portefeuille_vukovic2020",
})
if ident == "C3_critic_machines_decoupe":
    af.encadre_ecart("E-25", "C1 « épaisseur maximale de la pièce » est noté − dans le cours (on l'attendrait +). "
                             "Le sens du cours est conservé : il ne change pas les poids CRITIC (la normalisation "
                             "min-max inversée et |ρ| donnent les mêmes σ et |ρ|), seulement le classement ultérieur.")
M = editeur_matrice(f"critic_{ident}", charger_matrice(ident), avec_poids=False, nom_fichier="critic.csv")
if M is None:
    st.stop()
try:
    r = poids_critic(M.valeurs, M.sens)
except ErreurMethode as exc:
    st.error(str(exc))
    st.stop()

alt, crit = M.alternatives, M.criteres
with af.etape("Étape 1 — Matrice normalisée rᵢⱼ"):
    af.matrice(r.R, alt, crit)
with af.etape("Étape 2 — Moyennes et corrélations ρⱼₖ", ouverte=True):
    af.vecteur(r.moyennes, crit, "r̄ⱼ")
    gr.afficher(gr.heatmap(r.rho, crit, crit, "Corrélations ρⱼₖ entre critères", divergente=True, zmin=-1, zmax=1))
with af.etape("Étape 3 — Écarts types, conflit et indice Cⱼ", ouverte=True):
    st.dataframe(pd.DataFrame({"σⱼ": np.round(r.sigma, 4), "Σₖ (1 − |ρⱼₖ|)": np.round(r.conflit, 4),
                               "Cⱼ": np.round(r.C, 4)}, index=crit).T)
with af.etape("Étape 4 — Poids wⱼ = Cⱼ / Σ Cⱼ", ouverte=True):
    af.vecteur(r.w, crit, "wⱼ")

c1, c2 = st.columns(2)
with c1:
    gr.afficher(gr.barres_groupees(crit, {"Contraste σⱼ": r.sigma, "Conflit Σ(1 − |ρ|) / n": r.conflit / M.n},
                                   "Contraste et conflit de chaque critère"))
with c2:
    gr.afficher(gr.barres(crit, r.w, "Poids CRITIC", "wⱼ"))
j = int(np.argmax(r.w))
st.success(f"**{crit[j]}** combine le plus de dispersion (σ = {r.sigma[j]:.3f}) et d'indépendance "
           f"(Σ(1 − |ρ|) = {r.conflit[j]:.3f}) : c'est le critère le plus lourd (w = {r.w[j]:.4f}).")

if st.button("Utiliser ces poids dans la page « Chaînage »", icon=":material/link:"):
    publier_poids(f"CRITIC — {ident}", M.criteres, r.w)
    st.toast("Poids CRITIC envoyés vers la page de chaînage.")
