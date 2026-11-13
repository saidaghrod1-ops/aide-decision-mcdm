import numpy as np
import pandas as pd
import streamlit as st

from core.commun import ErreurMethode
from core.donnees import dataset
from core.ponderation.bwm import bwm, controles
from ui import affichage as af
from ui import graphiques as gr
from ui.donnees import charger_tableau, choisir_dataset, editeur_tableau
from ui.etat import publier_poids

af.entete("BWM — Best-Worst Method", "Pondération subjective")

with st.expander("Théorie", expanded=True):
    st.markdown("L'expert choisit le critère le plus important **B** et le moins important **W**, puis note de 1 à 9 "
                "le vecteur **BO** ($a_{Bj}$ : B comparé à chaque critère) et le vecteur **OW** ($a_{jW}$ : chaque "
                "critère comparé à W). BWM demande $2n-3$ comparaisons contre $n(n-1)/2$ pour l'AHP.")
    st.latex(r"\min \xi \quad \text{s.c.}\quad \lvert w_B - w_j\,a_{Bj}\rvert \le \xi,\quad \lvert w_j - w_W\,a_{jW}\rvert \le \xi,"
             r"\quad \sum_j w_j = 1,\ w_j \ge 0")
    st.markdown("C'est un programme linéaire en $x = (w_1, \\dots, w_n, \\xi)$, écrit sous la forme $A\\,x \\le b$. "
                "Plus $\\xi^*$ est proche de 0, plus les jugements sont cohérents.")

ident = choisir_dataset("bwm", {"Exemple du cours : choisir un fournisseur": "C2_bwm_fournisseur"})
d = dataset(ident)
table = editeur_tableau(f"bwm_{ident}", charger_tableau(ident, "vecteurs"), colonnes_attendues=["critere", "BO", "OW"],
                        nom_fichier="bwm.csv",
                        column_config={"BO": st.column_config.NumberColumn("BO (B → autres)", min_value=1, max_value=9),
                                       "OW": st.column_config.NumberColumn("OW (autres → W)", min_value=1, max_value=9)})
try:
    criteres = [str(c) for c in table["critere"]]
    aB, aW = (pd.to_numeric(table[c]).to_numpy(float) for c in ("BO", "OW"))
    if np.isnan(aB).any() or np.isnan(aW).any() or len(criteres) < 2:
        raise ValueError
except (ValueError, KeyError):
    st.error("Chaque critère doit avoir une note BO et une note OW.")
    st.stop()
c1, c2 = st.columns(2)
best = c1.selectbox("Critère le plus important (B)", range(len(criteres)), index=int(np.argmin(aB)), format_func=lambda j: criteres[j])
worst = c2.selectbox("Critère le moins important (W)", range(len(criteres)), index=int(np.argmin(aW)), format_func=lambda j: criteres[j])
for message in controles(aB, aW, best, worst):
    st.warning(message)
try:
    r = bwm(aB, aW, best, worst)
except ErreurMethode as exc:
    st.error(str(exc))
    st.stop()

with af.etape("Contraintes sous la forme A x ≤ b"):
    colonnes = [f"w{j + 1}" for j in range(r.n)] + ["ξ"]
    df = pd.DataFrame(r.A_ub, columns=colonnes)
    df.insert(0, "Contrainte", list(r.libelles))
    st.dataframe(df, hide_index=True)
af.vecteur(r.w, criteres, "Poids wⱼ*")
st.metric("ξ*", f"{r.xi:.4f}")
gr.afficher(gr.barres(criteres, r.w, "Poids optimaux", "wⱼ"))
attendus = d.get("resultats_cours")
if attendus and np.array_equal(aB, [1, 3, 4, 8]) and np.array_equal(aW, [8, 5, 3, 1]) and (best, worst) == (0, 3):
    af.note(f"Le cours imprime w = ({', '.join(f'{x:.3f}' for x in attendus['poids'])}) avec ξ* ≈ 0,08. Ces poids ne "
            f"vérifient pas les contraintes du modèle ; la solution exacte de ce même programme linéaire est affichée ici "
            f"(même ξ* ≈ 0,08, même ordre des critères).")
if st.button("Utiliser ces poids dans la page « Chaînage »", icon=":material/link:"):
    publier_poids(f"BWM — {d['titre']}", tuple(criteres), r.w)
    st.toast("Poids BWM envoyés vers la page de chaînage.")
