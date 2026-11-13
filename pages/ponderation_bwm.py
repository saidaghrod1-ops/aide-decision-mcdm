import numpy as np
import pandas as pd
import streamlit as st

from core.commun import ErreurMethode
from core.donnees import dataset
from core.ponderation.bwm import bwm, controles, nb_comparaisons, violation_max
from ui import affichage as af
from ui import graphiques as gr
from ui.donnees import charger_tableau, choisir_dataset, editeur_tableau
from ui.etat import publier_poids

af.entete("BWM — Best-Worst Method", "Pondération subjective", "35–43")

with st.expander("Théorie (cours p. 35–41)", expanded=False):
    st.markdown("""
**BWM (Rezaei, 2015).** L'expert désigne le critère le plus important (**Best**, B) et le moins important (**Worst**, W),
puis compare chaque critère à ces deux repères seulement, avec les scores 1 à 9 de l'AHP.

1. Choisir B et W. 2. Vecteur **Best-to-Others** (BO) : $a_{Bj}$. 3. Vecteur **Others-to-Worst** (OW) : $a_{jW}$.
4. Calculer les poids optimaux et l'indicateur de cohérence $\\xi^*$.

Contrôles de bon sens : $a_{BB} = a_{WW} = 1$, et la note la plus élevée du vecteur BO porte sur W.
Idéalement $w_B / w_j = a_{Bj}$ et $w_j / w_W = a_{jW}$ ; on cherche les poids qui minimisent le plus grand écart $\\xi$ :
""")
    st.latex(r"\min \xi \quad \text{s.c.}\quad \lvert w_B - w_j\,a_{Bj}\rvert \le \xi,\quad \lvert w_j - w_W\,a_{jW}\rvert \le \xi,"
             r"\quad \sum_j w_j = 1,\ w_j \ge 0")
    st.markdown("Forme linéaire (p. 40), avec $x = (w_1, \\dots, w_n, \\xi)$ et $f(x) = \\xi$ :")
    st.latex(r"w_B - w_j a_{Bj} - \xi \le 0,\quad -\xi - w_B + w_j a_{Bj} \le 0,\quad "
             r"w_j - w_W a_{jW} - \xi \le 0,\quad -\xi - w_j + w_W a_{jW} \le 0")
    st.markdown("Programme linéaire (p. 41) : $\\min_x c^T x$ sous $A_{ub}\\,x \\le b_{ub}$, $A_{eq}\\,x = b_{eq}$ et "
                "$l \\le x \\le u$ (contraintes complétées, E-28). Résolution : `scipy.optimize.linprog`.")

ident = choisir_dataset("bwm", {"Exemple : choisir un fournisseur (p. 37)": "C2_bwm_fournisseur"})
d = dataset(ident)
defaut = charger_tableau(ident, "vecteurs")
table = editeur_tableau(f"bwm_{ident}", defaut, titre="Vecteurs BO et OW (notes de 1 à 9)",
                        colonnes_attendues=["critere", "BO", "OW"], nom_fichier="bwm.csv",
                        column_config={"critere": st.column_config.TextColumn("Critère", required=True),
                                       "BO": st.column_config.NumberColumn("BO : B → autres", min_value=1, max_value=9),
                                       "OW": st.column_config.NumberColumn("OW : autres → W", min_value=1, max_value=9)})
try:
    criteres = [str(c) for c in table["critere"]]
    aB = pd.to_numeric(table["BO"]).to_numpy(float)
    aW = pd.to_numeric(table["OW"]).to_numpy(float)
    if np.isnan(aB).any() or np.isnan(aW).any() or len(criteres) < 2:
        raise ValueError
except (ValueError, KeyError):
    st.error("Chaque critère doit avoir une note BO et une note OW.")
    st.stop()

c1, c2 = st.columns(2)
best_defaut = criteres.index(d["best"]) if d.get("best") in criteres else int(np.argmin(aB))
worst_defaut = criteres.index(d["worst"]) if d.get("worst") in criteres else int(np.argmin(aW))
best = c1.selectbox("Critère le plus important (B)", range(len(criteres)), index=best_defaut, format_func=lambda j: criteres[j])
worst = c2.selectbox("Critère le moins important (W)", range(len(criteres)), index=worst_defaut, format_func=lambda j: criteres[j])
for message in controles(aB, aW, best, worst):
    st.warning(message)
n_bwm, n_ahp = nb_comparaisons(len(criteres))
st.caption(f"Nombre de comparaisons : BWM 2n − 3 = {n_bwm} ; AHP n(n − 1)/2 = {n_ahp}.")

try:
    r = bwm(aB, aW, best, worst)
except ErreurMethode as exc:
    st.error(str(exc))
    st.stop()

with af.etape("Étape 4.a — Variables, objectif et contraintes sous la forme A x ≤ b", ouverte=True):
    colonnes = [f"w{j + 1}" for j in range(r.n)] + ["ξ"]
    st.markdown(f"Variables $x = ({', '.join(colonnes)})$, objectif $c = {list(r.c.astype(int))}$ (on minimise ξ).")
    df = pd.DataFrame(r.A_ub, columns=colonnes)
    df.insert(0, "Contrainte", list(r.libelles))
    df["≤ b"] = r.b_ub
    st.dataframe(df, hide_index=True)
    st.markdown(f"Égalité : $A_{{eq}} = {r.A_eq.astype(int).tolist()[0]}$, $b_{{eq}} = 1$ ; bornes $w_j \\ge 0$, $\\xi \\ge 0$.")
with af.etape("Étape 4.b — Solution optimale", ouverte=True):
    af.vecteur(r.w, criteres, "wⱼ*")
    st.markdown(f"**ξ\\* = {r.xi:.4f}** — plus ξ\\* est proche de 0, plus les jugements de l'expert sont cohérents.")

gr.afficher(gr.barres(criteres, r.w, "Poids optimaux (BWM)", "wⱼ"))
ecarts = pd.DataFrame({"|w_B − a_Bj w_j|": np.round(r.ecarts_best, 4), "|w_j − a_jW w_W|": np.round(r.ecarts_worst, 4)},
                      index=criteres)
st.markdown("**Écarts aux égalités idéales** (les contraintes actives atteignent ξ\\*)")
st.dataframe(ecarts)
st.success(f"Le critère le plus lourd est **{criteres[int(np.argmax(r.w))]}** (w = {r.w.max():.3f}) ; "
           f"ξ* = {r.xi:.3f}.")

attendus = d.get("resultats_cours")
if attendus and np.array_equal(aB, [1, 3, 4, 8]) and np.array_equal(aW, [8, 5, 3, 1]) and (best, worst) == (0, 3):
    w_cours = np.array(attendus["poids"])
    af.encadre_ecart("E-1", (
        f"le cours (p. 42) donne w = ({', '.join(f'{x:.3f}' for x in w_cours)}) et ξ* = {attendus['xi']}. "
        f"Le programme linéaire du cours, avec ces mêmes vecteurs, a une solution **unique** : "
        f"w = ({', '.join(f'{x:.4f}' for x in r.w)}), ξ* = {r.xi:.4f}. Le ξ* concorde (≈ 0,08) et l'ordre des critères "
        f"est le même, mais les poids imprimés violent les contraintes du modèle : leur plus grand écart vaut "
        f"{violation_max(aB, aW, best, worst, w_cours):.3f}, bien au-dessus de ξ*. L'application affiche donc la "
        "solution du modèle du cours."))
    st.dataframe(pd.DataFrame({"Poids imprimés p. 42": w_cours, "Solution du modèle (application)": np.round(r.w, 4)},
                              index=criteres))

if st.button("Utiliser ces poids dans la page « Chaînage »", icon=":material/link:"):
    publier_poids(f"BWM — {d['titre']}", tuple(criteres), r.w)
    st.toast("Poids BWM envoyés vers la page de chaînage.")
