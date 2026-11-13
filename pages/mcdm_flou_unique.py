import numpy as np
import pandas as pd
import streamlit as st

from core.commun import ErreurMethode
from core.donnees import lire_sens
from core.mcdm_flou.alternative_unique import evaluation_floue
from ui import affichage as af
from ui import graphiques as gr
from ui.donnees import charger_tableau, editeur_tableau

af.entete("Évaluation floue d'une alternative unique", "MCDM flou", "132–134",
          "Les évaluations des experts s'expriment souvent par des termes (bon, passable, mauvais) plutôt que par des "
          "chiffres : chaque évaluation devient un nombre flou triangulaire x̃ = (a, b, c) — a pessimiste, b le plus "
          "probable, c optimiste.")

with st.expander("Méthode (cours p. 133–134)", expanded=True):
    st.markdown("**1. Normalisation** : $(a, b, c) \\div c_{max}$, avec $c_{max}$ le plus grand $c$ parmi les alternatives.  \n"
                "**2. Agrégation** : $\\tilde S = \\sum_i w_i \\cdot \\tilde x_i$ (somme pondérée).  \n"
                "**3. Défuzzification** (centre de gravité) : $S = (a + b + c)/3$, un score unique pour classer.  \n"
                "La **largeur** $c - a$ de $\\tilde S$ mesure l'incertitude, à surveiller en plus du score.")
    st.caption("Le cours ne traite ici que des critères bénéfice ; pour un critère coût, la normalisation de la TOPSIS floue "
               "(p. 136, ordre corrigé E-4) est appliquée.")

table = editeur_tableau("flou_unique", charger_tableau("C9_flou_alternative_unique", "tableau"),
                        titre="Évaluations floues de l'alternative (exemple p. 134)",
                        colonnes_attendues=["critere", "sens", "poids", "a", "b", "c"], nom_fichier="evaluation_floue.csv",
                        column_config={"sens": st.column_config.SelectboxColumn("sens", options=["+", "-"])})
try:
    criteres = [str(c) for c in table["critere"]]
    D = table[["a", "b", "c"]].apply(pd.to_numeric).to_numpy(float)
    w = pd.to_numeric(table["poids"]).to_numpy(float)
    sens = np.array([lire_sens(s) for s in table["sens"]])
    r = evaluation_floue(D, w, sens)
except (ErreurMethode, ValueError, KeyError) as exc:
    st.error(f"Données invalides : {exc}")
    st.stop()

with af.etape("1. Normalisation", ouverte=True):
    st.dataframe(pd.DataFrame({"D̃ (a ; b ; c)": af.triplets(D, 3), "R̃ normalisée": af.triplets(r.R, 3), "wᵢ": w},
                              index=criteres))
with af.etape("2. Agrégation S̃ = Σ wᵢ x̃ᵢ", ouverte=True):
    st.dataframe(pd.DataFrame({"wᵢ · x̃ᵢ": af.triplets(r.termes, 3)}, index=criteres))
    st.markdown(f"S̃ = **{af.triplets(r.S_flou, 3)[0]}**")
with af.etape("3. Défuzzification", ouverte=True):
    st.latex(rf"S = \frac{{{r.S_flou[0]:.4f} + {r.S_flou[1]:.4f} + {r.S_flou[2]:.4f}}}{{3}} = {r.S:.4f}")
c1, c2 = st.columns(2)
c1.metric("Score défuzzifié S", f"{r.S:.4f}")
c2.metric("Largeur c − a (incertitude)", f"{r.largeur:.4f}")
gr.afficher(gr.nombres_flous({**{f"R̃ {c}": t for c, t in zip(criteres, r.R)}, "S̃ agrégé": r.S_flou},
                             "Nombres flous normalisés et score agrégé", "valeur normalisée"))
st.caption("Résultats du cours (p. 134) : S̃ = (0,74 ; 0,85 ; 1,00), S = 0,86, largeur 0,26.")
