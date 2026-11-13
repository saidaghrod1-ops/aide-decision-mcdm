import numpy as np
import pandas as pd
import streamlit as st

from core.commun import ErreurMethode
from core.donnees import lire_sens
from core.mcdm_flou.alternative_unique import evaluation_floue
from ui import affichage as af
from ui import graphiques as gr
from ui.donnees import charger_tableau, editeur_tableau

af.entete("Évaluation floue d'une alternative unique", "MCDM flou",
          "Chaque évaluation est un nombre flou triangulaire x̃ = (a, b, c) : pessimiste, plus probable, optimiste.")

with st.expander("Méthode", expanded=True):
    st.latex(r"\text{1. } \tilde r=(a,b,c)\div c_{max}\qquad \text{2. } \tilde S=\sum_i w_i\,\tilde x_i\qquad "
             r"\text{3. } S=\frac{a+b+c}{3}")
    st.markdown("La largeur $c - a$ de $\\tilde S$ mesure l'incertitude, à surveiller en plus du score.")

st.caption("Exemple du cours : ROA et liquidité.")
table = editeur_tableau("flou_unique", charger_tableau("C9_flou_alternative_unique", "tableau"),
                        colonnes_attendues=["critere", "sens", "poids", "a", "b", "c"], nom_fichier="evaluation_floue.csv",
                        column_config={"sens": st.column_config.SelectboxColumn("sens", options=["+", "-"])})
try:
    criteres = [str(c) for c in table["critere"]]
    D = table[["a", "b", "c"]].apply(pd.to_numeric).to_numpy(float)
    r = evaluation_floue(D, pd.to_numeric(table["poids"]).to_numpy(float), np.array([lire_sens(s) for s in table["sens"]]))
except (ErreurMethode, ValueError, KeyError) as exc:
    st.error(f"Données invalides : {exc}")
    st.stop()
st.dataframe(pd.DataFrame({"R̃ normalisée": af.triplets(r.R, 3), "wᵢ · x̃ᵢ": af.triplets(r.termes, 3)}, index=criteres))
c1, c2, c3 = st.columns(3)
c1.metric("S̃ agrégé", af.triplets(r.S_flou, 2)[0])
c2.metric("Score défuzzifié S", f"{r.S:.4f}")
c3.metric("Largeur (incertitude)", f"{r.largeur:.4f}")
gr.afficher(gr.nombres_flous({**{c: t for c, t in zip(criteres, r.R)}, "S̃": r.S_flou}, None, "valeur normalisée"))
