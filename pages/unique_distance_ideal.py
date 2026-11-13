import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from core.alternative_unique.distance_ideal import distance_ideal
from core.alternative_unique.maut import maut
from core.commun import ErreurMethode
from ui import affichage as af
from ui import graphiques as gr
from ui.alternative_unique import editeur_dossier

af.entete("Distance à l'idéal", "Évaluation d'une alternative unique (évaluation absolue)", "89",
          "Même avec une seule alternative, on peut la comparer à un point idéal A⁺ et à un point anti-idéal A⁻ "
          "(logique de TOPSIS) : on obtient un indice de performance.")

with st.expander("Théorie (cours p. 89)", expanded=True):
    st.latex(r"D^+=\sqrt{\sum_j (x_j-x_j^+)^2},\qquad D^-=\sqrt{\sum_j (x_j-x_j^-)^2},\qquad C=\frac{D^-}{D^++D^-}")
    af.encadre_ecart("E-15", "la formule du cours porte sur les valeurs brutes, d'unités différentes. Le calcul est fait "
                             "sur les utilités normalisées uⱼ ∈ [0, 1] (standardisation recommandée p. 54) : "
                             "A⁺ = (1, …, 1) et A⁻ = (0, …, 0). La formule reste non pondérée, comme dans le cours.")

st.caption("Jeu créé : scoring de crédit (même dossier que la page MAUT).")
dossier = editeur_dossier("distance_dossier")
if dossier is None:
    st.stop()
try:
    m = maut(dossier.valeurs, dossier.xmin, dossier.xmax, dossier.sens, dossier.poids)
    r = distance_ideal(m.u)
except ErreurMethode as exc:
    st.error(str(exc))
    st.stop()

with af.etape("Utilités et écarts aux points idéal et anti-idéal", ouverte=True):
    st.dataframe(pd.DataFrame({"uⱼ": np.round(r.u, 4), "(uⱼ − 1)²": np.round((r.u - 1) ** 2, 4),
                               "(uⱼ − 0)²": np.round(r.u ** 2, 4)}, index=dossier.criteres).T)
c1, c2, c3 = st.columns(3)
c1.metric("D⁺ (distance à l'idéal)", f"{r.D_plus:.4f}")
c2.metric("D⁻ (distance à l'anti-idéal)", f"{r.D_moins:.4f}")
c3.metric("Indice C", f"{r.C:.4f}")

fig = go.Figure()
theta = list(dossier.criteres) + [dossier.criteres[0]]
for nom, valeurs, coul in (("Point idéal A⁺", np.ones(len(r.u)), gr.PALETTE[2]), ("Alternative", r.u, gr.PALETTE[0])):
    fig.add_trace(go.Scatterpolar(r=list(valeurs) + [valeurs[0]], theta=theta, name=nom, line=dict(color=coul, width=2),
                                  fill="toself" if nom == "Alternative" else None, opacity=0.8))
fig.update_layout(polar=dict(radialaxis=dict(range=[0, 1])), paper_bgcolor=gr.SURFACE, height=420,
                  title="Utilités de l'alternative et point idéal", legend=dict(orientation="h"))
gr.afficher(fig)
st.info(f"C = {r.C:.4f} (0 = anti-idéal, 1 = idéal). Pour comparaison, le score MAUT pondéré de la même alternative vaut "
        f"S(A) = {m.S:.4f} : la distance à l'idéal du cours n'utilise pas les poids, d'où l'écart entre les deux indices.")
