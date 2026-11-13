import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from core.classement.topsis import topsis
from core.commun import ErreurMethode
from core.donnees import MatriceDecision
from core.ia.dynamique import prevoir_arima
from ui import affichage as af
from ui import graphiques as gr
from ui.donnees import charger_matrice, charger_tableau, editeur_matrice, poids_valides

af.entete("MCDM dynamique", "MCDM + IA",
          "Choisir la meilleure action (tech, énergie, bancaire) selon des critères qui évoluent dans le temps : "
          "rendement (↑), volatilité (↓), liquidité (↑).")

with st.expander("Démarche", expanded=True):
    st.latex(r"\hat x_{ij}(t+1) = \text{Modèle}\big(x_{ij}(t)\big)")
    st.markdown("Un modèle de séries temporelles (ici **ARIMA**) prévoit chaque critère à $t + 1$ ; la matrice des "
                "prévisions est ensuite classée (ici TOPSIS). Limites : les erreurs de prévision se propagent au "
                "classement, qui dépend aussi du modèle choisi.")

series = charger_tableau("D7_dynamique_series", "series")
c1, c2, c3 = st.columns(3)
ordre = (int(c1.number_input("p", 0, 5, 1)), int(c2.number_input("d", 0, 2, 0)), int(c3.number_input("q", 0, 5, 0)))


@st.cache_data(show_spinner="Ajustement des modèles ARIMA…")
def _prevoir(df: pd.DataFrame, ordre):
    return {k: prevoir_arima(g.sort_values("periode")["valeur"].to_numpy(float), ordre).valeur
            for k, g in df.groupby(["alternative", "critere"], sort=False)}


alternatives = list(dict.fromkeys(series["alternative"]))
criteres = list(dict.fromkeys(series["critere"]))
try:
    prev = _prevoir(series, ordre)
except ErreurMethode as exc:
    st.error(str(exc))
    st.stop()
crit_vu = st.selectbox("Critère affiché", criteres)
fig = go.Figure()
for i, a in enumerate(alternatives):
    g = series[(series["alternative"] == a) & (series["critere"] == crit_vu)].sort_values("periode")
    fig.add_trace(go.Scatter(x=g["periode"], y=g["valeur"], name=a, mode="lines", line=dict(color=gr.couleur(i), width=2)))
    fig.add_trace(go.Scatter(x=[g["periode"].max() + 1], y=[prev[(a, crit_vu)]], mode="markers", showlegend=False,
                             marker=dict(size=12, symbol="diamond", color=gr.couleur(i))))
fig.update_layout(title=f"{crit_vu} : 60 mois et prévision à t + 1 (losange)", paper_bgcolor=gr.SURFACE,
                  plot_bgcolor=gr.SURFACE, height=380, legend=dict(orientation="h"))
gr.afficher(fig)

cours = charger_matrice("C13_dynamique_prevision")
source = st.radio("Matrice à classer", ["Prévisions ARIMA", "Tableau de prévision du cours"], horizontal=True)
if source == "Prévisions ARIMA":
    X = np.round([[prev[(a, c)] for c in criteres] for a in alternatives], 4)
    defaut, cle = MatriceDecision(tuple(alternatives), tuple(criteres), X, cours.sens), f"dyn_{ordre}"
else:
    defaut, cle = cours, "dyn_cours"
M = editeur_matrice(cle, defaut, nom_fichier="previsions.csv")
if M is None:
    st.stop()
w = poids_valides(M, cle)
if w is None:
    st.stop()
try:
    r = topsis(M.valeurs, M.sens, w)
except ErreurMethode as exc:
    st.error(str(exc))
    st.stop()
af.classement(M.alternatives, {"RC*": r.RC}, r.rangs)
st.success(f"Meilleure action à t + 1 : **{M.alternatives[int(np.argmax(r.RC))]}** (poids égaux par défaut).")
