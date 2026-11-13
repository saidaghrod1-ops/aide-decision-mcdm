import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from core.classement.ponderation_simple import waspas
from core.classement.topsis import topsis
from core.classement.vikor import vikor
from core.commun import ErreurMethode
from core.donnees import MatriceDecision
from core.ia.dynamique import prevoir_arima
from ui import affichage as af
from ui import graphiques as gr
from ui.donnees import charger_matrice, charger_tableau, editeur_matrice, editeur_tableau, poids_valides

af.entete("MCDM dynamique", "MCDM + IA", "93–94",
          "Choisir la meilleure action parmi A1 (tech), A2 (énergie) et A3 (bancaire) selon trois critères qui évoluent "
          "dans le temps : rendement attendu (↑), risque / volatilité (↓) et liquidité (↑).")

with st.expander("Démarche (cours p. 94)", expanded=True):
    st.markdown("Pour chaque alternative et chaque critère, un modèle de séries temporelles (ARIMA, LSTM, Transformers, "
                "GARCH…) prévoit la valeur à l'horizon $t + 1$, ce qui donne la matrice de décision finale :")
    st.latex(r"\hat x_{ij}(t+1) = \text{Modèle}\big(x_{ij}(t)\big)")
    st.markdown("**Limites** : propagation des erreurs (erreur de ML → MCDM biaisé) ; dépendance aux modèles de ML utilisés.  \n"
                "Cette page utilise **ARIMA**, premier modèle cité ; LSTM, Transformers et GARCH sont seulement cités.")

st.header("1. Séries temporelles")
st.caption("Jeu créé : 60 mois par action et par critère (processus autorégressifs centrés sur le tableau de la p. 94).")
series_df = editeur_tableau("dyn_series", charger_tableau("D7_dynamique_series", "series"),
                            colonnes_attendues=["periode", "alternative", "critere", "valeur"], nom_fichier="series.csv")
try:
    series_df = series_df.assign(valeur=pd.to_numeric(series_df["valeur"]), periode=pd.to_numeric(series_df["periode"]))
except ValueError:
    st.error("Les colonnes « periode » et « valeur » doivent être numériques.")
    st.stop()
alternatives = list(dict.fromkeys(series_df["alternative"]))
criteres = list(dict.fromkeys(series_df["critere"]))

st.header("2. Prévision ARIMA(p, d, q) à t + 1")
c1, c2, c3 = st.columns(3)
ordre = (c1.number_input("p", 0, 5, 1), c2.number_input("d", 0, 2, 0), c3.number_input("q", 0, 5, 0))


@st.cache_data(show_spinner="Ajustement des modèles ARIMA…")
def _prevoir(df: pd.DataFrame, ordre):
    res = {}
    for (a, c), g in df.groupby(["alternative", "critere"], sort=False):
        p = prevoir_arima(g.sort_values("periode")["valeur"].to_numpy(float), ordre)
        res[(a, c)] = (p.valeur, p.ecart_type_residus)
    return res


try:
    prev = _prevoir(series_df, tuple(int(o) for o in ordre))
    X_prev = np.array([[prev[(a, c)][0] for c in criteres] for a in alternatives])
except (ErreurMethode, KeyError, ValueError) as exc:
    st.error(f"Prévision impossible : {exc}")
    st.stop()

crit_vu = st.selectbox("Critère à afficher", criteres)
fig = go.Figure()
for i, a in enumerate(alternatives):
    g = series_df[(series_df["alternative"] == a) & (series_df["critere"] == crit_vu)].sort_values("periode")
    fig.add_trace(go.Scatter(x=g["periode"], y=g["valeur"], name=a, mode="lines", line=dict(color=gr.couleur(i), width=2)))
    fig.add_trace(go.Scatter(x=[g["periode"].max() + 1], y=[prev[(a, crit_vu)][0]], name=f"{a} — prévision t + 1",
                             mode="markers", marker=dict(size=12, symbol="diamond", color=gr.couleur(i)),
                             error_y=dict(type="data", array=[prev[(a, crit_vu)][1]], visible=True)))
fig.update_layout(title=f"{crit_vu} : historique et prévision (± 1 écart type des résidus)", paper_bgcolor=gr.SURFACE,
                  plot_bgcolor=gr.SURFACE, height=420, legend=dict(orientation="h"))
gr.afficher(fig)
cours = charger_matrice("C13_dynamique_prevision")
st.dataframe(pd.DataFrame(np.round(X_prev, 4), index=alternatives, columns=criteres).assign(Source="Prévision ARIMA"))
st.dataframe(pd.DataFrame(cours.valeurs, index=cours.alternatives, columns=cours.criteres).assign(Source="Tableau du cours (p. 94)"))

st.header("3. Matrice de décision et classement")
source = st.radio("Matrice à classer", ["Prévisions ARIMA", "Tableau du cours (p. 94)"], horizontal=True)
if source.startswith("Prévisions"):
    sens = np.array([cours.sens[cours.criteres.index(c)] if c in cours.criteres else 1 for c in criteres])
    defaut = MatriceDecision(tuple(alternatives), tuple(criteres), np.round(X_prev, 4), sens)
    cle = f"dyn_matrice_{'_'.join(map(str, ordre))}"
else:
    defaut, cle = cours, "dyn_matrice_cours"
st.caption("Le cours ne donne pas de poids pour cet exemple : poids égaux par défaut (E-19). Modifiez une prévision "
           "pour observer la propagation d'une erreur de prévision sur le classement.")
M = editeur_matrice(cle, defaut, nom_fichier="previsions.csv")
if M is None:
    st.stop()
w = poids_valides(M, cle)
if w is None:
    st.stop()
methode = st.radio("Méthode de classement", ["TOPSIS", "VIKOR", "WASPAS"], horizontal=True)
try:
    if methode == "TOPSIS":
        r = topsis(M.valeurs, M.sens, w); score, rg, nom = r.RC, r.rangs, "RC*"
    elif methode == "VIKOR":
        r = vikor(M.valeurs, M.sens, w); score, rg, nom = r.Q, r.rangs_Q, "Q (plus petit = meilleur)"
    else:
        r = waspas(M.valeurs, M.sens, w); score, rg, nom = r.Q, r.rangs_Q, "Q WASPAS"
except ErreurMethode as exc:
    st.error(str(exc))
    st.stop()
af.classement(M.alternatives, {nom: score}, rg)
st.success(f"Meilleure action à t + 1 selon {methode} : **{M.alternatives[int(np.argmin(rg))]}**.")
