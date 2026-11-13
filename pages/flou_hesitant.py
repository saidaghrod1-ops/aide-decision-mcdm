import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from core.commun import ErreurMethode
from core.flou import hesitant as hf
from ui import affichage as af
from ui import graphiques as gr
from ui.donnees import charger_tableau, editeur_tableau

af.entete("Ensembles flous hésitants", "Logique floue", "125–130",
          "Torra (2010) : quand plusieurs décideurs ne s'accordent pas sur un degré d'appartenance, celui-ci peut prendre "
          "un ensemble de valeurs h = {γ₁, γ₂, …, γₖ} ⊂ [0, 1].")

with st.expander("Théorie (p. 126–130)", expanded=False):
    st.latex(r"h^\alpha=\bigcup_{\gamma\in h}\{\gamma^\alpha\},\quad \alpha h=\bigcup\{1-(1-\gamma)^\alpha\},\quad "
             r"h_1\oplus h_2=\bigcup\{\gamma_1+\gamma_2-\gamma_1\gamma_2\},\quad h_1\otimes h_2=\bigcup\{\gamma_1\gamma_2\}")
    st.latex(r"s(h)=\frac1l\sum_{\gamma\in h}\gamma,\qquad \sigma(h)=\Big[\frac1l\sum_{\gamma\in h}(\gamma-s(h))^2\Big]^{1/2}")
    st.markdown("Comparaison : le plus grand score l'emporte ; à score égal, la **plus petite déviation** l'emporte.")
    st.latex(r"\lVert h_1-h_2\rVert=\frac1l\sum_{k=1}^{l}\lvert h_1^{\sigma(k)}-h_2^{\sigma(k)}\rvert,\qquad l=\max\{l(h_1),l(h_2)\}")
    af.encadre_ecart("E-20", "le cours ne dit pas comment compléter l'ensemble le plus court pour la distance : il est "
                             "prolongé par sa plus petite valeur (pessimiste, par défaut) ou sa plus grande (optimiste).")

table = editeur_tableau("hesitant_table", charger_tableau("C11_hesitants_p129", "tableau"),
                        titre="Éléments hésitants (valeurs séparées par « ; »)", colonnes_attendues=["nom", "valeurs"],
                        nom_fichier="hesitants.csv")
try:
    noms = [str(n) for n in table["nom"]]
    H = [hf.lire(v) for v in table["valeurs"]]
except ErreurMethode as exc:
    st.error(str(exc))
    st.stop()
if not H:
    st.stop()

st.header("1. Score, déviation et comparaison (exemple p. 129)")
st.dataframe(pd.DataFrame({"h": noms, "Valeurs": ["{" + "; ".join(f"{g:g}" for g in h) + "}" for h in H],
                           "s(h)": [round(hf.score(h), 4) for h in H], "σ(h)": [round(hf.deviation(h), 4) for h in H]}),
             hide_index=True)
ordre = hf.classer(H)
texte = noms[ordre[0]]
for i, j in zip(ordre, ordre[1:]):
    texte += (" = " if hf.comparer(H[i], H[j]) == 0 else " > ") + noms[j]
st.success(f"Classement : **{texte}**.")
egalites = [(noms[i], noms[j]) for a, i in enumerate(range(len(H))) for j in range(i + 1, len(H))
            if abs(hf.score(H[i]) - hf.score(H[j])) < 1e-12]
for a, b in egalites:
    st.caption(f"{a} et {b} ont le même score : la plus petite déviation l'emporte.")
fig = go.Figure()
for i, (n, h) in enumerate(zip(noms, H)):
    fig.add_trace(go.Scatter(x=list(h), y=[n] * len(h), mode="markers", name=n, marker=dict(size=11, color=gr.couleur(i)),
                             hovertemplate=n + " : γ = %{x}<extra></extra>"))
    fig.add_trace(go.Scatter(x=[hf.score(h)], y=[n], mode="markers", showlegend=False,
                             marker=dict(size=16, symbol="line-ns-open", color=gr.ENCRE, line=dict(width=3)),
                             error_x=dict(type="data", array=[hf.deviation(h)], color=gr.ENCRE_2),
                             hovertemplate=n + " : s = %{x:.4f}<extra></extra>"))
fig.update_layout(title="Valeurs γ, score s(h) (trait) et déviation σ(h) (barre)", paper_bgcolor=gr.SURFACE,
                  plot_bgcolor=gr.SURFACE, height=320, xaxis=dict(range=[0, 1]))
gr.afficher(fig)

st.header("2. Opérations et distance (p. 127, 130)")
c1, c2, c3 = st.columns(3)
i1 = c1.selectbox("h₁", range(len(H)), format_func=lambda i: noms[i])
i2 = c2.selectbox("h₂", range(len(H)), index=min(1, len(H) - 1), format_func=lambda i: noms[i])
a = c3.number_input("α (> 0)", 0.01, 20.0, 2.0)
fmt = lambda h: "{" + "; ".join(f"{g:.4g}" for g in h) + "}"
st.dataframe(pd.DataFrame({"Résultat": [fmt(hf.puissance(H[i1], a)), fmt(hf.scalaire(a, H[i1])), fmt(hf.somme(H[i1], H[i2])),
                                        fmt(hf.produit(H[i1], H[i2]))]},
                          index=["h₁^α", "α h₁", "h₁ ⊕ h₂", "h₁ ⊗ h₂"]))
extension = st.radio("Extension de l'ensemble le plus court", ["pessimiste", "optimiste"], horizontal=True)
l = max(len(H[i1]), len(H[i2]))
st.markdown(f"Ensembles prolongés à l = {l} : h₁ = {fmt(hf.prolonger(H[i1], l, extension))}, "
            f"h₂ = {fmt(hf.prolonger(H[i2], l, extension))}.")
st.success(f"Distance hésitante de Hamming ‖h₁ − h₂‖ = **{hf.distance_hamming(H[i1], H[i2], extension):.4f}**.")
