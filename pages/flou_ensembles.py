import json

import numpy as np
import pandas as pd
import streamlit as st

from core.commun import ErreurMethode
from core.donnees import chemin, dataset
from core.flou.appartenance import alpha_coupe, appartenance, caracteristique, proprietes
from ui import affichage as af
from ui import graphiques as gr


@st.cache_data
def exemples() -> dict:
    with chemin(dataset("C14_ensembles_flous_exemples")["fichiers"]["json"]).open(encoding="utf-8") as f:
        return json.load(f)


EX = exemples()
af.entete("Ensembles flous", "Logique floue",
          "La logique floue (Zadeh, 1965) traite l'incertitude subjective : un élément appartient à un ensemble avec un "
          "degré μ(x) ∈ [0, 1], et non plus seulement 0 ou 1.")

with st.expander("Théorie", expanded=True):
    st.latex(r"\text{Ensemble classique : } 1_A(x)\in\{0,1\}\qquad \text{Ensemble flou : } A=\{\langle x,\mu_A(x)\rangle\},\ \mu_A(x)\in[0,1]")
    st.latex(r"\text{Triangulaire }(a,b,c),\quad \text{trapézoïdale }(a,b,c,d),\quad \text{gaussienne } \mu(x)=e^{-\frac{(x-m)^2}{2\sigma^2}}")
    st.latex(r"h(A)=\sup\mu_A,\quad supp(A)=\{x\mid\mu_A(x)>0\},\quad noy(A)=\{x\mid\mu_A(x)=1\},\quad "
             r"\alpha\text{-cut}=\{x\mid\mu_A(x)\ge\alpha\}")

st.header("Exemple : l'âge")
age = EX["age"]
x_age = np.linspace(*age["univers"], 1001)
series = {nom: appartenance(x_age, e["type"], e["parametres"]) for nom, e in age["ensembles"].items()}
series["Adulte (classique, ≥ 18 ans)"] = caracteristique(x_age, EX["adulte_crisp"]["seuil_age"])
gr.afficher(gr.fonctions_appartenance(x_age, series, None, "âge"))
age_val = st.slider("Âge", 0.0, 100.0, 30.0, 0.5)
st.dataframe(pd.DataFrame({n: [round(float(appartenance(np.array([age_val]), e["type"], e["parametres"])[0]), 3)]
                           for n, e in age["ensembles"].items()}, index=["μ"]))
st.caption("À 45 ans, μ(âge moyen) = 1 ; à 30 ans, μ(âge moyen) = 0,5, autant que pour « jeune ».")

st.header("Fonction d'appartenance et propriétés")
type_mf = st.radio("Type", ["triangulaire", "trapezoidale", "gaussienne"], horizontal=True,
                   format_func=lambda t: {"trapezoidale": "trapézoïdale"}.get(t, t))
c = st.columns(4)
try:
    if type_mf == "triangulaire":
        p = [c[i].number_input(n, value=float(v), key=f"tri_{n}") for i, (n, v) in enumerate(zip("abc", EX["average"]["parametres"]))]
    elif type_mf == "trapezoidale":
        p = [c[i].number_input(n, value=float(v), key=f"trap_{n}") for i, (n, v) in enumerate(zip("abcd", [25, 35, 55, 65]))]
    else:
        p = [c[0].number_input("m", value=15.0), c[1].number_input("σ", value=2.5, min_value=0.01)]
    bornes = (min(p) - 5, max(p) + 5) if type_mf != "gaussienne" else (p[0] - 5 * p[1], p[0] + 5 * p[1])
    x = np.linspace(*bornes, 2001)
    mu = appartenance(x, type_mf, p)
except ErreurMethode as exc:
    st.error(str(exc))
    st.stop()
alpha = st.slider("α", 0.05, 1.0, 0.5, 0.05)
pr, coupe = proprietes(x, mu), alpha_coupe(x, mu, alpha)
gr.afficher(gr.fonctions_appartenance(x, {"μ_A(x)": mu}, None, "x", zone=coupe, niveau=alpha))
fmt = lambda e: "∅" if e is None else (f"{{{e[0]:g}}}" if np.isclose(e[0], e[1]) else f"[{e[0]:.2f} ; {e[1]:.2f}]")
st.dataframe(pd.DataFrame({"Hauteur": [round(pr.hauteur, 3)], "Support": [fmt(pr.support)], "Noyau": [fmt(pr.noyau)],
                           f"{alpha:g}-coupe": [fmt(coupe)], "Convexe": ["oui" if pr.convexe else "non"]}), hide_index=True)
st.caption("Exemple « Average » (10, 15, 20) : noyau {15}, hauteur 1, support ]10, 20[, 0,5-coupe [12,5 ; 17,5].")

st.header("Variable linguistique : le pourboire")
pb = EX["pourboire"]
xp = np.linspace(*pb["univers"], 601)
gr.afficher(gr.fonctions_appartenance(xp, {n: appartenance(xp, e["type"], e["parametres"]) for n, e in pb["ensembles"].items()},
                                      None, "pourboire"))
st.caption("Variable linguistique (V, X, T_V) : V = pourboire, X = ℝ⁺, T_V = {faible, moyen, élevé}.")
