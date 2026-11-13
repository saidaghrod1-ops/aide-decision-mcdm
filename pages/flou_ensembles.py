import json

import numpy as np
import pandas as pd
import streamlit as st

from core.commun import ErreurMethode
from core.donnees import chemin, dataset
from core.flou.appartenance import alpha_coupe, appartenance, caracteristique, par_points, proprietes
from ui import affichage as af
from ui import graphiques as gr


@st.cache_data
def exemples() -> dict:
    with chemin(dataset("C14_ensembles_flous_exemples")["fichiers"]["json"]).open(encoding="utf-8") as f:
        return json.load(f)


EX = exemples()
af.entete("Ensembles flous", "Logique floue", "96–107")

with st.expander("Types d'incertitude (p. 96)", expanded=False):
    st.markdown("1. **Incertitude aléatoire** (stochastique) : variabilité des phénomènes naturels, modélisée par les probabilités.  \n"
                "2. **Incertitude épistémique** (subjective) : due au raisonnement humain et au manque d'information ; "
                "la décision s'appuie sur des termes linguistiques. La **logique floue** (Zadeh, 1965) est adaptée à ce cas.")

# ---------------------------------------------------------------------------
st.header("1. Logique booléenne et logique floue (p. 97–104)")
st.markdown("Ensemble classique : $1_A(x) = 1$ si $x \\in A$, 0 sinon. Ensemble flou : "
            "$A = \\{\\langle x, \\mu_A(x)\\rangle \\mid x \\in X\\}$ avec $\\mu_A(x) \\in [0, 1]$.")
age = EX["age"]
x_age = np.linspace(*age["univers"], 1001)
age_val = st.slider("Âge", 0.0, 100.0, 30.0, 0.5)
series = {nom: appartenance(x_age, e["type"], e["parametres"]) for nom, e in age["ensembles"].items()}
series["Adulte (ensemble classique, ≥ 18 ans)"] = caracteristique(x_age, EX["adulte_crisp"]["seuil_age"])
gr.afficher(gr.fonctions_appartenance(x_age, series, "Ensembles de l'âge (p. 98 et 104)", "âge", zone=None))
degres = {nom: float(appartenance(np.array([age_val]), e["type"], e["parametres"])[0]) for nom, e in age["ensembles"].items()}
st.dataframe(pd.DataFrame({"μ(x)": degres}).T.round(3))
st.latex(r"\mu_{\text{Âge moyen}}(x)=\begin{cases}0 & x\le 25\\ \frac{x-25}{10} & 25<x\le 35\\ 1 & 35<x\le 55\\ "
         r"\frac{65-x}{10} & 55<x\le 65\\ 0 & x>65\end{cases}")
st.caption("Exemple du cours : à 45 ans, μ(âge moyen) = 1 ; à 30 ans, μ(âge moyen) = 0,5, autant que pour « jeune ». "
           "« Jeune » et « âgé » sont lus sur la figure de la p. 104.")

# ---------------------------------------------------------------------------
st.header("2. Fonctions d'appartenance et propriétés (p. 102–107)")
st.latex(r"h(A)=\sup_x \mu_A(x),\quad supp(A)=\{x \mid \mu_A(x)>0\},\quad noy(A)=\{x \mid \mu_A(x)=1\},\quad "
         r"\alpha\text{-cut}(A)=\{x \mid \mu_A(x)\ge\alpha\}")
st.markdown("A est **normalisé** si h(A) = 1 ; **convexe** si $\\mu_A(\\lambda x_1+(1-\\lambda)x_2)\\ge\\min(\\mu_A(x_1),\\mu_A(x_2))$.")
type_mf = st.radio("Fonction d'appartenance", ["triangulaire", "trapezoidale", "gaussienne", "définie par points"], horizontal=True,
                   format_func=lambda t: {"trapezoidale": "trapézoïdale"}.get(t, t))
c = st.columns(4)
try:
    if type_mf == "triangulaire":
        p = [c[i].number_input(n, value=v, key=f"tri_{n}") for i, (n, v) in enumerate(zip("abc", EX["average"]["parametres"]))]
        x = np.linspace(min(p) - 5, max(p) + 5, 2001); mu = appartenance(x, "triangulaire", p)
    elif type_mf == "trapezoidale":
        p = [c[i].number_input(n, value=float(v), key=f"trap_{n}") for i, (n, v) in enumerate(zip("abcd", [25, 35, 55, 65]))]
        x = np.linspace(min(p) - 10, max(p) + 10, 2001); mu = appartenance(x, "trapezoidale", p)
    elif type_mf == "gaussienne":
        m = c[0].number_input("m", value=15.0); s = c[1].number_input("σ", value=2.5, min_value=0.01)
        x = np.linspace(m - 5 * s, m + 5 * s, 2001); mu = appartenance(x, "gaussienne", (m, s))
    else:
        pts = st.data_editor(pd.DataFrame({"x": [0.0, 5, 10, 15, 20, 25, 30], "μ": [0, 1, 0.3, 0.3, 1, 0.2, 0]}),
                             num_rows="dynamic", hide_index=True, key="mf_points")
        pts = pts.dropna().sort_values("x")
        x = np.linspace(pts["x"].min(), pts["x"].max(), 2001); mu = par_points(x, pts["x"], pts["μ"])
except (ErreurMethode, ValueError) as exc:
    st.error(str(exc))
    st.stop()
alpha = st.slider("α (pour l'α-coupe)", 0.05, 1.0, 0.5, 0.05)
pr = proprietes(x, mu)
coupe = alpha_coupe(x, mu, alpha)
gr.afficher(gr.fonctions_appartenance(x, {"μ_A(x)": mu}, "Fonction d'appartenance et α-coupe", "x", zone=coupe, niveau=alpha))
fmt = lambda e: "∅" if e is None else (f"{{{e[0]:g}}}" if np.isclose(e[0], e[1]) else f"[{e[0]:.3f} ; {e[1]:.3f}]")
st.dataframe(pd.DataFrame({"Hauteur h(A)": [round(pr.hauteur, 4)], "Normalisé": ["oui" if pr.normalise else "non"],
                           "Support (enveloppe)": [fmt(pr.support)], "Noyau": [fmt(pr.noyau)],
                           f"{alpha:g}-coupe": [fmt(coupe)], "Convexe": ["oui" if pr.convexe else "non"]}), hide_index=True)
st.caption("Exemple du cours (p. 106), triangle (10, 15, 20) : noyau {15}, hauteur 1, support ]10, 20[ (ouvert, μ > 0), "
           "0,5-coupe [12,5 ; 17,5]. Les propriétés sont calculées sur une grille de 2 001 points. Pour voir un ensemble "
           "non convexe, choisissez « définie par points ».")

# ---------------------------------------------------------------------------
st.header("3. Variable linguistique (p. 107)")
pb = EX["pourboire"]
st.markdown("Une variable linguistique est un triplet $(V, X, T_V)$ : ici $V$ = pourboire, $X = \\mathbb{R}^+$, "
            "$T_V$ = {faible, moyen, élevé}.")
xp = np.linspace(*pb["univers"], 601)
gr.afficher(gr.fonctions_appartenance(xp, {n: appartenance(xp, e["type"], e["parametres"]) for n, e in pb["ensembles"].items()},
                                      "Pourboire", "pourboire"))
val = st.slider("Montant du pourboire", 0.0, 30.0, 12.0, 0.5)
st.dataframe(pd.DataFrame({n: [round(float(appartenance(np.array([val]), e["type"], e["parametres"])[0]), 3)]
                           for n, e in pb["ensembles"].items()}, index=["μ"]))
