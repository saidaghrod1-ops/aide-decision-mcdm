import numpy as np
import pandas as pd
import streamlit as st

from core.commun import ErreurMethode
from core.flou import intervalles as itv
from core.flou import nombres_flous as nf
from ui import affichage as af
from ui import graphiques as gr

af.entete("Nombres flous et nombres d'intervalle", "Logique floue")

st.header("Nombres flous trapézoïdaux")
st.markdown("$N(a, b, c, d)$ ; si $b = c$ : nombre flou triangulaire. Exemple : $N(4, 5, 6, 7)$ décrit la mention "
            "« moyenne » d'un étudiant. Pour $A$ et $B$ positifs :")
st.latex(r"\begin{aligned}A+B&=(a_1+b_1,\dots,a_4+b_4) & A-B&=(a_1-b_4,a_2-b_3,a_3-b_2,a_4-b_1)\\"
         r"A\otimes B&=(a_1b_1,\dots,a_4b_4) & A\oslash B&=(a_1/b_4,a_2/b_3,a_3/b_2,a_4/b_1),\quad \lambda A=(\lambda a_1,\dots,\lambda a_4)\end{aligned}")
c1, c2, c3 = st.columns([2, 2, 1])
A = np.array([c1.number_input(f"a{i + 1}", value=float(v), key=f"nfA{i}") for i, v in enumerate([4, 5, 6, 7])])
B = np.array([c2.number_input(f"b{i + 1}", value=float(v), key=f"nfB{i}") for i, v in enumerate([1, 2, 3, 4])])
lam = c3.number_input("λ", value=2.0, min_value=0.0)
resultats = {}
for nom, f in {"A + B": lambda: nf.addition(A, B), "A − B": lambda: nf.soustraction(A, B),
               "A ⊗ B": lambda: nf.multiplication(A, B), "λA": lambda: nf.scalaire(lam, A), "A ⊘ B": lambda: nf.division(A, B)}.items():
    try:
        resultats[nom] = f()
    except ErreurMethode as exc:
        st.error(f"{nom} : {exc}")
if resultats:
    st.dataframe(pd.DataFrame({k: np.round(v, 4) for k, v in resultats.items()}, index=["1", "2", "3", "4"]).T)
    gr.afficher(gr.nombres_flous({"A": A, "B": B, "A + B": resultats.get("A + B", A)}, None, "x"))

st.header("Nombres d'intervalle")
st.latex(r"A=[a_1,a_2],\quad m(A)=\frac{a_1+a_2}{2},\quad w(A)=\frac{a_2-a_1}{2},\quad A\le B\iff m(A)\le m(B)")
st.latex(r"A+B=[a_1+b_1,\,a_2+b_2],\quad A-B=[a_1-b_2,\,a_2-b_1],\quad \frac AB=\Big[\frac{a_1}{b_2},\frac{a_2}{b_1}\Big]\ (0\notin B)")
c1, c2 = st.columns(2)
IA = np.array([c1.number_input("A : a1", value=1.0), c1.number_input("A : a2", value=3.0)])
IB = np.array([c2.number_input("B : b1", value=2.0), c2.number_input("B : b2", value=5.0)])
try:
    res = {"A + B": itv.addition(IA, IB), "A − B": itv.soustraction(IA, IB), "A × B": itv.multiplication(IA, IB),
           "A / B": itv.division(IA, IB)}
    st.dataframe(pd.DataFrame({k: [f"[{v[0]:.4g} ; {v[1]:.4g}]"] for k, v in res.items()}, index=["résultat"]))
    st.markdown(f"m(A) = {itv.centre(IA):g}, m(B) = {itv.centre(IB):g} → **A {'≤' if itv.inferieur_ou_egal(IA, IB) else '>'} B**")
except ErreurMethode as exc:
    st.error(str(exc))
