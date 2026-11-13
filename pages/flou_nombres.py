import numpy as np
import pandas as pd
import streamlit as st

from core.commun import ErreurMethode
from core.flou import intervalles as itv
from core.flou import nombres_flous as nf
from ui import affichage as af
from ui import graphiques as gr

af.entete("Nombres flous et nombres d'intervalle", "Logique floue", "108–111")

st.header("1. Nombres flous trapézoïdaux (p. 108–109)")
st.markdown("Un nombre flou trapézoïdal $N(a, b, c, d)$ généralise un nombre réel. Exemple : $N(4, 5, 6, 7)$ décrit la "
            "mention « moyenne » d'un étudiant. Si $b = c$ : nombre flou triangulaire ; si $a = b = c = d$ : nombre réel.  \n"
            "Pour $A(a_1, a_2, a_3, a_4)$ et $B(b_1, b_2, b_3, b_4)$ positifs :")
st.latex(r"\begin{aligned}A+B&=(a_1+b_1,a_2+b_2,a_3+b_3,a_4+b_4) & A-B&=(a_1-b_4,a_2-b_3,a_3-b_2,a_4-b_1)\\"
         r"A\otimes B&=(a_1b_1,a_2b_2,a_3b_3,a_4b_4) & \lambda A&=(\lambda a_1,\lambda a_2,\lambda a_3,\lambda a_4)\\"
         r"A\oslash B&=(a_1/b_4,a_2/b_3,a_3/b_2,a_4/b_1)\end{aligned}")


def saisir(nom: str, defaut):
    st.markdown(f"**{nom}**")
    cols = st.columns(4)
    return np.array([cols[i].number_input(f"{nom}{i + 1}", value=float(v), key=f"nf_{nom}{i}") for i, v in enumerate(defaut)])


c1, c2 = st.columns(2)
with c1:
    A = saisir("A", [4, 5, 6, 7])
with c2:
    B = saisir("B", [1, 2, 3, 4])
lam = st.number_input("λ", value=2.0, min_value=0.0)
resultats, erreurs = {}, {}
for nom, f in {"A + B": lambda: nf.addition(A, B), "A − B": lambda: nf.soustraction(A, B),
               "A ⊗ B": lambda: nf.multiplication(A, B), "λA": lambda: nf.scalaire(lam, A),
               "A ⊘ B": lambda: nf.division(A, B)}.items():
    try:
        resultats[nom] = f()
    except ErreurMethode as exc:
        erreurs[nom] = str(exc)
for nom, e in erreurs.items():
    st.error(f"{nom} : {e}")
if resultats:
    st.dataframe(pd.DataFrame({k: np.round(v, 4) for k, v in resultats.items()}, index=["1", "2", "3", "4"]).T)
    choix = st.selectbox("Opération à tracer", list(resultats))
    gr.afficher(gr.nombres_flous({"A": A, "B": B, choix: resultats[choix]}, f"A, B et {choix}", "x"))

# ---------------------------------------------------------------------------
st.header("2. Nombres d'intervalle (p. 110–111)")
st.markdown("$A = [a_1, a_2]$ ; fonction d'appartenance égale à 1 sur l'intervalle et 0 ailleurs (nombre flou spécial).")
st.latex(r"m(A)=\frac{a_1+a_2}{2},\qquad w(A)=\frac{a_2-a_1}{2},\qquad A\le B\iff m(A)\le m(B)")
st.latex(r"A+B=[a_1+b_1,a_2+b_2],\ A-B=[a_1-b_2,a_2-b_1],\ A\times B=[\min(\cdot),\max(\cdot)],\ "
         r"\frac AB=\left[\frac{a_1}{b_2},\frac{a_2}{b_1}\right]\ (0\notin B)")
c1, c2, c3 = st.columns(3)
IA = np.array([c1.number_input("a1", value=1.0), c1.number_input("a2", value=3.0)])
IB = np.array([c2.number_input("b1", value=2.0), c2.number_input("b2", value=5.0)])
lam_i = c3.number_input("λ (≥ 0)", value=2.0, min_value=0.0, key="lam_itv")
af.encadre_ecart("E-27", "le cours définit a1 < a2 puis évoque le cas a1 = a2 : l'application accepte a1 ≤ a2.")
try:
    lignes = {"λA": itv.scalaire(lam_i, IA), "A + B": itv.addition(IA, IB), "A − B": itv.soustraction(IA, IB),
              "A × B": itv.multiplication(IA, IB)}
    try:
        lignes["A / B"] = itv.division(IA, IB)
    except ErreurMethode as exc:
        st.warning(f"A / B : {exc}")
    st.dataframe(pd.DataFrame({k: [f"[{v[0]:.4g} ; {v[1]:.4g}]"] for k, v in lignes.items()}, index=["résultat"]))
    st.markdown(f"m(A) = {itv.centre(IA):g}, w(A) = {itv.largeur(IA):g} ; m(B) = {itv.centre(IB):g}, w(B) = {itv.largeur(IB):g} "
                f"→ **A {'≤' if itv.inferieur_ou_egal(IA, IB) else '>'} B**.")
except ErreurMethode as exc:
    st.error(str(exc))
