import pandas as pd
import streamlit as st

from core.commun import ErreurMethode
from core.flou import orthopair as qr
from core.flou import pictural as pf
from ui import affichage as af
from ui import graphiques as gr

af.entete("Pythagoricien, q-ROF et pictural", "Logique floue", "Extensions des ensembles flous intuitionnistes.")

st.header("Pythagoricien et q-Rung Orthopair")
st.latex(r"\text{IFS : }\mu+\nu\le1\qquad \text{PFS : }\mu^2+\nu^2\le1\qquad \text{q-ROF : }\mu^q+\nu^q\le1,\quad "
         r"\pi=\sqrt[q]{1-\mu^q-\nu^q}")
c1, c2, c3 = st.columns(3)
mu = c1.slider("μ", 0.0, 1.0, 0.8, 0.01)
nu = c2.slider("ν", 0.0, 1.0, 0.7, 0.01)
q = c3.slider("q", 1, 10, 3)
st.dataframe(pd.DataFrame([{"Ensemble": nom, "Appartient": "oui" if qr.est_valide(mu, nu, k) else "non",
                            "π": round(qr.hesitation(mu, nu, k), 4) if qr.est_valide(mu, nu, k) else "—"}
                           for nom, k in (("Intuitionniste (q = 1)", 1), ("Pythagoricien (q = 2)", 2), (f"q-ROF (q = {q})", q))]),
             hide_index=True)
gr.afficher(gr.plan_mu_nu({"(μ, ν)": (mu, nu)}, {"μ + ν = 1": qr.frontiere(1), "μ² + ν² = 1": qr.frontiere(2),
                                                   f"μ^{q} + ν^{q} = 1": qr.frontiere(q)}, "Espaces IFN, PFN et q-ROFN"))
st.markdown("**À valeur d'intervalle** : $\\mu = [\\mu_L, \\mu_U]$, $\\nu = [\\nu_L, \\nu_U]$, "
            "$\\pi_L = \\sqrt[q]{1-\\mu_U^q-\\nu_U^q}$, $\\pi_U = \\sqrt[q]{1-\\mu_L^q-\\nu_L^q}$.")
c = st.columns(4)
try:
    pl, pu = qr.hesitation_intervalle(c[0].number_input("μL", 0.0, 1.0, 0.3), c[1].number_input("μU", 0.0, 1.0, 0.5),
                                      c[2].number_input("νL", 0.0, 1.0, 0.2), c[3].number_input("νU", 0.0, 1.0, 0.4), q)
    st.markdown(f"π = [{pl:.4f} ; {pu:.4f}]")
except ErreurMethode as exc:
    st.error(str(exc))

st.header("Ensembles flous picturaux")
st.markdown("Réponses **oui** (μ), **non** (ν), **abstention** (π) et **refus** = $1 - (\\mu + \\nu + \\pi)$, "
            "avec $\\mu + \\nu + \\pi \\le 1$ (exemple du vote).")
st.latex(r"A\oplus B=(\mu_A+\mu_B-\mu_A\mu_B,\ \nu_A\nu_B,\ \pi_A\pi_B),\quad \lambda A=(1-(1-\mu_A)^\lambda,\ \nu_A^\lambda,\ \pi_A^\lambda)")
c1, c2, c3 = st.columns(3)
try:
    A = pf.creer(c1.number_input("μ_A", 0.0, 1.0, 0.3), c1.number_input("ν_A", 0.0, 1.0, 0.3), c1.number_input("π_A", 0.0, 1.0, 0.3))
    B = pf.creer(c2.number_input("μ_B", 0.0, 1.0, 0.5), c2.number_input("ν_B", 0.0, 1.0, 0.2), c2.number_input("π_B", 0.0, 1.0, 0.1))
    lam = c3.number_input("λ", 0.01, 20.0, 2.0)
    res = {"A": A, "B": B, "A ⊕ B": pf.addition(A, B), "λA": pf.scalaire(lam, A)}
    st.dataframe(pd.DataFrame({k: [round(x.mu, 4), round(x.nu, 4), round(x.pi, 4), round(x.refus, 4)] for k, x in res.items()},
                              index=["μ (oui)", "ν (non)", "π (abstention)", "refus"]))
    if not all(x.valide for x in res.values()):
        st.warning("Un résultat sort de l'ensemble pictural (μ + ν + π > 1).")
except ErreurMethode as exc:
    st.error(str(exc))
