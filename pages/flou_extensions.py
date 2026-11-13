import pandas as pd
import streamlit as st

from core.commun import ErreurMethode
from core.flou import orthopair as qr
from core.flou import pictural as pf
from ui import affichage as af
from ui import graphiques as gr

af.entete("Ensembles pythagoriciens, q-ROF et picturaux", "Logique floue", "118–124",
          "Extensions des ensembles flous intuitionnistes présentées par le cours.")

st.header("1. Pythagoricien, q-Rung Orthopair et q-ROF à valeur d'intervalle (p. 118–121)")
st.latex(r"\text{Intuitionniste : }\mu+\nu\le1\qquad \text{Pythagoricien : }\mu^2+\nu^2\le1,\ \pi=\sqrt{1-\mu^2-\nu^2}")
st.latex(r"\text{q-ROF } (q\ge1) : \mu^q+\nu^q\le1,\qquad \pi=\sqrt[q]{1-\mu^q-\nu^q}")
af.encadre_ecart("E-2", "la p. 119 écrit π = (1 − μ^q + ν^q)^(1/q). Le signe + est une coquille : la formule cohérente avec la "
                        "p. 118 (q = 2) et la p. 121 est (1 − μ^q − ν^q)^(1/q).")
c1, c2, c3 = st.columns(3)
mu = c1.slider("μ (appartenance)", 0.0, 1.0, 0.8, 0.01)
nu = c2.slider("ν (non-appartenance)", 0.0, 1.0, 0.7, 0.01)
q = c3.slider("q", 1, 10, 3)
lignes = []
for nom, qq in (("Intuitionniste (q = 1)", 1), ("Pythagoricien (q = 2)", 2), (f"q-ROF (q = {q})", q)):
    valide = qr.est_valide(mu, nu, qq)
    lignes.append({"Ensemble": nom, "μ^q + ν^q": round(mu ** qq + nu ** qq, 4), "Appartient": "oui" if valide else "non",
                   "π": round(qr.hesitation(mu, nu, qq), 4) if valide else "—"})
st.dataframe(pd.DataFrame(lignes), hide_index=True)
pq = qr.plus_petit_q(mu, nu)
st.markdown(f"Plus petit q entier tel que μ^q + ν^q ≤ 1 : **{pq if pq else 'aucun (μ = ν = 1)'}**.")
gr.afficher(gr.plan_mu_nu({"(μ, ν)": (mu, nu)}, {"IFN : μ + ν = 1": qr.frontiere(1), "PFN : μ² + ν² = 1": qr.frontiere(2),
                                                   f"q-ROFN : μ^{q} + ν^{q} = 1": qr.frontiere(q)},
                          "Espaces des nombres IFN, PFN et q-ROFN (figure p. 120)"))
st.caption("Quand q augmente, l'espace admissible s'agrandit vers le carré [0, 1]² (q → ∞).")

st.subheader("q-ROF à valeur d'intervalle (p. 121)")
st.latex(r"\mu=[\mu_L,\mu_U],\ \nu=[\nu_L,\nu_U],\ \mu_U^q+\nu_U^q\le1,\quad \pi_L=\sqrt[q]{1-\mu_U^q-\nu_U^q},\ "
         r"\pi_U=\sqrt[q]{1-\mu_L^q-\nu_L^q}")
c = st.columns(4)
try:
    pl, pu = qr.hesitation_intervalle(c[0].number_input("μL", 0.0, 1.0, 0.3), c[1].number_input("μU", 0.0, 1.0, 0.5),
                                      c[2].number_input("νL", 0.0, 1.0, 0.2), c[3].number_input("νU", 0.0, 1.0, 0.4), q)
    st.success(f"Degré d'hésitation (q = {q}) : π = [{pl:.4f} ; {pu:.4f}].")
except ErreurMethode as exc:
    st.error(str(exc))

# ---------------------------------------------------------------------------
st.header("2. Ensembles flous picturaux (p. 122–124)")
st.markdown("Réponses possibles : **oui** (μ, appartenance positive), **non** (ν, négative), **abstention** (π, neutre) "
            "et **refus** : $1 - (\\mu + \\nu + \\pi)$, avec $\\mu + \\nu + \\pi \\le 1$ (exemple du vote).")
st.latex(r"A\oplus B=(\mu_A+\mu_B-\mu_A\mu_B,\ \nu_A\nu_B,\ \pi_A\pi_B),\quad \lambda A=(1-(1-\mu_A)^\lambda,\ \nu_A^\lambda,\ \pi_A^\lambda),\quad "
         r"A^\lambda=(\mu_A^\lambda,\ 1-(1-\nu_A)^\lambda,\ 1-(1-\pi_A)^\lambda)")
af.encadre_ecart("E-3, E-32", "la p. 124 écrit l'exposant n au lieu de λ (corrigé). De plus, λA et A^λ peuvent produire un "
                              "triplet qui viole μ + ν + π ≤ 1 (ex. A = (0,3 ; 0,3 ; 0,3), λ = 2 pour A^λ) : le résultat "
                              "est alors signalé comme invalide.")
c1, c2, c3 = st.columns(3)
try:
    A = pf.creer(c1.number_input("μ_A", 0.0, 1.0, 0.3), c1.number_input("ν_A", 0.0, 1.0, 0.3), c1.number_input("π_A", 0.0, 1.0, 0.3))
    B = pf.creer(c2.number_input("μ_B", 0.0, 1.0, 0.5), c2.number_input("ν_B", 0.0, 1.0, 0.2), c2.number_input("π_B", 0.0, 1.0, 0.1))
    lam = c3.number_input("λ (> 0)", 0.01, 20.0, 2.0, key="lam_pf")
    res = {"A": A, "B": B, "A ⊕ B": pf.addition(A, B), "λA": pf.scalaire(lam, A), "A^λ": pf.puissance(A, lam)}
    st.dataframe(pd.DataFrame({k: [round(x.mu, 4), round(x.nu, 4), round(x.pi, 4), round(x.refus, 4), "oui" if x.valide else "NON"]
                               for k, x in res.items()}, index=["μ (oui)", "ν (non)", "π (abstention)", "refus", "valide"]))
    invalides = [k for k, x in res.items() if not x.valide]
    if invalides:
        st.warning("Résultat hors de l'ensemble pictural (μ + ν + π > 1) : " + ", ".join(invalides) + " (E-32).")
except ErreurMethode as exc:
    st.error(str(exc))
