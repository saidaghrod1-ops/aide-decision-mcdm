import json

import numpy as np
import pandas as pd
import streamlit as st

from core.commun import ErreurMethode
from core.donnees import chemin, dataset
from core.flou import intuitionniste as ifs
from core.flou.orthopair import frontiere
from ui import affichage as af
from ui import graphiques as gr
from ui.donnees import editeur_tableau

af.entete("Ensembles flous intuitionnistes", "Logique floue", "112–117",
          "Atanassov (1986) ajoute un degré de non-appartenance ν et un degré d'hésitation π.")

with chemin(dataset("C14_ensembles_flous_exemples")["fichiers"]["json"]).open(encoding="utf-8") as f:
    EX = json.load(f)

st.header("1. Définition et exemple du vote (p. 112–114)")
st.latex(r"A=\{\langle x,\mu_A(x),\nu_A(x)\rangle\mid x\in X\},\quad 0\le\mu_A(x)+\nu_A(x)\le1,\quad \pi_A(x)=1-\mu_A(x)-\nu_A(x)")
v = EX["vote_intuitionniste"]
c1, c2, c3 = st.columns(3)
pour = c1.number_input("Pour", 0, 1000, v["pour"]); contre = c2.number_input("Contre", 0, 1000, v["contre"])
absents = c3.number_input("Absents", 0, 1000, v["absents"])
try:
    a = ifs.depuis_vote(pour, contre, absents)
    st.markdown(f"μ = {a.mu:.3f} (pour), ν = {a.nu:.3f} (contre), **π = {a.pi:.3f}** (ni pour ni contre). "
                "Plus π est élevé, plus l'incertitude est grande.")
except ErreurMethode as exc:
    st.error(str(exc))

st.header("2. Opérations (p. 115)")
st.latex(r"\begin{aligned}\alpha_1+\alpha_2&=(\mu_1+\mu_2-\mu_1\mu_2,\ \nu_1\nu_2) & \alpha_1\times\alpha_2&=(\mu_1\mu_2,\ \nu_1+\nu_2-\nu_1\nu_2)\\"
         r"\lambda\alpha&=(1-(1-\mu)^\lambda,\ \nu^\lambda) & \alpha^\lambda&=(\mu^\lambda,\ 1-(1-\nu)^\lambda)\end{aligned}")
c1, c2, c3 = st.columns(3)
try:
    a1 = ifs.NFI(c1.number_input("μ₁", 0.0, 1.0, EX["nombre_intuitionniste"]["mu"], 0.05),
                 c1.number_input("ν₁", 0.0, 1.0, EX["nombre_intuitionniste"]["nu"], 0.05))
    a2 = ifs.NFI(c2.number_input("μ₂", 0.0, 1.0, 0.5, 0.05), c2.number_input("ν₂", 0.0, 1.0, 0.4, 0.05))
    lam = c3.number_input("λ (> 0)", 0.01, 20.0, 2.0)
    res = {"α₁": a1, "α₂": a2, "α₁ + α₂": ifs.addition(a1, a2), "α₁ × α₂": ifs.multiplication(a1, a2),
           "λα₁": ifs.scalaire(lam, a1), "α₁^λ": ifs.puissance(a1, lam)}
    st.dataframe(pd.DataFrame({k: [round(x.mu, 4), round(x.nu, 4), round(x.pi, 4), round(ifs.score(x), 4), round(ifs.precision(x), 4)]
                               for k, x in res.items()}, index=["μ", "ν", "π", "Score S = μ − ν", "Précision H = μ + ν"]))
    gr.afficher(gr.plan_mu_nu({k: (x.mu, x.nu) for k, x in res.items()}, {"μ + ν = 1": frontiere(1)}, "Nombres dans le plan (μ, ν)"))
except ErreurMethode as exc:
    st.error(str(exc))

st.header("3. Agrégation IFWA et classement (p. 116–117)")
st.latex(r"IFWA_\sigma(\alpha_1,\dots,\alpha_k)=\Big(1-\prod_{j=1}^{k}(1-\mu_j)^{\sigma_j},\ \prod_{j=1}^{k}\nu_j^{\sigma_j}\Big)")
st.markdown("Comparaison : si $S(\\alpha_i) > S(\\alpha_j)$ alors $\\alpha_i > \\alpha_j$ ; si les scores sont égaux, "
            "la plus grande précision $H$ l'emporte ; à score et précision égaux, $\\alpha_i = \\alpha_j$.")
table = editeur_tableau("ifs_table", pd.DataFrame({"nom": ["α1", "α2", "α3", "α4"], "mu": [0.6, 0.5, 0.4, 0.3],
                                                   "nu": [0.3, 0.2, 0.2, 0.1], "sigma": [0.4, 0.3, 0.2, 0.1]}),
                        titre="Nombres intuitionnistes et poids σ", nom_fichier="nombres_intuitionnistes.csv")
try:
    nombres = [ifs.NFI(float(m), float(n)) for m, n in zip(table["mu"], table["nu"])]
    noms = [str(x) for x in table["nom"]]
    agr = ifs.ifwa(nombres, pd.to_numeric(table["sigma"]).to_numpy(float))
    st.success(f"IFWA = (μ = {agr.mu:.4f} ; ν = {agr.nu:.4f}), π = {agr.pi:.4f}.")
    ordre = ifs.classer(nombres)
    st.dataframe(pd.DataFrame({"Nombre": [noms[i] for i in ordre], "Score S": [round(ifs.score(nombres[i]), 4) for i in ordre],
                               "Précision H": [round(ifs.precision(nombres[i]), 4) for i in ordre]}), hide_index=True)
    st.caption("Classement du plus grand au plus petit selon la fonction score, puis la fonction de précision.")
except (ErreurMethode, ValueError) as exc:
    st.error(str(exc))
