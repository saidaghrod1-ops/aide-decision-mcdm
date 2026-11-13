import numpy as np
import pandas as pd
import streamlit as st

from core.commun import ErreurMethode
from core.flou import intuitionniste as ifs
from ui import affichage as af
from ui.donnees import editeur_tableau

af.entete("Ensembles flous intuitionnistes", "Logique floue",
          "Atanassov (1986) ajoute un degré de non-appartenance ν et un degré d'hésitation π.")

with st.expander("Théorie", expanded=True):
    st.latex(r"\alpha=(\mu,\nu),\quad \mu+\nu\le1,\quad \pi=1-\mu-\nu")
    st.latex(r"\alpha_1+\alpha_2=(\mu_1+\mu_2-\mu_1\mu_2,\ \nu_1\nu_2),\quad \alpha_1\times\alpha_2=(\mu_1\mu_2,\ \nu_1+\nu_2-\nu_1\nu_2),\quad "
             r"\lambda\alpha=(1-(1-\mu)^\lambda,\ \nu^\lambda)")
    st.latex(r"IFWA_\sigma(\alpha_1,\dots,\alpha_k)=\Big(1-\prod_j(1-\mu_j)^{\sigma_j},\ \prod_j\nu_j^{\sigma_j}\Big),\qquad "
             r"S(\alpha)=\mu-\nu,\quad H(\alpha)=\mu+\nu")
    st.markdown("Comparaison : le plus grand score $S$ l'emporte ; à score égal, la plus grande précision $H$.")

st.header("Exemple du vote")
c1, c2, c3 = st.columns(3)
try:
    a = ifs.depuis_vote(c1.number_input("Pour", 0, 1000, 5), c2.number_input("Contre", 0, 1000, 3),
                        c3.number_input("Absents", 0, 1000, 2))
    st.markdown(f"μ = {a.mu:.2f}, ν = {a.nu:.2f}, **π = {a.pi:.2f}** (degré d'hésitation).")
except ErreurMethode as exc:
    st.error(str(exc))

st.header("Agrégation IFWA et classement")
table = editeur_tableau("ifs_table", pd.DataFrame({"nom": ["α1", "α2", "α3", "α4"], "mu": [0.6, 0.5, 0.4, 0.3],
                                                   "nu": [0.3, 0.2, 0.2, 0.1], "sigma": [0.4, 0.3, 0.2, 0.1]}),
                        nom_fichier="nombres_intuitionnistes.csv")
try:
    nombres = [ifs.NFI(float(m), float(n)) for m, n in zip(table["mu"], table["nu"])]
    noms = [str(x) for x in table["nom"]]
    agr = ifs.ifwa(nombres, pd.to_numeric(table["sigma"]).to_numpy(float))
    ordre = ifs.classer(nombres)
    st.dataframe(pd.DataFrame({"Nombre": [noms[i] for i in ordre], "π": [round(nombres[i].pi, 4) for i in ordre],
                               "Score S": [round(ifs.score(nombres[i]), 4) for i in ordre],
                               "Précision H": [round(ifs.precision(nombres[i]), 4) for i in ordre]}), hide_index=True)
    st.success(f"IFWA = (μ = {agr.mu:.4f} ; ν = {agr.nu:.4f}) ; classement : {' > '.join(noms[i] for i in ordre)}.")
except (ErreurMethode, ValueError) as exc:
    st.error(str(exc))
