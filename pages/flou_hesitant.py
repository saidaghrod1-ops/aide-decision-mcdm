import pandas as pd
import streamlit as st

from core.commun import ErreurMethode
from core.flou import hesitant as hf
from ui import affichage as af
from ui.donnees import charger_tableau, editeur_tableau

af.entete("Ensembles flous hésitants", "Logique floue",
          "Quand plusieurs décideurs ne s'accordent pas, le degré d'appartenance devient un ensemble de valeurs "
          "h = {γ₁, …, γₖ} (Torra, 2010).")

with st.expander("Théorie", expanded=True):
    st.latex(r"s(h)=\frac1l\sum_{\gamma\in h}\gamma,\qquad \sigma(h)=\Big[\frac1l\sum_{\gamma\in h}(\gamma-s(h))^2\Big]^{1/2}")
    st.markdown("Comparaison : le plus grand score l'emporte ; à score égal, la **plus petite déviation** l'emporte.")
    st.latex(r"h_1\oplus h_2=\bigcup\{\gamma_1+\gamma_2-\gamma_1\gamma_2\},\quad h_1\otimes h_2=\bigcup\{\gamma_1\gamma_2\},\quad "
             r"\lVert h_1-h_2\rVert=\frac1l\sum_k\lvert h_1^{\sigma(k)}-h_2^{\sigma(k)}\rvert")

st.caption("Exemple du cours : comparer h1, h2 et h3 (valeurs séparées par « ; »).")
table = editeur_tableau("hesitant_table", charger_tableau("C11_hesitants_p129", "tableau"),
                        colonnes_attendues=["nom", "valeurs"], nom_fichier="hesitants.csv")
try:
    noms = [str(n) for n in table["nom"]]
    H = [hf.lire(v) for v in table["valeurs"]]
except ErreurMethode as exc:
    st.error(str(exc))
    st.stop()
if not H:
    st.stop()
st.dataframe(pd.DataFrame({"h": noms, "s(h)": [round(hf.score(h), 4) for h in H],
                           "σ(h)": [round(hf.deviation(h), 4) for h in H]}), hide_index=True)
ordre = hf.classer(H)
texte = noms[ordre[0]]
for i, j in zip(ordre, ordre[1:]):
    texte += (" = " if hf.comparer(H[i], H[j]) == 0 else " > ") + noms[j]
st.success(f"Classement : **{texte}**.")

st.subheader("Opérations et distance")
c1, c2 = st.columns(2)
i1 = c1.selectbox("h₁", range(len(H)), format_func=lambda i: noms[i])
i2 = c2.selectbox("h₂", range(len(H)), index=min(1, len(H) - 1), format_func=lambda i: noms[i])
fmt = lambda h: "{" + "; ".join(f"{g:.4g}" for g in h) + "}"
st.markdown(f"h₁ ⊕ h₂ = {fmt(hf.somme(H[i1], H[i2]))}  \nh₁ ⊗ h₂ = {fmt(hf.produit(H[i1], H[i2]))}  \n"
            f"Distance de Hamming ‖h₁ − h₂‖ = **{hf.distance_hamming(H[i1], H[i2]):.4f}**")
st.caption("Pour la distance, l'ensemble le plus court est complété par sa plus petite valeur.")
