import numpy as np
import pandas as pd
import streamlit as st

from core.commun import a_des_exaequo, rangs_moyens
from core.robustesse.spearman import matrice_spearman, valeur_critique
from ui import affichage as af
from ui import graphiques as gr
from ui.donnees import charger_tableau, choisir_dataset, editeur_tableau

af.entete("Corrélation de Spearman", "Robustesse et comparaison des méthodes",
          "Mesurer l'accord entre deux classements des mêmes n alternatives.")

with st.expander("Théorie", expanded=True):
    st.latex(r"\rho = 1 - \frac{6\sum d_i^2}{n(n^2-1)},\qquad d_i = \text{rang}_1(i) - \text{rang}_2(i)")
    st.markdown("ρ = −1 : classements inversés ; ρ = 0 : aucun lien ; ρ = +1 : classements identiques. "
                "Pour n = 6, seul ρ ≥ 0,886 est significatif à 5 %.")

ident = choisir_dataset("spearman", {"Exemple du cours : 6 alternatives, 3 méthodes": "C7_spearman_rangs"})
table = editeur_tableau(f"spearman_{ident}", charger_tableau(ident, "rangs"), nom_fichier="rangs.csv")
methodes = [c for c in table.columns if c != "alternative"]
try:
    alternatives = [str(a) for a in table["alternative"]]
    R = table[methodes].apply(pd.to_numeric).to_numpy(float)
    if np.isnan(R).any() or len(methodes) < 2 or len(alternatives) < 2:
        raise ValueError
except (ValueError, KeyError):
    st.error("Il faut une colonne « alternative » et au moins deux colonnes de rangs complètes.")
    st.stop()
if any(a_des_exaequo(R[:, j]) for j in range(len(methodes))):
    st.caption("Classements avec ex æquo : la formule est appliquée aux rangs moyens (approximation).")
    R = np.column_stack([rangs_moyens(R[:, j], plus_grand_meilleur=False) for j in range(len(methodes))])
n = len(alternatives)

c1, c2 = st.columns(2)
m1 = c1.selectbox("Méthode 1", methodes, index=0)
m2 = c2.selectbox("Méthode 2", methodes, index=1)
d = R[:, methodes.index(m1)] - R[:, methodes.index(m2)]
st.dataframe(pd.DataFrame({"dᵢ": d, "dᵢ²": d ** 2}, index=alternatives).T)
st.latex(rf"\rho = 1 - \frac{{6 \times {np.sum(d ** 2):g}}}{{{n}\,({n}^2 - 1)}} = {1 - 6 * np.sum(d ** 2) / (n * (n ** 2 - 1)):.4f}")

noms, M = matrice_spearman({m: R[:, j] for j, m in enumerate(methodes)})
gr.afficher(gr.heatmap(M, noms, noms, "Matrice de corrélation entre les méthodes", divergente=True, zmin=-1, zmax=1, decimales=2))
c = valeur_critique(n)
if c is not None:
    sig = [f"{noms[i]} – {noms[j]}" for i in range(len(noms)) for j in range(i + 1, len(noms)) if M[i, j] >= c - 1e-9]
    st.markdown(f"Seuil de significativité (n = {n}, 5 %) : **ρ ≥ {c:.3f}** → paires significatives : "
                + (", ".join(sig) if sig else "aucune") + ".")
hauts = [set(np.array(alternatives)[R[:, j] <= 2]) for j in range(len(methodes))]
if all(h == hauts[0] for h in hauts):
    st.success(f"Les deux premières places ({', '.join(sorted(hauts[0]))}) sont les mêmes pour toutes les méthodes : "
               "ce choix est robuste ; les autres rangs demandent une analyse de sensibilité.")
