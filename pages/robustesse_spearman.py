import numpy as np
import pandas as pd
import streamlit as st

from core.commun import a_des_exaequo, rangs_moyens
from core.robustesse.spearman import matrice_spearman, valeur_critique
from ui import affichage as af
from ui import graphiques as gr
from ui.donnees import charger_tableau, choisir_dataset, editeur_tableau

af.entete("Corrélation de Spearman", "Robustesse et comparaison des méthodes", "77–79",
          "Mesurer l'accord entre deux classements des mêmes n alternatives.")

with st.expander("Théorie (cours p. 77)", expanded=True):
    st.latex(r"\rho = 1 - \frac{6\sum d_i^2}{n(n^2-1)},\qquad d_i = \text{rang}_1(i) - \text{rang}_2(i)")
    st.markdown("ρ = −1 : classements inversés ; ρ = 0 : aucun lien ; ρ = +1 : classements identiques.  \n"
                "Valeur critique (test bilatéral à 5 %) : ρ doit dépasser un seuil qui dépend de n "
                "(0,886 pour n = 6, p. 79). Les seuils affichés sont calculés exactement par énumération des "
                "permutations, pour 4 ≤ n ≤ 10.")

ident = choisir_dataset("spearman", {"Exemple : 6 alternatives, 3 méthodes (p. 78)": "C7_spearman_rangs"})
table = editeur_tableau(f"spearman_{ident}", charger_tableau(ident, "rangs"),
                        titre="Rangs de chaque alternative (1 = la meilleure)", nom_fichier="rangs.csv")
methodes = [c for c in table.columns if c != "alternative"]
try:
    alternatives = [str(a) for a in table["alternative"]]
    R = table[methodes].apply(pd.to_numeric).to_numpy(float)
    if np.isnan(R).any() or len(methodes) < 2 or len(alternatives) < 2:
        raise ValueError
except (ValueError, KeyError):
    st.error("Il faut une colonne « alternative » et au moins deux colonnes de rangs complètes.")
    st.stop()
n = len(alternatives)
exaequo = any(a_des_exaequo(R[:, j]) for j in range(len(methodes)))
if exaequo:
    st.warning("Certains classements contiennent des ex æquo : la formule du cours est appliquée aux rangs moyens "
               "et devient approximative (E-21).")
    R = np.column_stack([rangs_moyens(R[:, j], plus_grand_meilleur=False) for j in range(len(methodes))])

noms, Mrho = matrice_spearman({m: R[:, j] for j, m in enumerate(methodes)})
c = valeur_critique(n)

st.subheader("Calcul détaillé pour une paire de méthodes")
c1, c2 = st.columns(2)
m1 = c1.selectbox("Méthode 1", methodes, index=0)
m2 = c2.selectbox("Méthode 2", methodes, index=1)
r1, r2 = R[:, methodes.index(m1)], R[:, methodes.index(m2)]
d = r1 - r2
st.dataframe(pd.DataFrame({m1: r1, m2: r2, "dᵢ": d, "dᵢ²": d ** 2}, index=alternatives).T)
rho = 1 - 6 * np.sum(d ** 2) / (n * (n ** 2 - 1))
st.latex(rf"\rho = 1 - \frac{{6 \times {np.sum(d ** 2):g}}}{{{n}\,({n}^2 - 1)}} = {rho:.4f}")

st.subheader("Matrice de corrélation entre les méthodes")
gr.afficher(gr.heatmap(Mrho, noms, noms, "ρ de Spearman", divergente=True, zmin=-1, zmax=1, decimales=2))
if c is None:
    st.info(f"n = {n} : aucune valeur critique exacte n'est disponible ici (n < 5 ou n > 10).")
else:
    significatives = [f"{noms[i]} – {noms[j]} ({Mrho[i, j]:.2f})" for i in range(len(noms)) for j in range(i + 1, len(noms))
                      if Mrho[i, j] >= c - 1e-9]
    st.markdown(f"**Valeur critique pour n = {n} (bilatéral, 5 %) : ρ ≥ {c:.3f}.** Paires significatives : "
                + (", ".join(significatives) if significatives else "aucune") + ".")

st.subheader("Observations")
ref = methodes[0]
lignes = []
for k in range(1, 4):
    hauts = [set(np.array(alternatives)[R[:, j] <= k]) for j in range(len(methodes))]
    if all(h == hauts[0] for h in hauts) and len(hauts[0]) == k:
        lignes.append(f"Les {k} premières places sont occupées par les mêmes alternatives dans tous les classements : "
                      f"{', '.join(sorted(hauts[0]))}.")
for j, m in enumerate(methodes[1:], start=1):
    nb = int(np.sum(R[:, j] != R[:, 0]))
    lignes.append(f"{m} change le rang de {nb} alternative{'s' if nb > 1 else ''} sur {n} par rapport à {ref}.")
st.markdown("\n".join(f"- {l}" for l in lignes))
st.caption("Conclusion type (p. 79) : un choix commun à toutes les méthodes est robuste ; les rangs qui diffèrent "
           "demandent une analyse de sensibilité (poids, normalisation).")
