import numpy as np
import pandas as pd
import streamlit as st

from core.classement.ponderation_simple import waspas
from core.classement.topsis import topsis
from core.classement.vikor import vikor
from core.commun import ErreurMethode, rangs, rangs_moyens
from core.robustesse.spearman import matrice_spearman, valeur_critique
from ui import affichage as af
from ui import graphiques as gr
from ui.donnees import charger_matrice, choisir_dataset, editeur_matrice, options_matrices, poids_valides

af.entete("Comparaison multi-méthodes", "Robustesse et comparaison des méthodes", "12, 76–79",
          "Bonne pratique du cours (p. 12) : appliquer plusieurs méthodes au même problème et comparer les classements.")

ident = choisir_dataset("comparaison", options_matrices("D1_selection_modele_ml"))
M = editeur_matrice(f"comparaison_{ident}", charger_matrice(ident), nom_fichier="comparaison.csv")
if M is None:
    st.stop()
w = poids_valides(M, f"comparaison_{ident}")
if w is None:
    st.stop()
c1, c2 = st.columns(2)
lam = c1.slider("λ (WASPAS)", 0.0, 1.0, 0.5, 0.05)
v = c2.slider("v (VIKOR)", 0.0, 1.0, 0.5, 0.05)

scores, rangs_methodes, refus = {}, {}, {}
calculs = {
    "WSM": lambda: (waspas(M.valeurs, M.sens, w, 1.0).Q1, True),
    "WPM": lambda: (waspas(M.valeurs, M.sens, w, 0.0).Q2, True),
    f"WASPAS (λ = {lam:.2f})": lambda: (waspas(M.valeurs, M.sens, w, lam).Q, True),
    "TOPSIS": lambda: (topsis(M.valeurs, M.sens, w).RC, True),
    f"VIKOR (Q, v = {v:.2f})": lambda: (vikor(M.valeurs, M.sens, w, v).Q, False),
}
for nom, calcul in calculs.items():
    try:
        s, plus_grand = calcul()
    except ErreurMethode as exc:
        refus[nom] = str(exc)
        continue
    scores[nom] = s
    rangs_methodes[nom] = (rangs(s, plus_grand), rangs_moyens(s, plus_grand))
for nom, raison in refus.items():
    st.warning(f"**{nom}** ne s'applique pas à ces données : {raison}.")
if len(rangs_methodes) < 2:
    st.error("Il faut au moins deux méthodes applicables pour comparer.")
    st.stop()

alt = M.alternatives
st.subheader("Rangs par méthode (1 = meilleure)")
table = pd.DataFrame({m: r[0] for m, r in rangs_methodes.items()}, index=alt)
st.dataframe(table)
gr.afficher(gr.rangs_par_methode(alt, {m: r[0] for m, r in rangs_methodes.items()}, "Rang de chaque alternative selon la méthode"))

st.subheader("Accord entre les méthodes (Spearman)")
noms, R = matrice_spearman({m: r[1] for m, r in rangs_methodes.items()})
gr.afficher(gr.heatmap(R, noms, noms, "ρ de Spearman entre classements", divergente=True, zmin=-1, zmax=1, decimales=2))
if any(len(np.unique(np.round(s, 10))) < len(s) for s in scores.values()):
    st.caption("Des ex æquo existent : ρ est calculé sur les rangs moyens (approximation, E-21).")
c = valeur_critique(M.m)
st.markdown(f"Valeur critique (n = {M.m}, bilatéral 5 %) : " + ("non disponible pour ce n." if c is None else f"ρ ≥ {c:.3f}."))

meilleures = {m: alt[int(np.argmin(r[0]))] for m, r in rangs_methodes.items()}
st.subheader("Interprétation")
distinctes = sorted(set(meilleures.values()))
if len(distinctes) == 1:
    st.success(f"Toutes les méthodes désignent **{distinctes[0]}** : ce choix est robuste au changement de méthode.")
else:
    st.warning("Les méthodes ne désignent pas la même meilleure alternative : "
               + " ; ".join(f"{m} → {a}" for m, a in meilleures.items())
               + ". Le choix dépend de la méthode : il faut le justifier et mener une analyse de sensibilité (p. 12).")
moy = R[np.triu_indices(len(noms), 1)].mean()
st.markdown(f"Corrélation moyenne entre méthodes : **{moy:.2f}**.")
