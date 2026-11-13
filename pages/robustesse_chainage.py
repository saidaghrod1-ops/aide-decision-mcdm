import numpy as np
import pandas as pd
import streamlit as st

from core.classement.ponderation_simple import waspas
from core.classement.topsis import topsis
from core.classement.vikor import vikor
from core.commun import ErreurMethode, rangs
from core.donnees import chemin, dataset, lire_tableau_csv
from core.ponderation.bwm import bwm
from core.ponderation.critic import poids_critic
from core.ponderation.entropie import poids_entropie
from ui import affichage as af
from ui import graphiques as gr
from ui.donnees import charger_matrice, choisir_dataset, editeur_matrice, options_matrices
from ui.etat import poids_publies

af.entete("Chaînage pondération → classement", "Robustesse et comparaison des méthodes",
          "Étape 1 : déterminer les poids ; étape 2 : classer. On injecte des poids obtenus par plusieurs méthodes "
          "dans plusieurs méthodes de classement.")

ident = choisir_dataset("chainage", options_matrices("D2_fournisseurs_5offres", "D1_selection_modele_ml",
                                                     "C5_topsis_voitures"))
M = editeur_matrice(f"chainage_{ident}", charger_matrice(ident), nom_fichier="chainage.csv")
if M is None:
    st.stop()

sources: dict[str, np.ndarray] = {}
if abs(M.poids_ou_egaux().sum() - 1) <= 1e-3:
    sources["Saisis"] = M.poids_ou_egaux()
for nom, f in {"Entropie": lambda: poids_entropie(M.valeurs, M.sens, "lineaire").w,
               "CRITIC": lambda: poids_critic(M.valeurs, M.sens).w}.items():
    try:
        sources[nom] = f()
    except ErreurMethode:
        pass
if M.n == 4:   # vecteurs BO/OW de l'exemple du cours, dans l'ordre des critères
    v = lire_tableau_csv(chemin(dataset("C2_bwm_fournisseur")["fichiers"]["vecteurs"]))[1]
    sources["BWM (vecteurs de l'exemple du cours)"] = bwm([float(l[1]) for l in v], [float(l[2]) for l in v], 0, 3).w
for nom, pp in poids_publies().items():
    if len(pp.criteres) == M.n:
        sources[nom] = pp.poids
st.caption("Les poids calculés dans les pages AHP, BWM, entropie ou CRITIC (bouton « Utiliser ces poids ») apparaissent ici.")
choix = st.multiselect("Poids", list(sources), default=list(sources))
methodes = st.multiselect("Méthodes de classement", ["TOPSIS", "VIKOR", "WASPAS", "WSM", "WPM"], default=["TOPSIS", "VIKOR", "WASPAS"])
if not choix or not methodes:
    st.stop()
st.dataframe(pd.DataFrame({s: np.round(sources[s], 4) for s in choix}, index=M.criteres).T)
gr.afficher(gr.barres_groupees(M.criteres, {s: sources[s] for s in choix}, "Poids selon la méthode de pondération", "wⱼ"))


def meilleure(methode: str, w: np.ndarray) -> str:
    if methode == "VIKOR":
        s, plus_grand = vikor(M.valeurs, M.sens, w).Q, False
    elif methode == "TOPSIS":
        s, plus_grand = topsis(M.valeurs, M.sens, w).RC, True
    else:
        r = waspas(M.valeurs, M.sens, w, {"WASPAS": 0.5, "WSM": 1.0, "WPM": 0.0}[methode])
        s, plus_grand = r.Q, True
    return M.alternatives[int(np.argmin(rangs(s, plus_grand)))]


grille = pd.DataFrame(index=choix, columns=methodes, dtype=object)
for s in choix:
    for m in methodes:
        try:
            grille.loc[s, m] = meilleure(m, sources[s])
        except ErreurMethode:
            grille.loc[s, m] = "—"
st.markdown("**Meilleure alternative pour chaque couple (poids, méthode)**")
st.dataframe(grille)
distinctes = set(grille.values.ravel()) - {"—"}
(st.success if len(distinctes) == 1 else st.warning)(
    f"{len(distinctes)} meilleure(s) alternative(s) différente(s) : " + ", ".join(sorted(distinctes))
    + (". Le choix est robuste." if len(distinctes) == 1 else ". Le choix dépend des poids et/ou de la méthode."))
