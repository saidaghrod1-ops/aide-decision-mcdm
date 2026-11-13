import numpy as np
import pandas as pd
import streamlit as st

from core.classement.ponderation_simple import waspas
from core.classement.topsis import topsis
from core.classement.vikor import vikor
from core.commun import ErreurMethode, rangs, rangs_moyens
from core.donnees import chemin, dataset, lire_tableau_csv
from core.ponderation.ahp import poids_ahp
from core.ponderation.bwm import bwm
from core.ponderation.critic import poids_critic
from core.ponderation.entropie import poids_entropie
from core.robustesse.spearman import matrice_spearman
from ui import affichage as af
from ui import graphiques as gr
from ui.donnees import charger_matrice, choisir_dataset, editeur_comparaison, editeur_matrice, options_matrices
from ui.etat import poids_publies

af.entete("Chaînage pondération → classement", "Robustesse et comparaison des méthodes", "11–12",
          "Étape 1 du processus (p. 11) : déterminer les poids ; étape 2 : classer les alternatives. "
          "Cette page injecte des poids obtenus par différentes méthodes dans différentes méthodes de classement.")

ident = choisir_dataset("chainage", options_matrices("D2_fournisseurs_5offres"))
M = editeur_matrice(f"chainage_{ident}", charger_matrice(ident), titre="Matrice de décision (les poids saisis ici "
                    "forment la source « Manuels »)", nom_fichier="chainage.csv")
if M is None:
    st.stop()
crit, n = M.criteres, M.n

st.header("1. Poids")
sources: dict[str, np.ndarray] = {}
manuels = M.poids_ou_egaux()
if abs(manuels.sum() - 1) <= 1e-3:
    sources["Manuels"] = manuels
for nom, calcul in {"Entropie": lambda: poids_entropie(M.valeurs, M.sens, "lineaire").w,
                    "CRITIC": lambda: poids_critic(M.valeurs, M.sens).w}.items():
    try:
        sources[nom] = calcul()
    except ErreurMethode as exc:
        st.caption(f"{nom} non applicable : {exc}.")

with st.expander("Poids BWM (vecteurs BO et OW pour ces critères)", expanded=False):
    vecteurs_cours = lire_tableau_csv(chemin(dataset("C2_bwm_fournisseur")["fichiers"]["vecteurs"]))[1]
    bo0 = [float(l[1]) for l in vecteurs_cours] if n == 4 else [1.0] * n
    ow0 = [float(l[2]) for l in vecteurs_cours] if n == 4 else [1.0] * n
    if n == 4:
        st.caption("Par défaut : vecteurs de l'exemple du cours (p. 37), dans l'ordre des critères.")
    table = st.data_editor(pd.DataFrame({"Critère": crit, "BO": bo0, "OW": ow0}), hide_index=True, disabled=["Critère"],
                           key=f"chainage_bwm_{ident}")
    cb, cw = st.columns(2)
    b = cb.selectbox("Meilleur critère (B)", range(n), index=int(np.argmin(table["BO"])), format_func=lambda j: crit[j])
    wst = cw.selectbox("Pire critère (W)", range(n), index=int(np.argmin(table["OW"])), format_func=lambda j: crit[j])
    if b != wst:
        try:
            sources["BWM"] = bwm(table["BO"].to_numpy(float), table["OW"].to_numpy(float), b, wst).w
        except (ErreurMethode, ValueError) as exc:
            st.error(f"BWM : {exc}")
    else:
        st.warning("B et W doivent être différents.")

with st.expander("Poids AHP (matrice de comparaison des critères)", expanded=False):
    if n <= 10:
        res = editeur_comparaison(f"chainage_ahp_{ident}", crit, np.ones((n, n)),
                                  titre="Matrice de comparaison par paires (à compléter)")
        if res is not None:
            p = poids_ahp(res[1])
            if np.allclose(res[1], 1):
                st.caption("Matrice laissée à 1 partout : poids égaux. Saisissez vos jugements pour obtenir des poids AHP.")
            elif not p.coherent:
                st.warning(f"CR = {p.CR:.3f} ≥ 0,1 : matrice non consistante, poids AHP non retenus.")
            else:
                sources["AHP"] = p.w

for nom, pp in poids_publies().items():
    if len(pp.criteres) == n:
        sources[f"Envoyés : {nom}"] = pp.poids

choix = st.multiselect("Sources de poids à comparer", list(sources), default=list(sources))
if not choix:
    st.stop()
st.dataframe(pd.DataFrame({s: np.round(sources[s], 4) for s in choix}, index=crit).T)
gr.afficher(gr.barres_groupees(crit, {s: sources[s] for s in choix}, "Poids selon la méthode de pondération", "wⱼ"))

st.header("2. Classements")
methodes = st.multiselect("Méthodes de classement", ["TOPSIS", "VIKOR", "WASPAS", "WSM", "WPM"],
                          default=["TOPSIS", "VIKOR", "WASPAS"])


def classer(methode: str, w: np.ndarray):
    if methode == "TOPSIS":
        return topsis(M.valeurs, M.sens, w).RC, True
    if methode == "VIKOR":
        return vikor(M.valeurs, M.sens, w, 0.5).Q, False
    if methode == "WASPAS":
        return waspas(M.valeurs, M.sens, w, 0.5).Q, True
    if methode == "WSM":
        return waspas(M.valeurs, M.sens, w, 1.0).Q1, True
    return waspas(M.valeurs, M.sens, w, 0.0).Q2, True


meilleures = pd.DataFrame(index=choix, columns=methodes, dtype=object)
tous_rangs = {}
for s in choix:
    for m in methodes:
        try:
            score, plus_grand = classer(m, sources[s])
        except ErreurMethode as exc:
            meilleures.loc[s, m] = f"non applicable ({exc})"
            continue
        r = rangs(score, plus_grand)
        tous_rangs[f"{s} → {m}"] = (r, rangs_moyens(score, plus_grand))
        meilleures.loc[s, m] = M.alternatives[int(np.argmin(r))]
st.markdown("**Meilleure alternative pour chaque couple (poids, méthode)**")
st.dataframe(meilleures)
if tous_rangs:
    st.markdown("**Rangs complets**")
    st.dataframe(pd.DataFrame({k: v[0] for k, v in tous_rangs.items()}, index=M.alternatives))
    if len(tous_rangs) >= 2:
        noms, R = matrice_spearman({k: v[1] for k, v in tous_rangs.items()})
        gr.afficher(gr.heatmap(R, noms, noms, "ρ de Spearman entre tous les classements", divergente=True,
                               zmin=-1, zmax=1, decimales=2))

    st.header("3. Interprétation")
    valides = meilleures.map(lambda x: x if isinstance(x, str) and not x.startswith("non applicable") else None)
    par_methode = {m: valides[m].dropna().nunique() for m in methodes}
    par_poids = {s: valides.loc[s].dropna().nunique() for s in choix}
    st.markdown("- Nombre de meilleures alternatives différentes **quand on change de pondération** (méthode fixée) : "
                + ", ".join(f"{m} : {k}" for m, k in par_methode.items()) + ".\n"
                "- Nombre de meilleures alternatives différentes **quand on change de méthode** (poids fixés) : "
                + ", ".join(f"{s} : {k}" for s, k in par_poids.items()) + ".")
    if max(par_methode.values(), default=1) > max(par_poids.values(), default=1):
        st.info("Ici, le choix des **poids** influence davantage la décision que le choix de la méthode de classement.")
    elif max(par_methode.values(), default=1) < max(par_poids.values(), default=1):
        st.info("Ici, le choix de la **méthode de classement** influence davantage la décision que les poids.")
    else:
        st.info("Poids et méthode ont ici une influence comparable sur la meilleure alternative.")
