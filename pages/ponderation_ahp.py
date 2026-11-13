import numpy as np
import pandas as pd
import streamlit as st

from core.commun import ErreurMethode
from core.donnees import dataset
from core.ponderation.ahp import RI, SEUIL_CR, classement_ahp, poids_ahp
from core.robustesse.sensibilite import balayage_poids, changements_de_premier
from ui import affichage as af
from ui import graphiques as gr
from ui.donnees import charger_comparaison, choisir_dataset, editeur_comparaison
from ui.etat import publier_poids

af.entete("AHP — Analytic Hierarchy Process", "Pondération subjective · classement compensatoire", "23–34")

with st.expander("Théorie (cours p. 23–31)", expanded=False):
    st.markdown("""
L'AHP se base sur les **comparaisons par paires** et le **test de consistance**.
1) Construire une hiérarchie : objectif, critères, alternatives.
2) Comparer chaque critère aux autres avec l'échelle numérique de Saaty, de **1 (importances égales)**
à **9 (extrêmement important)** ; $\\alpha_{ij}$ représente l'importance du critère $C_i$ par rapport au critère $C_j$ :
""")
    st.latex(r"\alpha_{ji} = \frac{1}{\alpha_{ij}}, \qquad AW = \lambda_{max} W")
    st.markdown("Calcul approché du vecteur propre (p. 26–27) : sommes des colonnes, division de chaque élément "
                "par la somme de sa colonne, moyenne de chaque ligne.")
    st.latex(r"s_j=\sum_i \alpha_{ij},\qquad w_i=\frac1n\sum_j \frac{\alpha_{ij}}{s_j}")
    st.markdown("Test de consistance (p. 28–29) :")
    st.latex(r"\lambda_{max}=\sum_i s_i\, w_i,\qquad CI=\frac{\lambda_{max}-n}{n-1},\qquad CR=\frac{CI}{RI},\qquad CR<0{,}1")
    st.dataframe(pd.DataFrame([RI], index=["RI"]).rename(columns=lambda n: f"n = {n}"))
    st.markdown("3) Priorités **locales** des alternatives : une matrice de comparaison par critère, même calcul.  \n"
                "4) Priorités **finales** : $\\sum_{j} w_j\\, w_{ji}$, où $w_{ji}$ est la priorité de l'alternative "
                "$A_i$ selon le critère $C_j$.  \n5) **Analyse de sensibilité** : le classement peut dépendre fortement des poids.")
    af.encadre_ecart("E-11", "la table du cours donne RI(10) = 1,56 ; la table usuelle de Saaty donne 1,49. "
                             "La valeur du cours est utilisée.")
    af.encadre_ecart("E-12", "RI n'est pas défini pour n ≤ 2 : une matrice 2 × 2 réciproque est toujours cohérente "
                             "(CI = 0), le CR est alors affiché « sans objet ».")

ident = choisir_dataset("ahp", {"Exercice : acheter une voiture (p. 32–34)": "C1_ahp_voiture",
                                "Implanter un hôpital parmi 4 villes (jeu créé)": "D3_ahp_hopital"})
d = dataset(ident)
fichier = d["fichiers"]["criteres"]
variante = ""
if "criteres_corrigee" in d["fichiers"]:
    choix = st.radio("Matrice des critères", ["Version incohérente (CR > 0,1)", "Version corrigée"], horizontal=True)
    if choix == "Version corrigée":
        fichier, variante = d["fichiers"]["criteres_corrigee"], "_corrigee"

# ---------------------------------------------------------------------------
st.header("1. Poids des critères")
etiquettes, A0 = charger_comparaison(fichier)
res = editeur_comparaison(f"ahp_{ident}{variante}", etiquettes, A0, titre="Matrice de comparaison des critères",
                          nom_fichier="ahp_criteres.csv")
if res is None:
    st.stop()
criteres, A = res
try:
    p = poids_ahp(A)
except ErreurMethode as exc:
    st.error(str(exc))
    st.stop()

with af.etape("Étape a — Somme de chaque colonne"):
    af.vecteur(p.sommes, criteres, "sⱼ = Σᵢ αᵢⱼ")
with af.etape("Étape b — Diviser chaque élément par la somme de sa colonne"):
    af.matrice(p.N, criteres, criteres)
with af.etape("Étape c — Vecteur propre normalisé : moyenne de chaque ligne", ouverte=True):
    af.vecteur(p.w, criteres, "wᵢ")
with af.etape("Étape d — Test de consistance", ouverte=True):
    st.latex(r"\lambda_{max} = " + " + ".join(f"{s:.3f}\\times{w:.3f}" for s, w in zip(p.sommes, p.w))
             + f" = {p.lambda_max:.4f}")
    if p.CR is None:
        st.info("n ≤ 2 : la matrice est réciproque, donc toujours cohérente (CR sans objet, E-12).")
    else:
        st.latex(rf"CI = \frac{{{p.lambda_max:.4f} - {p.n}}}{{{p.n} - 1}} = {p.CI:.4f}, \quad "
                 rf"RI({p.n}) = {p.RI}, \quad CR = \frac{{{p.CI:.4f}}}{{{p.RI}}} = {p.CR:.4f}")
c1, c2, c3 = st.columns(3)
c1.metric("λmax", f"{p.lambda_max:.4f}")
c2.metric("CI", "—" if p.CI is None else f"{p.CI:.4f}")
c3.metric("CR", "—" if p.CR is None else f"{p.CR:.4f}")
if p.coherent:
    st.success(f"CR < {SEUIL_CR} : la matrice est **consistante**, les poids peuvent être conservés (p. 29).")
else:
    st.error(f"CR ≥ {SEUIL_CR} : la matrice **n'est pas consistante**. Le décideur doit revoir ses jugements "
             "avant de poursuivre (p. 29).")
gr.afficher(gr.barres(criteres, p.w, "Poids des critères (AHP)", "wⱼ"))
with st.expander("Comparaison avec le vecteur propre exact (AW = λmax W, p. 25)"):
    st.dataframe(pd.DataFrame({"Méthode approchée (cours)": np.round(p.w, 4), "Vecteur propre exact": np.round(p.w_exact, 4)},
                              index=criteres))
    st.caption(f"λmax approché = {p.lambda_max:.4f} ; valeur propre exacte = {p.lambda_exact:.4f}.")
if st.button("Utiliser ces poids dans la page « Chaînage »", icon=":material/link:"):
    publier_poids(f"AHP — {d['titre']}", criteres, p.w)
    st.toast("Poids AHP envoyés vers la page de chaînage.")

# ---------------------------------------------------------------------------
st.header("2. Classement des alternatives (étapes 3 et 4)")
fichiers_alt = d["fichiers"]["alternatives"]
alternatives_ref, _ = charger_comparaison(next(iter(fichiers_alt.values())))
onglets = st.tabs(list(criteres))
locales = []
for onglet, critere in zip(onglets, criteres):
    with onglet:
        if critere in fichiers_alt:
            noms, B0 = charger_comparaison(fichiers_alt[critere])
        else:
            noms, B0 = alternatives_ref, np.ones((len(alternatives_ref),) * 2)
        r = editeur_comparaison(f"ahp_{ident}_alt_{critere}", noms, B0,
                                titre=f"Comparaison des alternatives selon « {critere} »",
                                nom_fichier=f"ahp_{critere}.csv")
        if r is None:
            st.stop()
        locales.append(r)
noms_alt = locales[0][0]
if any(l[0] != noms_alt for l in locales):
    st.error("Les matrices des alternatives doivent porter sur les mêmes alternatives, dans le même ordre.")
    st.stop()
try:
    cl = classement_ahp(A, [l[1] for l in locales])
except ErreurMethode as exc:
    st.error(str(exc))
    st.stop()

with af.etape("Étape 3 — Priorités locales (et cohérence de chaque matrice)", ouverte=True):
    af.matrice(cl.W_local, noms_alt, criteres, legende="w_ji : priorité de l'alternative Aᵢ selon le critère Cⱼ (E-13).")
    st.dataframe(pd.DataFrame({"CR": ["sans objet (n ≤ 2)" if l.CR is None else f"{l.CR:.4f}" for l in cl.locales],
                               "Cohérente": ["oui" if l.coherent else "non" for l in cl.locales]}, index=criteres).T)
    if not all(l.coherent for l in cl.locales):
        st.warning("Au moins une matrice d'alternatives n'est pas consistante (CR ≥ 0,1).")
with af.etape("Étape 4 — Priorités finales Σⱼ wⱼ wⱼᵢ", ouverte=True):
    tableau = pd.DataFrame(np.round(cl.W_local, 4), index=noms_alt, columns=criteres)
    tableau.loc["Poids des critères"] = np.round(p.w, 4)
    tableau["Priorités finales"] = list(np.round(cl.finales, 4)) + [np.nan]
    st.dataframe(tableau)
af.classement(noms_alt, {"Priorité finale": cl.finales}, cl.rangs)
gr.afficher(gr.barres_empilees(noms_alt, {c: cl.contributions[:, j] for j, c in enumerate(criteres)},
                               "Contribution de chaque critère à la priorité finale", "wⱼ · wⱼᵢ"))
meilleure = noms_alt[int(np.argmax(cl.finales))]
st.success(f"**{meilleure}** obtient la priorité finale la plus élevée ({cl.finales.max():.4f}) : c'est la meilleure alternative.")

# ---------------------------------------------------------------------------
st.header("3. Analyse de sensibilité (étape 5)")
st.caption("On fait varier le poids d'un critère de 0 à 1 ; les autres poids sont réajustés proportionnellement "
           "pour que leur somme reste égale à 1.")
k = st.selectbox("Critère dont on fait varier le poids", range(len(criteres)), format_func=lambda j: criteres[j])
ts, S, _ = balayage_poids(p.w, k, lambda w: cl.W_local @ w)
gr.afficher(gr.courbes(ts, {a: S[:, i] for i, a in enumerate(noms_alt)}, f"Priorités finales selon le poids de « {criteres[k]} »",
                       f"poids de {criteres[k]}", "priorité finale", marqueur_x=float(p.w[k])))
changements = changements_de_premier(ts, np.array([np.argsort(np.argsort(-s)) + 1 for s in S]))
if changements:
    st.info("La meilleure alternative change : " + " ; ".join(
        f"vers w = {t:.2f}, {noms_alt[a]} → {noms_alt[b]}" for t, a, b in changements))
else:
    st.info("La meilleure alternative reste la même sur tout l'intervalle : le choix est robuste à ce poids.")
