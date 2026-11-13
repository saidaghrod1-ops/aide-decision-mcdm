import numpy as np
import pandas as pd
import streamlit as st

from core.commun import ErreurMethode
from core.donnees import dataset
from core.ponderation.ahp import RI, SEUIL_CR, classement_ahp, poids_ahp
from core.robustesse.sensibilite import balayage_poids
from ui import affichage as af
from ui import graphiques as gr
from ui.donnees import charger_comparaison, choisir_dataset, editeur_comparaison
from ui.etat import publier_poids

af.entete("AHP — Analytic Hierarchy Process", "Pondération subjective · classement")

with st.expander("Théorie", expanded=True):
    st.markdown("On compare les critères deux à deux sur l'échelle de Saaty, de **1** (importances égales) à "
                "**9** (extrêmement important) ; $\\alpha_{ij}$ est l'importance de $C_i$ par rapport à $C_j$.")
    st.latex(r"\alpha_{ji}=\frac{1}{\alpha_{ij}},\qquad s_j=\sum_i \alpha_{ij},\qquad w_i=\frac1n\sum_j \frac{\alpha_{ij}}{s_j}")
    st.latex(r"\lambda_{max}=\sum_i s_i\, w_i,\qquad CI=\frac{\lambda_{max}-n}{n-1},\qquad CR=\frac{CI}{RI}<0{,}1")
    st.dataframe(pd.DataFrame([RI], index=["RI"]).rename(columns=lambda n: f"n = {n}"))
    st.markdown("Classement : priorités locales $w_{ji}$ des alternatives pour chaque critère (même calcul), puis "
                "priorités finales $\\sum_j w_j\\, w_{ji}$.")

ident = choisir_dataset("ahp", {"Exercice du cours : acheter une voiture": "C1_ahp_voiture",
                                "Implanter un hôpital parmi 4 villes": "D3_ahp_hopital"})
d = dataset(ident)
fichier, variante = d["fichiers"]["criteres"], ""
if "criteres_corrigee" in d["fichiers"] and st.toggle("Utiliser la matrice corrigée (la matrice initiale est incohérente)"):
    fichier, variante = d["fichiers"]["criteres_corrigee"], "_corrigee"

st.header("1. Poids des critères")
etiquettes, A0 = charger_comparaison(fichier)
res = editeur_comparaison(f"ahp_{ident}{variante}", etiquettes, A0, nom_fichier="ahp_criteres.csv")
if res is None:
    st.stop()
criteres, A = res
try:
    p = poids_ahp(A)
except ErreurMethode as exc:
    st.error(str(exc))
    st.stop()
with af.etape("Calcul : sommes des colonnes, matrice normalisée, poids"):
    af.vecteur(p.sommes, criteres, "sⱼ")
    af.matrice(p.N, criteres, criteres)
af.vecteur(p.w, criteres, "Poids wᵢ")
c1, c2, c3 = st.columns(3)
c1.metric("λmax", f"{p.lambda_max:.4f}")
c2.metric("CI", "—" if p.CI is None else f"{p.CI:.4f}")
c3.metric("CR", "—" if p.CR is None else f"{p.CR:.4f}")
if p.coherent:
    st.success(f"CR < {SEUIL_CR} : matrice consistante, les poids peuvent être conservés."
               + (" (Pour n ≤ 2, une matrice réciproque est toujours cohérente.)" if p.CR is None else ""))
else:
    st.warning(f"CR ≥ {SEUIL_CR} : matrice non consistante, le décideur doit revoir ses jugements.")
if st.button("Utiliser ces poids dans la page « Chaînage »", icon=":material/link:"):
    publier_poids(f"AHP — {d['titre']}", criteres, p.w)
    st.toast("Poids AHP envoyés vers la page de chaînage.")

st.header("2. Classement des alternatives")
fichiers_alt = d["fichiers"]["alternatives"]
alternatives_ref, _ = charger_comparaison(next(iter(fichiers_alt.values())))
locales = []
for onglet, critere in zip(st.tabs(list(criteres)), criteres):
    with onglet:
        noms, B0 = (charger_comparaison(fichiers_alt[critere]) if critere in fichiers_alt
                    else (alternatives_ref, np.ones((len(alternatives_ref),) * 2)))
        r = editeur_comparaison(f"ahp_{ident}_alt_{critere}", noms, B0, nom_fichier=f"ahp_{critere}.csv")
        if r is None:
            st.stop()
        locales.append(r)
noms_alt = locales[0][0]
if any(l[0] != noms_alt for l in locales):
    st.error("Les matrices des alternatives doivent porter sur les mêmes alternatives.")
    st.stop()
try:
    cl = classement_ahp(A, [l[1] for l in locales])
except ErreurMethode as exc:
    st.error(str(exc))
    st.stop()
tableau = pd.DataFrame(np.round(cl.W_local, 4), index=noms_alt, columns=criteres)
tableau.loc["Poids des critères"] = np.round(p.w, 4)
tableau["Priorité finale"] = list(np.round(cl.finales, 4)) + [np.nan]
st.dataframe(tableau)
gr.afficher(gr.barres(noms_alt, cl.finales, "Priorités finales", "Σⱼ wⱼ wⱼᵢ"))
st.success(f"Meilleure alternative : **{noms_alt[int(np.argmax(cl.finales))]}** ({cl.finales.max():.4f}).")

st.header("3. Analyse de sensibilité")
k = st.selectbox("Faire varier le poids du critère", range(len(criteres)), format_func=lambda j: criteres[j])
ts, S, _ = balayage_poids(p.w, k, lambda w: cl.W_local @ w)
gr.afficher(gr.courbes(ts, {a: S[:, i] for i, a in enumerate(noms_alt)}, None, f"poids de {criteres[k]}",
                       "priorité finale", marqueur_x=float(p.w[k])))
st.caption("Les autres poids sont réajustés proportionnellement pour garder Σ wⱼ = 1.")
