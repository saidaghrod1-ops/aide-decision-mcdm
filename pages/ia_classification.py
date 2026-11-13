import numpy as np
import pandas as pd
import streamlit as st

from core.commun import ErreurMethode
from core.ia.classification import CLASSES, entrainer, etiqueter, scores_wsm_reference, seuils_quartiles
from ui import affichage as af
from ui import graphiques as gr
from ui.donnees import charger_matrice, choisir_dataset, editeur_matrice
from ui.ia import choisir_poids

af.entete("De MCDM à la classification supervisée", "MCDM + IA — scénario 1", "91")

with st.expander("Démarche (cours p. 91)", expanded=True):
    st.markdown("""
1. **Pondération par MCDM** (AHP, CRITIC, entropie…).
2. Pour chaque alternative, calculer le score $S_i$ par une **technique MCDM additive** (ici WSM, $S_i = \\sum_j w_j r_{ij}$
   avec la normalisation linéaire du cours, p. 55).
3. **Labéliser** les alternatives à partir de $S_i$ (classes A, B, C, D) : base de données « critères → classe ».
4. **Entraîner un modèle de ML** (Random Forest) sur cette base.
5. **Prédire la classe d'une nouvelle alternative** sans recalculer tout le classement.
""")
    af.encadre_ecart("E-16, E-17", "le cours écrit Sᵢ = Σᵢ wᵢ xᵢ (indice de sommation à corriger, valeurs non normalisées) "
                                    "et ne fixe pas les seuils des classes : Sᵢ = Σⱼ wⱼ rᵢⱼ et classes par quartiles par défaut.")

ident = choisir_dataset("ia_classif", {"Historique de 300 fournisseurs (jeu créé)": "D6_fournisseurs_historique"})
M = editeur_matrice(f"ia_classif_{ident}", charger_matrice(ident), avec_poids=False, nom_fichier="fournisseurs.csv")
if M is None:
    st.stop()

st.header("1–2. Pondération et score MCDM")
w = choisir_poids("ia_classif_poids", M.valeurs, M.sens, M.criteres)
if w is None:
    st.stop()
st.markdown("Poids retenus : " + af.format_poids(M.criteres, w, 3))
try:
    S = scores_wsm_reference(M.valeurs, M.sens, w)
except ErreurMethode as exc:
    st.error(str(exc))
    st.stop()

st.header("3. Étiquettes A, B, C, D")
mode = st.radio("Seuils des classes", ["Quartiles de S (25 % par classe)", "Seuils fixés"], horizontal=True)
if mode.startswith("Quartiles"):
    seuils = seuils_quartiles(S)
else:
    c1, c2, c3 = st.columns(3)
    seuils = (c1.number_input("A si S ≥", value=0.80, step=0.01), c2.number_input("B si S ≥", value=0.70, step=0.01),
              c3.number_input("C si S ≥", value=0.60, step=0.01))
try:
    classes = etiqueter(S, seuils)
except ErreurMethode as exc:
    st.error(str(exc))
    st.stop()
st.caption(f"Seuils : A ≥ {seuils[0]:.4f} ; B ≥ {seuils[1]:.4f} ; C ≥ {seuils[2]:.4f} ; D sinon.")
effectifs = pd.Series(classes).value_counts().reindex(list(CLASSES), fill_value=0)
st.dataframe(pd.DataFrame({"Effectif": effectifs}).T)
gr.afficher(gr.histogrammes({f"Classe {c}": S[classes == c] for c in CLASSES if (classes == c).any()},
                            "Distribution des scores MCDM par classe", "Sᵢ"))
with st.expander("Base de données d'apprentissage (critères → classe)"):
    base = pd.DataFrame(M.valeurs, columns=M.criteres, index=M.alternatives)
    base["S"] = np.round(S, 4)
    base["Classe"] = classes
    st.dataframe(base)

st.header("4. Modèle de ML : Random Forest")
c1, c2, c3 = st.columns(3)
n_arbres = c1.slider("Nombre d'arbres", 10, 300, 150, 10)
profondeur = c2.select_slider("Profondeur maximale", options=["illimitée", 2, 3, 4, 6, 8, 12], value="illimitée")
part_test = c3.slider("Part réservée au test", 0.1, 0.5, 0.25, 0.05)


@st.cache_resource(show_spinner="Apprentissage du modèle…")
def _apprendre(X, etiquettes, n_arbres, profondeur, part_test):
    return entrainer(X, etiquettes, n_arbres, None if profondeur == "illimitée" else int(profondeur), part_test)


try:
    res = _apprendre(M.valeurs, tuple(classes), n_arbres, profondeur, part_test)
except (ErreurMethode, ValueError) as exc:
    st.error(f"Apprentissage impossible : {exc}")
    st.stop()
c1, c2 = st.columns([1, 2])
c1.metric("Exactitude sur le test", f"{res.exactitude:.3f}", f"{res.n_test} alternatives de test", delta_color="off")
with c2:
    gr.afficher(gr.heatmap(res.confusion, [f"prédit {c}" for c in CLASSES], [f"réel {c}" for c in CLASSES],
                           "Matrice de confusion (test)", decimales=0))

st.header("5. Prédire la classe d'une nouvelle alternative")
colonnes = st.columns(M.n)
nouvelle = np.array([col.number_input(c, value=float(np.median(M.valeurs[:, j])), key=f"ia_classif_new_{j}")
                     for j, (col, c) in enumerate(zip(colonnes, M.criteres))])
try:
    s_nouv = float(scores_wsm_reference(nouvelle[None, :], M.sens, w, reference=M.valeurs)[0])
except ErreurMethode as exc:
    st.error(str(exc))
    st.stop()
classe_mcdm = etiqueter([s_nouv], seuils)[0]
classe_ml = res.modele.predict(nouvelle[None, :])[0]
c1, c2 = st.columns(2)
c1.metric("Classe prédite par le Random Forest", classe_ml)
c2.metric("Classe par le calcul MCDM direct", classe_mcdm, f"S = {s_nouv:.4f}", delta_color="off")
(st.success if classe_ml == classe_mcdm else st.warning)(
    "Le modèle reproduit la classe MCDM." if classe_ml == classe_mcdm else
    "Le modèle et le calcul MCDM divergent : l'alternative est proche d'un seuil de classe, ou le modèle généralise mal.")
