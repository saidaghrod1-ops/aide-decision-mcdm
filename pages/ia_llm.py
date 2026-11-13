import numpy as np
import pandas as pd
import streamlit as st

from core.classement.ponderation_simple import waspas
from core.classement.topsis import topsis
from core.commun import ErreurMethode
from ui import affichage as af
from ui import graphiques as gr
from ui.donnees import charger_tableau, editeur_tableau

af.entete("LLM extracteur sémantique de scores", "MCDM & IA générative", "151–154")

with st.expander("Principe (cours p. 152–153)", expanded=True):
    st.markdown("De nombreuses décisions s'appuient sur des documents non structurés (CV, rapports d'audit, avis clients…). "
                "Un LLM fine-tuné sur des annotations d'experts reçoit un texte $T$ et une liste de critères "
                "$C = \\{c_1, \\dots, c_n\\}$, et produit un vecteur de scores qui devient une ligne de la matrice de décision :")
    st.latex(r"f_{LLM}(T, C) \rightarrow s = (s_1, \dots, s_n) \in [0, 10]^n,\qquad \text{Décision} = MCDM(s)")
    st.markdown("""
| | |
|---|---|
| **Entrées** | texte source T ; liste ordonnée de critères avec leurs définitions opérationnelles ; vecteur de poids w |
| **Sorties** | score par critère (ex. expérience 8/10, communication 7/10, prétentions salariales 3/10 : score inversé car critère coût) ; justification textuelle |
| **Données SFT** | triplets (T, C, s*) où s* est annoté par des experts ; augmentation par paraphrase et génération synthétique |
| **Avantages** | exploite l'information non structurée ; scores interprétables et justifiables ; tout domaine avec annotation d'experts |
| **Limites** | qualité dépendante des annotations ; subjectivité inter-annotateurs ; nécessite une rubrique de notation rigoureuse |
""")
    st.info("Cette page n'appelle aucun LLM : les scores sont **simulés**, rédigés selon la rubrique ci-dessous. "
            "Le CV du candidat 1 est l'exemple de la p. 154.")

st.subheader("Rubrique de notation (définitions opérationnelles des critères)")
st.markdown("""
- **Expérience technique** (0–10) : années d'expérience pertinente, technologies du poste, responsabilités techniques.
- **Communication / leadership** (0–10) : management, prise de parole, transmission démontrés par des éléments vérifiables.
- **Adéquation salariale** (0–10, **score inversé**) : 10 = prétentions basses par rapport à l'enveloppe du poste,
  0 = très au-dessus. Le critère est un coût, mais le score est déjà inversé : la colonne est donc un critère **+**.
""")

corpus = editeur_tableau("llm_corpus", charger_tableau("D8_llm_cv_corpus", "tableau"), titre="Extraits de CV et scores",
                         nom_fichier="cv_scores.csv")
criteres = ["Expérience technique", "Communication / leadership", "Adéquation salariale (coût, score inversé)"]
try:
    X = corpus[criteres].apply(pd.to_numeric).to_numpy(float)
    candidats = [str(c) for c in corpus["candidat"]]
    if np.isnan(X).any() or np.any((X < 0) | (X > 10)):
        raise ValueError
except (ValueError, KeyError):
    st.error("Chaque candidat doit avoir trois scores entre 0 et 10.")
    st.stop()

with st.expander("Exemple de la p. 154 : sortie du LLM pour le candidat 1", expanded=False):
    l = corpus.iloc[0]
    st.markdown(f"> {l['extrait_cv']}")
    st.dataframe(pd.DataFrame({"Score LLM": [f"{int(l[c])}/10" for c in criteres],
                               "Justification générée": [l["justification_experience"], l["justification_communication"],
                                                         l["justification_salaire"]]}, index=criteres))

st.subheader("Décision = MCDM(s)")
c1, c2, c3 = st.columns(3)
w = np.array([c1.number_input("w expérience", 0.0, 1.0, 0.4, 0.05), c2.number_input("w communication", 0.0, 1.0, 0.3, 0.05),
              c3.number_input("w adéquation salariale", 0.0, 1.0, 0.3, 0.05)])
if abs(w.sum() - 1) > 1e-3:
    st.error(f"La somme des poids vaut {w.sum():.2f} au lieu de 1.")
    st.stop()
methode = st.radio("Méthode", ["TOPSIS", "WSM"], horizontal=True)
double = st.checkbox("Déclarer (à tort) l'adéquation salariale comme critère coût « − » (double inversion, E-26)")
sens = np.array([1, 1, -1 if double else 1])


def classer(sens_):
    if methode == "TOPSIS":
        r = topsis(np.maximum(X, 1e-9), sens_, w)
        return r.RC, r.rangs
    r = waspas(np.maximum(X, 1e-9), sens_, w, 1.0)
    return r.Q1, r.rangs_Q1


try:
    score, rg = classer(sens)
    score_ok, rg_ok = classer(np.array([1, 1, 1]))
except ErreurMethode as exc:
    st.error(str(exc))
    st.stop()
af.classement(candidats, {"Score": score}, rg)
gr.afficher(gr.barres(candidats, score, f"Score {methode} des candidats", "score"))
if double:
    st.error(f"Avec la double inversion, le meilleur candidat devient **{candidats[int(np.argmin(rg))]}** au lieu de "
             f"**{candidats[int(np.argmin(rg_ok))]}** : des prétentions élevées sont alors récompensées. Le score étant "
             "déjà inversé par la rubrique, la colonne doit rester un critère « + ».")
else:
    st.success(f"Meilleur candidat selon {methode} : **{candidats[int(np.argmin(rg))]}**.")
