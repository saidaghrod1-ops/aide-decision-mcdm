"""Page d'accueil : cadre du cours, guide de choix d'une méthode (p. 12), format des données."""

import pandas as pd
import streamlit as st

from core.donnees import charger_catalogue
from ui.donnees import charger_matrice, editeur_matrice

st.title("Aide à la décision multicritère (MCDM)")
st.caption("Application de démonstration construite à partir du support de cours « Aide à la décision » (2026). "
           "Notations, formules et conventions du cours.")

st.markdown(
    """
**Objectifs pédagogiques généraux** (p. 2)
- Formaliser un problème de décision réel en modèle MCDM (alternatives, critères, poids).
- Choisir et appliquer la méthode MCDM adaptée à la nature des données (crisp vs floue).
- Comparer et interpréter des classements issus de méthodes différentes.
- Intégrer des méthodes MCDM dans un pipeline IA (agents LLM, RAG, aide à la décision automatisée).
"""
)

with st.expander("Le problème multicritère (p. 4–9)", expanded=False):
    st.markdown(
        """
Avec un seul critère, on retient l'alternative la mieux notée. En pratique, le choix doit tenir compte de
**plusieurs critères contradictoires** : aucune alternative n'est la meilleure sur tous les critères,
il faut **pondérer les critères** et trouver **une solution de compromis**. C'est le rôle des méthodes MCDM (p. 5).

Deux grandes catégories (p. 6) : l'**optimisation multi-objectifs (MODM)** et l'**analyse multi-attributs (MADM)**.
Un problème MADM est formulé par une matrice de décision (p. 8) :
"""
    )
    st.latex(r"D_{mn}=\begin{pmatrix} x_{11} & \cdots & x_{1n}\\ \vdots & \ddots & \vdots\\ x_{m1} & \cdots & x_{mn}\end{pmatrix},"
             r"\qquad W=(w_1,w_2,\dots,w_n)")
    st.markdown(
        """
où $x_{ij}$ est le score de l'alternative $A_i$ par rapport à l'attribut $C_j$.

**Typologie des critères (p. 9)**
- *Sens de préférence* : critères **positifs** (bénéfice, à maximiser, notés « + ») et **négatifs** (coût, à minimiser, notés « − »).
- *Relation* : critères indépendants ou dépendants (une méthode comme ANP est alors nécessaire).
- *Nature* : quantitatifs (valeurs numériques) ou qualitatifs (variables linguistiques à convertir en échelle numérique).
"""
    )

with st.expander("Taxonomie et processus de décision (p. 10–13)", expanded=False):
    st.markdown(
        """
1. **Précision des évaluations** : crisp MCDM (valeurs exactes) / fuzzy MCDM (préférences subjectives).
2. **Nombre de décideurs** : single MCDM / group MCDM (agrégation des avis).
3. **Nombre d'alternatives** : une seule (évaluation par rapport à des seuils) / plusieurs (comparer et classer).
4. **Étape du processus** :
   - *Étape 1 — poids des critères* : méthodes subjectives (AHP, BWM) et objectives (CRITIC, entropie) ;
   - *Étape 2 — classement* : méthodes additives / compensatoires (WSM, WPM, AHP, TOPSIS, VIKOR) et
     méthodes de surclassement (ELECTRE, PROMETHEE).

**Processus de prise de décision (p. 13)** : 1) identifier le problème ; 2) élaborer un ensemble d'alternatives ;
3) déterminer les critères d'évaluation et collecter les données ; 4) choisir une ou plusieurs méthodes pour la
pondération des critères et le classement des alternatives.
"""
    )
    st.info("ELECTRE III et PROMETHEE sont annoncées dans le cours (p. 74–75) sans contenu : "
            "elles n'ont donc pas de page dans cette application.")

# ---------------------------------------------------------------------------
st.header("Quelle méthode choisir ? (p. 12)")
st.write("Aucune méthode n'est meilleure que les autres dans l'absolu : le choix dépend du problème posé.")
c1, c2 = st.columns(2)
with c1:
    nature = st.radio("Nature des données", ["Valeurs exactes", "Jugements imprécis ou linguistiques"], key="guide_nature")
    decideurs = st.radio("Nombre de décideurs", ["Un seul expert", "Plusieurs experts"], key="guide_decideurs")
with c2:
    compensation = st.radio("Compensation entre critères", ["Acceptée", "Refusée (sécurité, environnement…)"],
                            key="guide_compensation")
    effort = st.radio("Effort de comparaison acceptable pour le décideur",
                      ["Beaucoup de comparaisons", "Peu de comparaisons", "Aucune comparaison"], key="guide_effort")

st.markdown("**Recommandations du cours**")
if effort == "Beaucoup de comparaisons":
    st.markdown("- Pondération : **AHP** (beaucoup de comparaisons par paires).")
    st.page_link("pages/ponderation_ahp.py", label="AHP", icon=":material/arrow_forward:")
elif effort == "Peu de comparaisons":
    st.markdown("- Pondération : **BWM** (2n − 3 comparaisons au lieu de n(n − 1)/2 pour AHP).")
    st.page_link("pages/ponderation_bwm.py", label="BWM", icon=":material/arrow_forward:")
else:
    st.markdown("- Pondération : **CRITIC** ou **entropie** (aucune comparaison, poids calculés à partir des données).")
    st.page_link("pages/ponderation_critic.py", label="CRITIC", icon=":material/arrow_forward:")
    st.page_link("pages/ponderation_entropie.py", label="Entropie", icon=":material/arrow_forward:")

if compensation.startswith("Refusée"):
    st.markdown("- Classement : le cours renvoie aux méthodes de **surclassement (ELECTRE, PROMETHEE)**, "
                "annoncées sans contenu (p. 74–75) et donc non traitées ici.")
elif nature.startswith("Jugements"):
    st.markdown("- Données linguistiques : **méthodes floues**"
                + (" avec agrégation des avis des décideurs (group MCDM)." if decideurs == "Plusieurs experts" else "."))
    st.page_link("pages/mcdm_flou_topsis.py", label="TOPSIS flou", icon=":material/arrow_forward:")
    st.page_link("pages/mcdm_flou_topsis_intuitionniste.py", label="TOPSIS intuitionniste", icon=":material/arrow_forward:")
else:
    st.markdown("- Classement : **méthodes additives / compensatoires** (WSM, TOPSIS ; aussi WPM, WASPAS, VIKOR).")
    st.page_link("pages/classement_topsis.py", label="TOPSIS", icon=":material/arrow_forward:")
    st.page_link("pages/classement_ponderation_simple.py", label="WSM, WPM, WASPAS", icon=":material/arrow_forward:")
    st.page_link("pages/classement_vikor.py", label="VIKOR", icon=":material/arrow_forward:")
    if decideurs == "Plusieurs experts":
        st.markdown("- Plusieurs experts : group MCDM avec une stratégie d'agrégation des avis.")

st.markdown(
    """
**Bonnes pratiques (p. 12)** : appliquer plusieurs méthodes et comparer les classements obtenus ;
mener une analyse de sensibilité et de robustesse (faire varier les poids et vérifier la stabilité du classement) ;
justifier explicitement le choix de la méthode retenue.
"""
)
st.page_link("pages/robustesse_comparaison.py", label="Comparaison multi-méthodes", icon=":material/arrow_forward:")

# ---------------------------------------------------------------------------
st.header("Données : modifier, importer, exporter")
st.markdown(
    """
Chaque page propose un jeu de données par défaut (exemple ou exercice du cours en priorité), **modifiable dans le
tableau**, **importable** et **exportable en CSV**. Format d'une matrice de décision :
"""
)
st.code(
    "alternative,C1 Style,C2 Fiabilité,C3 Économie de carburant,C4 Coût\n"
    "@sens,+,+,+,-\n"
    "@poids,0.2,0.1,0.4,0.3\n"
    "M1,7,9,9,8\n"
    "M2,8,7,8,7",
    language="text",
)
st.markdown(
    "- 1re colonne : nom de l'alternative ; une colonne par critère.\n"
    "- Lignes facultatives : `@sens` (`+` bénéfice, `-` coût ; `+` par défaut), `@poids` (poids égaux par défaut), "
    "`@unite`.\n"
    "- Séparateur virgule, point décimal ; les fractions comme `1/7` sont acceptées (matrices AHP)."
)

with st.expander("Essayer l'éditeur avec l'exercice TOPSIS du cours (p. 65)", expanded=False):
    editeur_matrice("accueil_demo", charger_matrice("C5_topsis_voitures"), nom_fichier="topsis_voitures.csv")

st.subheader("Jeux de données disponibles")
catalogue = charger_catalogue()
st.dataframe(
    pd.DataFrame(
        [{"Identifiant": d["id"], "Titre": d["titre"], "Origine": "Cours" if d["origine"] == "cours" else "Créé",
          "Pages du cours": d["pages_cours"], "Contexte": d["contexte"]} for d in catalogue]
    ),
    hide_index=True,
)
