import streamlit as st

st.title("Aide à la décision multicritère (MCDM)")
st.caption("Démonstration pas à pas des méthodes du cours « Aide à la décision ».")

st.markdown("""
Quand plusieurs critères **contradictoires** interviennent (coût, risque, qualité…) et qu'aucune alternative n'est
la meilleure partout, il faut **pondérer les critères** et trouver **une solution de compromis** : c'est le rôle des
méthodes MCDM. Le problème s'écrit avec une matrice de décision et un vecteur de poids :
""")
st.latex(r"D=\begin{pmatrix} x_{11} & \cdots & x_{1n}\\ \vdots & \ddots & \vdots\\ x_{m1} & \cdots & x_{mn}\end{pmatrix},"
         r"\qquad W=(w_1,\dots,w_n),\qquad \sum_j w_j = 1")
st.markdown("$x_{ij}$ : score de l'alternative $A_i$ selon le critère $C_j$ ; critère **+** (bénéfice, à maximiser) "
            "ou **−** (coût, à minimiser).")

c1, c2 = st.columns(2)
with c1:
    st.subheader("1. Pondérer les critères")
    st.page_link("pages/ponderation_ahp.py", label="AHP", icon=":material/arrow_forward:")
    st.page_link("pages/ponderation_bwm.py", label="BWM", icon=":material/arrow_forward:")
    st.page_link("pages/ponderation_entropie.py", label="Entropie", icon=":material/arrow_forward:")
    st.page_link("pages/ponderation_critic.py", label="CRITIC", icon=":material/arrow_forward:")
with c2:
    st.subheader("2. Classer les alternatives")
    st.page_link("pages/classement_ponderation_simple.py", label="WSM, WPM, WASPAS", icon=":material/arrow_forward:")
    st.page_link("pages/classement_topsis.py", label="TOPSIS", icon=":material/arrow_forward:")
    st.page_link("pages/classement_vikor.py", label="VIKOR", icon=":material/arrow_forward:")
    st.page_link("pages/robustesse_comparaison.py", label="Comparer les méthodes", icon=":material/arrow_forward:")

st.header("Quelle méthode choisir ?")
st.markdown("""
| Question | Réponse | Méthodes |
|---|---|---|
| Nature des données | valeurs exactes | méthodes classiques (crisp) |
| | jugements imprécis ou linguistiques | méthodes floues |
| Nombre de décideurs | un seul expert | single MCDM |
| | plusieurs | group MCDM, avec agrégation des avis |
| Compensation entre critères | acceptée | méthodes additives (WSM, TOPSIS…) |
| | refusée | surclassement (ELECTRE, PROMETHEE : annoncées sans contenu dans le cours) |
| Effort de comparaison | beaucoup | AHP |
| | moins | BWM |
| | aucun | CRITIC, entropie |
""")
st.info("Bonnes pratiques : appliquer plusieurs méthodes et comparer les classements ; faire varier les poids pour "
        "vérifier la stabilité du classement ; justifier le choix de la méthode.")

st.header("Données")
st.markdown("Chaque page part d'un exemple du cours (ou d'un jeu créé), **modifiable dans le tableau**, "
            "**importable** et **exportable en CSV**. Format d'une matrice de décision :")
st.code("alternative,C1 Style,C2 Fiabilité,C3 Économie,C4 Coût\n@sens,+,+,+,-\n@poids,0.2,0.1,0.4,0.3\nM1,7,9,9,8\nM2,8,7,8,7",
        language="text")
st.caption("Lignes @sens (+ / -) et @poids facultatives ; les fractions comme 1/7 sont acceptées.")
