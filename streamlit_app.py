"""Point d'entrée de l'application « Aide à la décision multicritère ».

Lancement local : streamlit run streamlit_app.py
"""

import streamlit as st

st.set_page_config(page_title="Aide à la décision multicritère", page_icon=":material/balance:", layout="wide")

# (fichier, titre, icône) regroupés par section, dans l'ordre du cours.
SECTIONS = {
    "Introduction": [
        ("pages/accueil.py", "Accueil", ":material/home:"),
    ],
    "Pondération des critères": [
        ("pages/ponderation_ahp.py", "AHP", ":material/account_tree:"),
        ("pages/ponderation_bwm.py", "BWM (Best-Worst)", ":material/swap_vert:"),
        ("pages/ponderation_entropie.py", "Entropie", ":material/scatter_plot:"),
        ("pages/ponderation_critic.py", "CRITIC", ":material/grid_on:"),
    ],
    "Classement des alternatives": [
        ("pages/classement_ponderation_simple.py", "WSM, WPM, WASPAS", ":material/functions:"),
        ("pages/classement_topsis.py", "TOPSIS", ":material/my_location:"),
        ("pages/classement_vikor.py", "VIKOR", ":material/handshake:"),
    ],
    "Robustesse et comparaison": [
        ("pages/robustesse_spearman.py", "Corrélation de Spearman", ":material/compare_arrows:"),
        ("pages/robustesse_comparaison.py", "Comparaison multi-méthodes", ":material/leaderboard:"),
        ("pages/robustesse_chainage.py", "Chaînage pondération → classement", ":material/link:"),
    ],
    "Alternative unique": [
        ("pages/unique_maut_seuil.py", "MAUT et seuil de décision", ":material/rule:"),
        ("pages/unique_distance_ideal.py", "Distance à l'idéal", ":material/straighten:"),
    ],
    "MCDM et IA": [
        ("pages/ia_classification.py", "De MCDM à la classification", ":material/category:"),
        ("pages/ia_xai_shap.py", "XAI (SHAP) et pondération", ":material/insights:"),
        ("pages/ia_dynamique.py", "MCDM dynamique", ":material/timeline:"),
        ("pages/ia_llm.py", "LLM extracteur de scores", ":material/description:"),
    ],
    "Logique floue": [
        ("pages/flou_ensembles.py", "Ensembles flous", ":material/blur_on:"),
        ("pages/flou_nombres.py", "Nombres flous et intervalles", ":material/change_history:"),
        ("pages/flou_intuitionniste.py", "Ensembles intuitionnistes", ":material/how_to_vote:"),
        ("pages/flou_extensions.py", "Pythagoricien, q-ROF, pictural", ":material/donut_large:"),
        ("pages/flou_hesitant.py", "Ensembles hésitants", ":material/scatter_plot:"),
    ],
    "MCDM flou": [
        ("pages/mcdm_flou_unique.py", "Alternative unique floue", ":material/filter_center_focus:"),
        ("pages/mcdm_flou_topsis.py", "TOPSIS flou", ":material/groups:"),
        ("pages/mcdm_flou_topsis_intuitionniste.py", "TOPSIS intuitionniste", ":material/diversity_3:"),
    ],
}

navigation = st.navigation(
    {section: [st.Page(fichier, title=titre, icon=icone, default=(fichier == "pages/accueil.py"))
               for fichier, titre, icone in pages]
     for section, pages in SECTIONS.items()}
)
navigation.run()
