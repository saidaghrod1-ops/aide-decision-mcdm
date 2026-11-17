# Aide à la décision multicritère (MCDM) — application Streamlit

Application de démonstration des méthodes du cours « Aide à la décision » (2026), avec le calcul pas à pas
de chaque méthode, sur les exemples et exercices du cours et sur des jeux de données créés :

- **Pondération des critères** : AHP, BWM (subjectives) ; entropie, CRITIC (objectives).
- **Classement des alternatives** : WSM, WPM, WASPAS, TOPSIS, VIKOR.
- **Robustesse** : corrélation de Spearman, comparaison multi-méthodes, chaînage pondération → classement, sensibilité aux poids.
- **Alternative unique** : MAUT, distance à l'idéal, choix du seuil de décision (courbe ROC), sur une même page.
- **Logique floue** : ensembles flous, nombres flous et intervalles, intuitionnistes, pythagoriciens, q-ROF, picturaux, hésitants.
- **MCDM flou** : alternative unique floue, TOPSIS floue (Chen, 2000), TOPSIS intuitionniste.

Interface en français ; notations, formules et conventions du cours ; aucune méthode hors cours.
L'inventaire du cours, ses conventions et **les écarts constatés avec le cours** (E-1 à E-32) sont dans [NOTES_COURS.md](NOTES_COURS.md).

## Lancer l'application en local

Python 3.12 :

```bash
python -m venv .venv
.venv\Scripts\activate            # Windows  (source .venv/bin/activate sous Linux/macOS)
pip install -r requirements-dev.txt
pytest                             # 127 tests + 1 écart documenté avec le cours (E-1)
streamlit run streamlit_app.py
```

## Déploiement sur Streamlit Community Cloud

1. Se connecter à <https://share.streamlit.io> avec le compte GitHub propriétaire du dépôt.
2. **Create app** → *Deploy a public app from GitHub* : dépôt `saidaghrod1-ops/aide-decision-mcdm`, branche `main`,
   fichier principal `streamlit_app.py`.
3. **Advanced settings** → Python **3.12** → **Deploy**. Aucun secret n'est nécessaire.
4. Chaque `git push` sur `main` redéploie l'application automatiquement.

## Organisation du dépôt

| Dossier | Rôle |
|---|---|
| `core/` | Calculs purs (numpy/scipy), sans Streamlit ; chaque méthode retourne toutes ses matrices intermédiaires |
| `ui/` | Composants Streamlit partagés (éditeurs de données, import/export CSV, graphiques Plotly) |
| `pages/` | Une page par méthode ; navigation définie dans `streamlit_app.py` |
| `data/` | `cours/` : exemples et exercices du cours, transcrits à l'identique ; `crees/` : jeux créés ; description dans `catalogue.json` |
| `scripts/` | `generer_datasets.py` : régénère les jeux créés (graines fixes) |
| `tests/` | Tests pytest : valeurs du cours, propriétés, formats de données, exécution de chaque page |

## Format CSV d'une matrice de décision

```
alternative,C1 Style,C2 Fiabilité,C3 Économie de carburant,C4 Coût
@sens,+,+,+,-
@poids,0.2,0.1,0.4,0.3
M1,7,9,9,8
M2,8,7,8,7
```

- 1re colonne : nom de l'alternative ; une colonne par critère.
- Lignes facultatives : `@sens` (`+` bénéfice, `-` coût ; `+` par défaut), `@poids` (poids égaux par défaut), `@unite`.
- Séparateur virgule, point décimal, encodage UTF-8 ; les fractions comme `1/7` sont acceptées.
- Matrices de comparaison AHP : CSV carré avec en-têtes ; les cases vides du triangle inférieur sont complétées par réciprocité.
- Chaque page permet de modifier les données dans un tableau, de les importer et de les exporter en CSV.

## Sources des données

- Exemples et exercices du support de cours (pages citées dans `data/catalogue.json`).
- Matrice de portefeuille de Vuković et al. (2020), reproduite dans le cours (p. 18).
- Exercice de TOPSIS floue de Chen (2000), reproduit dans le cours (p. 140–141).
- Jeu « sélection de modèle ML » : mesures obtenues sur le jeu Breast Cancer Wisconsin inclus dans scikit-learn.
- Autres jeux créés : synthétiques, graines fixes (voir `scripts/generer_datasets.py`).
