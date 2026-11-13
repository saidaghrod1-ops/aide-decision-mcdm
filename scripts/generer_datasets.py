"""Génère les jeux de données créés D1–D8 dans data/crees/ (graines fixes, résultat reproductible).

Usage : python scripts/generer_datasets.py
L'application ne lance jamais ce script : elle lit les CSV figés.
D1 nécessite scikit-learn et xgboost (requirements-dev.txt).
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

import numpy as np

RACINE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RACINE))
SORTIE = RACINE / "data" / "crees"
SORTIE.mkdir(parents=True, exist_ok=True)

from core.donnees import MatriceDecision, format_nombre, matrice_comparaison_vers_csv, matrice_vers_csv  # noqa: E402


def ecrire(nom: str, texte: str) -> None:
    (SORTIE / nom).write_text(texte, encoding="utf-8")
    print("écrit :", nom)


def ecrire_tableau(nom: str, entete, lignes) -> None:
    with (SORTIE / nom).open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow(entete)
        for l in lignes:
            w.writerow([format_nombre(x) if isinstance(x, (float, np.floating)) else x for x in l])
    print("écrit :", nom)


# ---------------------------------------------------------------- D1
def d1_selection_modele_ml() -> None:
    """Les 5 modèles et les 6 critères cités p. 14, validation croisée stratifiée 5 plis."""
    from sklearn.datasets import load_breast_cancer
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import cohen_kappa_score, make_scorer
    from sklearn.model_selection import StratifiedKFold, cross_validate
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.svm import SVC
    from xgboost import XGBClassifier

    X, y = load_breast_cancer(return_X_y=True)
    modeles = {
        "M1 SVM": make_pipeline(StandardScaler(), SVC(random_state=42)),
        "M2 Random Forest": RandomForestClassifier(n_estimators=200, random_state=42),
        "M3 Régression logistique": make_pipeline(StandardScaler(), LogisticRegression(max_iter=5000)),
        "M4 XGBoost": XGBClassifier(n_estimators=200, max_depth=3, learning_rate=0.1, random_state=42,
                                    eval_metric="logloss"),
        "M5 KNN": make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=5)),
    }
    metriques = {"Accuracy": "accuracy", "Recall": "recall", "Precision": "precision", "F1 score": "f1",
                 "AUC": "roc_auc", "Kappa": make_scorer(cohen_kappa_score)}
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    valeurs = []
    for nom, m in modeles.items():
        r = cross_validate(m, X, y, cv=cv, scoring=metriques)
        valeurs.append([round(float(np.mean(r[f"test_{k}"])), 4) for k in metriques])
    M = MatriceDecision(tuple(modeles), tuple(metriques), np.array(valeurs), np.ones(len(metriques), int))
    ecrire("selection_modele_ml.csv", matrice_vers_csv(M))


# ---------------------------------------------------------------- D2
def d2_fournisseurs() -> None:
    M = MatriceDecision(
        ("F1 Premium", "F2 Low-cost", "F3 Équilibré", "F4 Express", "F5 Responsable"),
        ("C1 Qualité", "C2 Prix", "C3 Délai de livraison", "C4 Engagement RSE"),
        np.array([[9.2, 145, 12, 70], [6.1, 98, 20, 45], [8.0, 118, 9, 62], [7.2, 125, 4, 55], [7.6, 130, 14, 92]]),
        np.array([1, -1, -1, 1]), None, ("/10", "DH/unité", "jours", "/100"))
    ecrire("fournisseurs_5offres.csv", matrice_vers_csv(M))


# ---------------------------------------------------------------- D3
SAATY = np.array([1 / 9, 1 / 8, 1 / 7, 1 / 6, 1 / 5, 1 / 4, 1 / 3, 1 / 2, 1, 2, 3, 4, 5, 6, 7, 8, 9])


def _matrice_saaty(scores) -> np.ndarray:
    """Matrice réciproque proche des rapports de scores, arrondie à l'échelle 1–9."""
    s = np.asarray(scores, float)
    n = len(s)
    A = np.ones((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            r = s[i] / s[j]
            A[i, j] = SAATY[np.argmin(np.abs(np.log(SAATY) - np.log(r)))]
            A[j, i] = 1 / A[i, j]
    return A


def d3_ahp_hopital() -> None:
    from core.ponderation.ahp import poids_ahp

    criteres = ["Population desservie", "Accessibilité", "Coût du terrain", "Personnel médical", "Impact environnemental"]
    villes = ["Ville A", "Ville B", "Ville C", "Ville D"]
    # Version incohérente : « Coût du terrain » jugé 3 fois plus important que « Personnel médical »,
    # alors que Personnel > Accessibilité > Coût du terrain.
    incoherente = np.array([
        [1, 3, 5, 2, 7],
        [1 / 3, 1, 3, 1 / 2, 5],
        [1 / 5, 1 / 3, 1, 3, 3],
        [1 / 2, 2, 1 / 3, 1, 4],
        [1 / 7, 1 / 5, 1 / 3, 1 / 4, 1]])
    corrigee = incoherente.copy()
    corrigee[2, 3], corrigee[3, 2] = 1 / 3, 3
    cr_inc, cr_cor = poids_ahp(incoherente).CR, poids_ahp(corrigee).CR
    print(f"D3 : CR incohérente = {cr_inc:.4f}, CR corrigée = {cr_cor:.4f}")
    assert cr_inc > 0.1 > cr_cor
    ecrire("ahp_hopital_criteres_incoherente.csv", matrice_comparaison_vers_csv(criteres, incoherente))
    ecrire("ahp_hopital_criteres_corrigee.csv", matrice_comparaison_vers_csv(criteres, corrigee))
    # Évaluations sous-jacentes des 4 villes (plus grand = meilleur pour le critère)
    scores = {
        "Population desservie": [9, 5, 3, 6],
        "Accessibilité": [4, 8, 6, 5],
        "Coût du terrain": [2, 5, 8, 6],          # terrain bon marché = meilleur
        "Personnel médical": [8, 6, 3, 5],
        "Impact environnemental": [3, 5, 8, 6],   # faible impact = meilleur
    }
    for c in criteres:
        A = _matrice_saaty(scores[c])
        assert poids_ahp(A).CR < 0.1, c
        nom = "ahp_hopital_" + c.lower().replace(" ", "_").replace("é", "e").replace("û", "u") + ".csv"
        ecrire(nom, matrice_comparaison_vers_csv(villes, A))


# ---------------------------------------------------------------- D4
def d4_electrification() -> None:
    echelle_eval = [("Très élevé", "TE", 0.90, 0.05), ("Élevé", "E", 0.75, 0.15),
                    ("Moyennement élevé", "ME", 0.60, 0.30), ("Moyen", "M", 0.50, 0.40),
                    ("Moyennement faible", "MF", 0.35, 0.55), ("Faible", "F", 0.20, 0.70),
                    ("Très faible", "TF", 0.05, 0.90)]
    echelle_poids = [("Très important", "TI", 0.90, 0.05), ("Important", "I", 0.75, 0.20),
                     ("Moyen", "M", 0.50, 0.40), ("Peu important", "PI", 0.30, 0.60),
                     ("Très peu important", "TPI", 0.10, 0.85)]
    ecrire_tableau("village_echelle_evaluations.csv", ["terme", "code", "mu", "nu"], echelle_eval)
    ecrire_tableau("village_echelle_importance.csv", ["terme", "code", "mu", "nu"], echelle_poids)
    criteres = ["C1 Coût d'investissement", "C2 Fiabilité", "C3 Impact environnemental", "C4 Acceptabilité sociale"]
    decideurs = ["Ingénieur", "Économiste", "Élu local"]
    evaluations = {
        "A1 Solaire PV + stockage": [["ME", "E", "ME"], ["ME", "M", "ME"], ["F", "F", "TF"], ["E", "E", "TE"]],
        "A2 Éolien": [["E", "E", "E"], ["M", "MF", "M"], ["MF", "M", "ME"], ["M", "MF", "MF"]],
        "A3 Hybride PV-éolien": [["TE", "E", "TE"], ["E", "E", "TE"], ["F", "MF", "F"], ["E", "ME", "E"]],
        "A4 Extension du réseau": [["M", "ME", "M"], ["TE", "E", "E"], ["ME", "E", "ME"], ["ME", "E", "ME"]],
    }
    lignes = [[a, decideurs[k], criteres[j], termes[k]]
              for a, par_critere in evaluations.items() for j, termes in enumerate(par_critere) for k in range(3)]
    ecrire_tableau("village_evaluations.csv", ["alternative", "decideur", "critere", "terme"], lignes)
    poids = {"Ingénieur": ["I", "TI", "M", "M"], "Économiste": ["TI", "I", "M", "PI"], "Élu local": ["I", "I", "I", "TI"]}
    ecrire_tableau("village_poids_criteres.csv", ["decideur", "critere", "terme"],
                   [[k, criteres[j], t] for k, ts in poids.items() for j, t in enumerate(ts)])
    ecrire_tableau("village_poids_decideurs.csv", ["decideur", "sigma"],
                   [["Ingénieur", 0.40], ["Économiste", 0.35], ["Élu local", 0.25]])
    ecrire_tableau("village_sens.csv", ["critere", "sens"], [[c, s] for c, s in zip(criteres, ["-", "+", "-", "+"])])


# ---------------------------------------------------------------- D5
CREDIT_CRITERES = ["C1 Revenu mensuel net", "C2 Taux d'endettement", "C3 Ancienneté professionnelle",
                   "C4 Incidents de paiement (24 mois)", "C5 Apport personnel"]
CREDIT_SENS = ["+", "-", "+", "-", "+"]
CREDIT_UNITES = ["DH", "%", "années", "nombre", "% du projet"]
CREDIT_MIN = [4000, 0, 0, 0, 0]
CREDIT_MAX = [40000, 45, 20, 6, 40]
CREDIT_POIDS = [0.30, 0.25, 0.15, 0.20, 0.10]


def d5_credit() -> None:
    dossier = [22000, 18, 8, 0, 20]
    ecrire_tableau("credit_dossier.csv", ["critere", "sens", "unite", "poids", "valeur", "x_min", "x_max"],
                   [[c, s, u, float(w), float(v), float(a), float(b)] for c, s, u, w, v, a, b in
                    zip(CREDIT_CRITERES, CREDIT_SENS, CREDIT_UNITES, CREDIT_POIDS, dossier, CREDIT_MIN, CREDIT_MAX)])
    rng = np.random.default_rng(2026)
    n = 200
    revenu = np.round(np.clip(rng.lognormal(np.log(13000), 0.45, n), 4000, 40000), -2)
    endet = np.round(np.clip(rng.normal(25, 10, n), 0, 45), 1)
    anciennete = np.round(np.clip(rng.gamma(2.0, 3.0, n), 0, 20), 1)
    incidents = np.clip(rng.poisson(0.9, n), 0, 6)
    apport = np.round(np.clip(rng.normal(15, 9, n), 0, 40), 1)
    X = np.column_stack([revenu, endet, anciennete, incidents, apport])
    s = np.array([1 if c == "+" else -1 for c in CREDIT_SENS])
    lo, hi = np.array(CREDIT_MIN, float), np.array(CREDIT_MAX, float)
    U = np.where(s > 0, (X - lo) / (hi - lo), (X - hi) / (lo - hi))
    S = U @ np.array(CREDIT_POIDS)
    proba = 1 / (1 + np.exp(-14 * (S - np.median(S))))
    y = (rng.uniform(size=n) < proba).astype(int)
    lignes = [[f"D{i + 1:03d}", *[float(v) for v in X[i]], "Accepter" if y[i] else "Rejeter"] for i in range(n)]
    ecrire_tableau("credit_historique.csv", ["dossier", *CREDIT_CRITERES, "decision"], lignes)


# ---------------------------------------------------------------- D6
def d6_fournisseurs_historique() -> None:
    rng = np.random.default_rng(91)
    n = 300
    qualite = np.round(rng.uniform(5, 10, n), 1)
    prix = np.round(60 + 8 * qualite + rng.normal(0, 9, n), 0)
    delai = np.round(rng.uniform(2, 25, n), 0)
    rse = np.round(rng.uniform(30, 95, n), 0)
    M = MatriceDecision(tuple(f"F{i + 1:03d}" for i in range(n)),
                        ("C1 Qualité", "C2 Prix", "C3 Délai de livraison", "C4 Engagement RSE"),
                        np.column_stack([qualite, prix, delai, rse]), np.array([1, -1, -1, 1]), None,
                        ("/10", "DH/unité", "jours", "/100"))
    ecrire("fournisseurs_historique.csv", matrice_vers_csv(M))


# ---------------------------------------------------------------- D7
def d7_series() -> None:
    rng = np.random.default_rng(94)
    T = 60
    moyennes = {"A1 Action tech": (8.0, 0.25, 100.0), "A2 Action énergie": (6.0, 0.15, 80.0),
                "A3 Action bancaire": (7.0, 0.20, 120.0)}
    criteres = ["Rendement attendu r (%)", "Volatilité σ", "Liquidité L"]
    bruits = (0.6, 0.02, 8.0)
    lignes = []
    for a, mus in moyennes.items():
        for c, mu, sd in zip(criteres, mus, bruits):
            x = np.empty(T); x[0] = mu
            for t in range(1, T):
                x[t] = mu + 0.6 * (x[t - 1] - mu) + rng.normal(0, sd)
            x = np.round(x, 4 if c.startswith("Vol") else 2)
            lignes += [[t + 1, a, c, float(x[t])] for t in range(T)]
    ecrire_tableau("dynamique_series.csv", ["periode", "alternative", "critere", "valeur"], lignes)


# ---------------------------------------------------------------- D8
def d8_cv() -> None:
    cvs = [
        ("Candidat 1", "Ingénieur logiciel avec 7 ans d'expérience en développement Python et Java, management d'une "
                       "équipe de 5 personnes, certifié PMP. Prétentions : 85 000 €/an.",
         8, 7, 3, "7 ans en Python/Java, management d'équipe et certification PMP valorisent l'expérience",
         "Management d'une équipe signalé mais pas de références externes ni d'exemples de communication",
         "Prétentions à 85 k€ situées dans la fourchette haute du marché pour le poste visé"),
        ("Candidat 2", "Développeuse backend, 4 ans d'expérience en Python et bases de données, contributrice open "
                       "source, animatrice de meetups techniques. Prétentions : 58 000 €/an.",
         6, 8, 7, "4 ans d'expérience ciblée, contributions open source vérifiables",
         "Animation de meetups : prise de parole régulière devant un public technique",
         "Prétentions dans la médiane du marché"),
        ("Candidat 3", "Ingénieur junior, 1 an d'alternance en développement web (JavaScript, un peu de Python). "
                       "Prétentions : 42 000 €/an.",
         3, 5, 9, "Une seule année d'expérience, en alternance, peu de Python",
         "Pas d'élément sur la communication hormis le travail en alternance",
         "Prétentions basses pour le poste"),
        ("Candidat 4", "Architecte logiciel, 12 ans d'expérience (Java, Python, cloud), a dirigé une équipe de 15 "
                       "personnes, conférencier. Prétentions : 110 000 €/an.",
         10, 9, 1, "12 ans d'expérience, architecture et cloud : profil très senior",
         "Direction d'une équipe de 15 personnes et conférences : communication très démontrée",
         "Prétentions nettement au-dessus de l'enveloppe du poste"),
        ("Candidat 5", "Ingénieure logiciel, 5 ans d'expérience en Python (data et API), tutrice d'étudiants, "
                       "certifiée Scrum Master. Prétentions : 65 000 €/an.",
         7, 7, 6, "5 ans en Python sur des sujets data et API",
         "Tutorat et rôle de Scrum Master : communication régulière en équipe",
         "Prétentions légèrement au-dessus de la médiane"),
    ]
    ecrire_tableau("llm_cv_corpus.csv",
                   ["candidat", "extrait_cv", "Expérience technique", "Communication / leadership",
                    "Adéquation salariale (coût, score inversé)", "justification_experience",
                    "justification_communication", "justification_salaire"],
                   [list(c) for c in cvs])


if __name__ == "__main__":
    etapes = {"D1": d1_selection_modele_ml, "D2": d2_fournisseurs, "D3": d3_ahp_hopital, "D4": d4_electrification,
              "D5": d5_credit, "D6": d6_fournisseurs_historique, "D7": d7_series, "D8": d8_cv}
    choix = sys.argv[1:] or list(etapes)
    for cle in choix:
        etapes[cle]()
