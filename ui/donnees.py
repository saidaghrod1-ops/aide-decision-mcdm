"""Éditeurs de données : st.data_editor + import CSV + export CSV + réinitialisation.

La « base » de chaque éditeur (DataFrames) est conservée dans st.session_state ; elle n'est remplacée
que lors d'un import, d'une réinitialisation ou d'un ajout/suppression de critère (un numéro de version
dans la clé du widget force alors son rafraîchissement).
"""

from __future__ import annotations

import hashlib

import numpy as np
import pandas as pd
import streamlit as st

from core.donnees import (
    ErreurDonnees,
    MatriceDecision,
    chemin,
    completer_reciproque,
    dataset,
    format_nombre,
    lire_matrice_comparaison_csv,
    lire_matrice_csv,
    lire_nombre,
    lire_sens,
    lire_tableau_csv,
    matrice_comparaison_vers_csv,
    matrice_vers_csv,
    tableau_vers_csv,
    texte_sens,
)

OPTIONS_MATRICES = {
    "Sélection d'un modèle de ML": "D1_selection_modele_ml",
    "Choisir un fournisseur parmi 5 offres": "D2_fournisseurs_5offres",
    "Exercice du cours : 4 machines (WASPAS)": "C4_waspas_machines",
    "Exercice du cours : 4 marques de voiture (TOPSIS)": "C5_topsis_voitures",
    "Exercice du cours : 3 machines (VIKOR)": "C6_vikor_machines",
    "Exercice du cours : machines de découpage (CRITIC)": "C3_critic_machines_decoupe",
    "Portefeuille de 18 actions (Vuković et al., 2020)": "C12_portefeuille_vukovic2020",
}


def options_matrices(*identifiants: str) -> dict[str, str]:
    """Sous-ensemble ordonné des datasets « matrice » (premier = par défaut)."""
    inverse = {v: k for k, v in OPTIONS_MATRICES.items()}
    return {inverse[i]: i for i in identifiants}


def choisir_dataset(cle: str, options: dict[str, str]) -> str:
    """Liste déroulante « Jeu de données » ; options = {libellé: identifiant du catalogue}."""
    libelle = st.selectbox("Jeu de données", list(options), key=f"{cle}__dataset")
    st.caption(dataset(options[libelle])["contexte"])
    return options[libelle]


# --------------------------------------------------------------------------
# Chargement depuis le catalogue (mis en cache)
# --------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def charger_matrice(identifiant: str, fichier: str = "matrice") -> MatriceDecision:
    return lire_matrice_csv(chemin(dataset(identifiant)["fichiers"][fichier]))


@st.cache_data(show_spinner=False)
def charger_tableau(identifiant: str, fichier: str) -> pd.DataFrame:
    entete, lignes = lire_tableau_csv(chemin(dataset(identifiant)["fichiers"][fichier]))
    return _typer(pd.DataFrame(lignes, columns=entete))


@st.cache_data(show_spinner=False)
def charger_comparaison(relatif: str) -> tuple[tuple[str, ...], np.ndarray]:
    return lire_matrice_comparaison_csv(chemin(relatif))


def _typer(df: pd.DataFrame) -> pd.DataFrame:
    for col in df.columns:
        try:
            df[col] = pd.to_numeric(df[col])
        except (ValueError, TypeError):
            pass
    return df


# --------------------------------------------------------------------------
# Outils d'état
# --------------------------------------------------------------------------
def _version(cle: str) -> int:
    return st.session_state.get(f"{cle}__version", 0)


def _remplacer_base(cle: str, base: dict) -> None:
    st.session_state[f"{cle}__base"] = base
    st.session_state[f"{cle}__version"] = _version(cle) + 1


def _base(cle: str, initiale) -> dict:
    if f"{cle}__base" not in st.session_state:
        _remplacer_base(cle, initiale())
    return st.session_state[f"{cle}__base"]


def _import_et_reset(cle: str, convertir, initiale, libelle: str) -> None:
    """Téléversement CSV (importé une seule fois) et bouton de réinitialisation."""
    fichier = st.file_uploader(libelle, type=["csv"], key=f"{cle}__upload")
    if fichier is not None:
        empreinte = hashlib.sha256(fichier.getvalue()).hexdigest()
        if st.session_state.get(f"{cle}__empreinte") != empreinte:
            st.session_state[f"{cle}__empreinte"] = empreinte
            try:
                _remplacer_base(cle, convertir(fichier.getvalue()))
                st.session_state.pop(f"{cle}__erreur", None)
                st.rerun()
            except ErreurDonnees as exc:
                st.session_state[f"{cle}__erreur"] = str(exc)
    if f"{cle}__erreur" in st.session_state:
        st.error(f"Import impossible : {st.session_state[f'{cle}__erreur']}")
    if st.button("Revenir aux données par défaut", key=f"{cle}__reset_{_version(cle)}"):
        st.session_state.pop(f"{cle}__empreinte", None)
        _remplacer_base(cle, initiale())
        st.rerun()


# --------------------------------------------------------------------------
# Matrice de décision
# --------------------------------------------------------------------------
def _base_matrice(M: MatriceDecision, avec_poids: bool) -> dict:
    valeurs = pd.DataFrame(M.valeurs, columns=list(M.criteres))
    valeurs.insert(0, "Alternative", list(M.alternatives))
    criteres = {"Critère": list(M.criteres), "Sens": [texte_sens(s) for s in M.sens]}
    if avec_poids:
        criteres["Poids"] = [float(w) for w in M.poids_ou_egaux()]
    criteres["Unité"] = list(M.unites) if M.unites else [""] * M.n
    return {"valeurs": valeurs, "criteres": pd.DataFrame(criteres)}


def _construire_matrice(valeurs: pd.DataFrame, criteres: pd.DataFrame, avec_poids: bool) -> MatriceDecision:
    valeurs = valeurs.dropna(how="all")
    if valeurs.empty:
        raise ErreurDonnees("la matrice ne contient aucune alternative")
    alternatives = tuple("" if pd.isna(a) else str(a).strip() for a in valeurs["Alternative"])
    noms = tuple("" if pd.isna(c) else str(c).strip() for c in criteres["Critère"])
    if any(a == "" for a in alternatives) or any(c == "" for c in noms):
        raise ErreurDonnees("chaque alternative et chaque critère doivent avoir un nom")
    try:
        X = valeurs.iloc[:, 1:].apply(pd.to_numeric).to_numpy(dtype=float)
    except (ValueError, TypeError) as exc:
        raise ErreurDonnees("la matrice contient une valeur non numérique") from exc
    if np.isnan(X).any():
        raise ErreurDonnees("la matrice contient des cellules vides")
    sens = np.array([lire_sens(s) for s in criteres["Sens"]])
    poids = None
    if avec_poids:
        poids = pd.to_numeric(criteres["Poids"]).to_numpy(dtype=float)
        if np.isnan(poids).any():
            raise ErreurDonnees("chaque critère doit avoir un poids")
    unites = tuple("" if pd.isna(u) else str(u) for u in criteres["Unité"])
    return MatriceDecision(alternatives, noms, X, sens, poids, unites)


def editeur_matrice(cle: str, defaut: MatriceDecision, *, avec_poids: bool = True,
                    nom_fichier: str = "matrice.csv") -> MatriceDecision | None:
    """Matrice de décision modifiable (valeurs, sens, poids) avec import/export CSV."""
    initiale = lambda: _base_matrice(defaut, avec_poids)
    base = _base(cle, initiale)
    v = _version(cle)
    valeurs = st.data_editor(base["valeurs"], key=f"{cle}__valeurs_{v}", num_rows="dynamic", hide_index=True,
                             column_config={"Alternative": st.column_config.TextColumn("Alternative", required=True)})
    config = {"Critère": st.column_config.TextColumn("Critère", required=True),
              "Sens": st.column_config.SelectboxColumn("Sens (+ bénéfice, − coût)", options=["+", "-"], required=True),
              "Unité": st.column_config.TextColumn("Unité")}
    if avec_poids:
        config["Poids"] = st.column_config.NumberColumn("Poids", min_value=0.0, format="%.4f")
    criteres = st.data_editor(base["criteres"], key=f"{cle}__criteres_{v}", num_rows="fixed", hide_index=True,
                              column_config=config)
    try:
        M = _construire_matrice(valeurs, criteres, avec_poids)
    except ErreurDonnees as exc:
        M = None
        st.error(f"Données invalides : {exc}")

    with st.expander("Importer, exporter ou modifier les critères"):
        st.caption("CSV : 1re colonne = alternative ; lignes facultatives @sens (+/-), @poids, @unite.")
        _import_et_reset(cle, lambda c: _base_matrice(lire_matrice_csv(c), avec_poids), initiale, "Importer un CSV")
        if M is not None:
            st.download_button("Exporter en CSV", matrice_vers_csv(M, inclure_poids=avec_poids), file_name=nom_fichier,
                               mime="text/csv", key=f"{cle}__export")
        c1, c2 = st.columns(2)
        nouveau = c1.text_input("Nouveau critère", key=f"{cle}__nouveau_{v}")
        if c1.button("Ajouter", key=f"{cle}__ajouter_{v}") and nouveau.strip() and nouveau.strip() not in list(criteres["Critère"]):
            vals = valeurs.copy()
            vals[nouveau.strip()] = np.nan
            ligne = {"Critère": nouveau.strip(), "Sens": "+", **({"Poids": 0.0} if avec_poids else {}), "Unité": ""}
            _remplacer_base(cle, {"valeurs": vals, "criteres": pd.concat([criteres, pd.DataFrame([ligne])], ignore_index=True)})
            st.rerun()
        if len(criteres) > 1:
            cible = c2.selectbox("Critère à supprimer", list(criteres["Critère"]), key=f"{cle}__cible_{v}")
            if c2.button("Supprimer", key=f"{cle}__supprimer_{v}"):
                j = list(criteres["Critère"]).index(cible)
                _remplacer_base(cle, {"valeurs": valeurs.drop(columns=valeurs.columns[j + 1]),
                                      "criteres": criteres.drop(index=criteres.index[j]).reset_index(drop=True)})
                st.rerun()
    if M is not None and avec_poids and abs(float(M.poids.sum()) - 1.0) > 1e-3:
        st.warning(f"La somme des poids vaut {M.poids.sum():.4f} au lieu de 1.")
    return M


def poids_valides(M: MatriceDecision, cle: str) -> np.ndarray | None:
    """Poids si Σ wⱼ = 1 ; sinon renormalisation proposée explicitement (jamais en silence)."""
    w = M.poids_ou_egaux()
    somme = float(w.sum())
    if abs(somme - 1.0) <= 1e-3:
        return w
    if somme > 0 and st.checkbox(f"Renormaliser les poids (diviser par {somme:.4f})", key=f"{cle}__renorm"):
        return w / somme
    st.error("La somme des poids doit valoir 1 : corrigez les poids ou cochez la renormalisation.")
    return None


# --------------------------------------------------------------------------
# Tableau générique
# --------------------------------------------------------------------------
def editeur_tableau(cle: str, defaut: pd.DataFrame, *, nom_fichier: str = "tableau.csv",
                    colonnes_attendues: list[str] | None = None, column_config: dict | None = None,
                    num_rows: str = "dynamic") -> pd.DataFrame:
    colonnes = colonnes_attendues or list(defaut.columns)

    def convertir(contenu: bytes) -> dict:
        entete, lignes = lire_tableau_csv(contenu, colonnes)
        return {"tableau": _typer(pd.DataFrame(lignes, columns=entete))[colonnes]}

    initiale = lambda: {"tableau": defaut.copy()}
    base = _base(cle, initiale)
    tableau = st.data_editor(base["tableau"], key=f"{cle}__tableau_{_version(cle)}", num_rows=num_rows,
                             hide_index=True, column_config=column_config)
    tableau = tableau.dropna(how="all").reset_index(drop=True)
    with st.expander("Importer ou exporter (CSV)"):
        st.caption(f"Colonnes attendues : {', '.join(colonnes)}")
        _import_et_reset(cle, convertir, initiale, "Importer un CSV")
        st.download_button("Exporter en CSV", tableau_vers_csv(tableau.columns, tableau.itertuples(index=False, name=None)),
                           file_name=nom_fichier, mime="text/csv", key=f"{cle}__export")
    return tableau


# --------------------------------------------------------------------------
# Matrice de comparaison par paires (AHP)
# --------------------------------------------------------------------------
def _base_comparaison(etiquettes, A) -> dict:
    df = pd.DataFrame([[format_nombre(x, fractions=True) for x in ligne] for ligne in np.asarray(A, float)],
                      columns=list(etiquettes))
    df.insert(0, "", list(etiquettes))
    return {"matrice": df}


def editeur_comparaison(cle: str, etiquettes, A_defaut, *, nom_fichier: str = "comparaison.csv"
                        ) -> tuple[tuple[str, ...], np.ndarray] | None:
    """Seul le triangle supérieur est lu ; le reste vaut 1/αᵢⱼ. Les fractions (« 1/3 ») sont acceptées."""
    initiale = lambda: _base_comparaison(etiquettes, A_defaut)
    base = _base(cle, initiale)
    v = _version(cle)
    df = st.data_editor(base["matrice"], key=f"{cle}__matrice_{v}", hide_index=True, num_rows="fixed", disabled=[""],
                        column_config={c: st.column_config.TextColumn(c) for c in base["matrice"].columns})
    noms = tuple(df[""])
    n = len(noms)
    A = np.full((n, n), np.nan)
    try:
        for i in range(n):
            for j in range(i + 1, n):
                A[i, j] = lire_nombre(df.iloc[i, j + 1])
        A = completer_reciproque(A)
    except ErreurDonnees as exc:
        st.error(f"Matrice invalide : {exc}")
        A = None
    st.caption("Seul le triangle supérieur est pris en compte ; le triangle inférieur vaut 1/αᵢⱼ.")
    with st.expander("Importer, exporter ou changer les éléments comparés"):
        _import_et_reset(cle, lambda c: _base_comparaison(*lire_matrice_comparaison_csv(c)), initiale, "Importer un CSV")
        if A is not None:
            st.download_button("Exporter en CSV", matrice_comparaison_vers_csv(noms, A), file_name=nom_fichier,
                               mime="text/csv", key=f"{cle}__export")
        liste = st.text_area("Éléments comparés (un par ligne)", "\n".join(noms), key=f"{cle}__noms_{v}")
        if st.button("Appliquer (matrice remise à 1)", key=f"{cle}__appliquer_{v}"):
            noms_l = [x.strip() for x in liste.splitlines() if x.strip()]
            if len(noms_l) >= 2 and len(set(noms_l)) == len(noms_l):
                _remplacer_base(cle, _base_comparaison(noms_l, np.ones((len(noms_l),) * 2)))
                st.rerun()
            st.error("Il faut au moins deux éléments, tous différents.")
    return None if A is None else (noms, A)
