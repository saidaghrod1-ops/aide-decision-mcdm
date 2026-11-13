"""Éditeurs de données : st.data_editor + import CSV + export CSV + réinitialisation.

Principe : la « base » de chaque éditeur (DataFrames) est conservée dans st.session_state.
Les modifications faites dans st.data_editor sont relues à chaque exécution ; la base n'est
remplacée que lors d'un import, d'une réinitialisation ou d'un ajout/suppression de critère
(un numéro de version dans la clé du widget force alors son rafraîchissement).
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


def choisir_dataset(cle: str, options: dict[str, str], aide: str | None = None) -> str:
    """Liste déroulante « jeu de données » ; options = {libellé: identifiant du catalogue}."""
    libelle = st.selectbox("Jeu de données", list(options), key=f"{cle}__dataset", help=aide)
    d = dataset(options[libelle])
    st.caption(f"{'Cours' if d['origine'] == 'cours' else 'Jeu créé'} · p. {d['pages_cours']} — {d['contexte']}"
               + (f" *{d['notes']}*" if d.get("notes") else ""))
    return options[libelle]


def poids_valides(M: MatriceDecision, cle: str) -> np.ndarray | None:
    """Poids de la matrice si Σ wⱼ = 1 ; sinon propose explicitement de les renormaliser (jamais en silence)."""
    w = M.poids_ou_egaux()
    somme = float(w.sum())
    if abs(somme - 1.0) <= 1e-3:
        return w
    if somme <= 0:
        st.error("La somme des poids est nulle.")
        return None
    if st.checkbox(f"Renormaliser les poids (diviser par leur somme {somme:.4f})", key=f"{cle}__renorm"):
        return w / somme
    st.error("Le cours impose Σ wⱼ = 1 (p. 53) : corrigez les poids ou cochez la renormalisation.")
    return None


OPTIONS_MATRICES = {
    "Sélection d'un modèle de ML (jeu créé, p. 14)": "D1_selection_modele_ml",
    "Choisir un fournisseur parmi 5 offres (jeu créé)": "D2_fournisseurs_5offres",
    "Exercice WASPAS : 4 machines (p. 58)": "C4_waspas_machines",
    "Exercice TOPSIS : 4 marques de voiture (p. 65)": "C5_topsis_voitures",
    "Exercice VIKOR : 3 machines (p. 72)": "C6_vikor_machines",
    "Exercice CRITIC : machines de découpage (p. 51)": "C3_critic_machines_decoupe",
    "MCDM dynamique : prévisions à t + 1 (p. 94)": "C13_dynamique_prevision",
    "Portefeuille Vuković et al. (p. 18)": "C12_portefeuille_vukovic2020",
}


def options_matrices(premier: str) -> dict[str, str]:
    """Options de datasets « matrice » avec `premier` (identifiant) en tête."""
    ordre = sorted(OPTIONS_MATRICES.items(), key=lambda kv: kv[1] != premier)
    return dict(ordre)


@st.cache_data(show_spinner=False)
def charger_comparaison(relatif: str) -> tuple[tuple[str, ...], np.ndarray]:
    return lire_matrice_comparaison_csv(chemin(relatif))


def _base_comparaison(etiquettes, A) -> dict:
    df = pd.DataFrame([[format_nombre(x, fractions=True) for x in ligne] for ligne in np.asarray(A, float)],
                      columns=list(etiquettes))
    df.insert(0, "", list(etiquettes))
    return {"matrice": df}


def editeur_comparaison(cle: str, etiquettes, A_defaut, *, titre: str | None = None,
                        nom_fichier: str = "comparaison.csv") -> tuple[tuple[str, ...], np.ndarray] | None:
    """Matrice de comparaison par paires : seul le triangle supérieur est lu, le reste est complété
    par réciprocité (α_ji = 1/α_ij) ; les fractions « 1/3 » sont acceptées."""
    if f"{cle}__base" not in st.session_state:
        _remplacer_base(cle, _base_comparaison(etiquettes, A_defaut))
    base = st.session_state[f"{cle}__base"]
    v = _version(cle)
    if titre:
        st.markdown(f"**{titre}**")
    with st.expander("Importer un CSV, changer les éléments comparés ou réinitialiser", expanded=False):
        fichier = st.file_uploader("Importer une matrice carrée (CSV)", type=["csv"], key=f"{cle}__upload")
        _importer_si_nouveau(cle, fichier, lambda c: _base_comparaison(*lire_matrice_comparaison_csv(c)))
        if f"{cle}__erreur_import" in st.session_state:
            st.error(f"Import impossible : {st.session_state[f'{cle}__erreur_import']}")
        noms = st.text_area("Éléments comparés (un par ligne)", "\n".join(base["matrice"][""]), key=f"{cle}__noms_{v}")
        if st.button("Appliquer (matrice remise à 1)", key=f"{cle}__appliquer_{v}"):
            liste = [x.strip() for x in noms.splitlines() if x.strip()]
            if len(liste) >= 2 and len(set(liste)) == len(liste):
                _remplacer_base(cle, _base_comparaison(liste, np.ones((len(liste), len(liste)))))
                st.rerun()
            else:
                st.error("Il faut au moins deux éléments, tous différents.")
        if st.button("Réinitialiser avec les données par défaut", key=f"{cle}__reset_{v}"):
            st.session_state.pop(f"{cle}__empreinte", None)
            _remplacer_base(cle, _base_comparaison(etiquettes, A_defaut))
            st.rerun()
    df = st.data_editor(base["matrice"], key=f"{cle}__matrice_{v}", hide_index=True, num_rows="fixed",
                        disabled=[""], column_config={c: st.column_config.TextColumn(c) for c in base["matrice"].columns})
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
        return None
    st.caption("Seul le triangle supérieur est pris en compte ; le triangle inférieur vaut 1/αᵢⱼ.")
    st.download_button("Exporter la matrice (CSV)", matrice_comparaison_vers_csv(noms, A), file_name=nom_fichier,
                       mime="text/csv", key=f"{cle}__export")
    return noms, A


# --------------------------------------------------------------------------
# Chargement depuis le catalogue
# --------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def charger_matrice(identifiant: str, fichier: str = "matrice") -> MatriceDecision:
    """Matrice de décision d'un dataset du catalogue (mise en cache)."""
    d = dataset(identifiant)
    return lire_matrice_csv(chemin(d["fichiers"][fichier]))


@st.cache_data(show_spinner=False)
def charger_tableau(identifiant: str, fichier: str) -> pd.DataFrame:
    """Tableau générique d'un dataset du catalogue (colonnes numériques converties)."""
    d = dataset(identifiant)
    entete, lignes = lire_tableau_csv(chemin(d["fichiers"][fichier]))
    return _typer(pd.DataFrame(lignes, columns=entete))


def _typer(df: pd.DataFrame) -> pd.DataFrame:
    """Convertit en nombres les colonnes entièrement numériques."""
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


def _importer_si_nouveau(cle: str, fichier, convertir) -> None:
    """Importe un fichier téléversé une seule fois (empreinte mémorisée)."""
    if fichier is None:
        return
    contenu = fichier.getvalue()
    empreinte = hashlib.sha256(contenu).hexdigest()
    if st.session_state.get(f"{cle}__empreinte") == empreinte:
        return
    st.session_state[f"{cle}__empreinte"] = empreinte
    try:
        _remplacer_base(cle, convertir(contenu))
    except ErreurDonnees as exc:
        st.session_state[f"{cle}__erreur_import"] = str(exc)
    else:
        st.session_state.pop(f"{cle}__erreur_import", None)
        st.rerun()


# --------------------------------------------------------------------------
# Éditeur de matrice de décision
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
    if any(a == "" for a in alternatives):
        raise ErreurDonnees("chaque alternative doit avoir un nom")
    noms = tuple("" if pd.isna(c) else str(c).strip() for c in criteres["Critère"])
    if any(c == "" for c in noms):
        raise ErreurDonnees("chaque critère doit avoir un nom")
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
                    titre: str = "Matrice de décision", nom_fichier: str = "matrice.csv") -> MatriceDecision | None:
    """Affiche l'éditeur complet d'une matrice de décision et retourne la matrice valide (ou None)."""
    if f"{cle}__base" not in st.session_state:
        _remplacer_base(cle, _base_matrice(defaut, avec_poids))
    base = st.session_state[f"{cle}__base"]
    v = _version(cle)

    st.subheader(titre)
    with st.expander("Importer un CSV ou réinitialiser", expanded=False):
        st.caption("Format : 1re colonne = alternative ; lignes facultatives @sens (+/-), @poids, @unite. "
                   "Voir la page d'accueil pour un exemple.")
        fichier = st.file_uploader("Importer une matrice (CSV)", type=["csv"], key=f"{cle}__upload")
        _importer_si_nouveau(cle, fichier,
                             lambda contenu: _base_matrice(lire_matrice_csv(contenu), avec_poids))
        if f"{cle}__erreur_import" in st.session_state:
            st.error(f"Import impossible : {st.session_state[f'{cle}__erreur_import']}")
        if st.button("Réinitialiser avec les données par défaut", key=f"{cle}__reset_{v}"):
            st.session_state.pop(f"{cle}__empreinte", None)
            _remplacer_base(cle, _base_matrice(defaut, avec_poids))
            st.rerun()

    valeurs = st.data_editor(
        base["valeurs"], key=f"{cle}__valeurs_{v}", num_rows="dynamic", hide_index=True,
        column_config={"Alternative": st.column_config.TextColumn("Alternative", required=True)},
    )
    st.markdown("**Critères** — sens : « + » bénéfice (à maximiser), « - » coût (à minimiser)")
    config = {
        "Critère": st.column_config.TextColumn("Critère", required=True),
        "Sens": st.column_config.SelectboxColumn("Sens", options=["+", "-"], required=True),
        "Unité": st.column_config.TextColumn("Unité"),
    }
    if avec_poids:
        config["Poids"] = st.column_config.NumberColumn("Poids", min_value=0.0, format="%.4f")
    criteres = st.data_editor(base["criteres"], key=f"{cle}__criteres_{v}", num_rows="fixed",
                              hide_index=True, column_config=config)

    with st.expander("Ajouter ou supprimer un critère", expanded=False):
        c1, c2 = st.columns(2)
        with c1:
            nouveau = st.text_input("Nom du nouveau critère", key=f"{cle}__nouveau_{v}")
            if st.button("Ajouter le critère", key=f"{cle}__ajouter_{v}") and nouveau.strip():
                if nouveau.strip() in list(criteres["Critère"]):
                    st.error("Ce critère existe déjà.")
                else:
                    vals = valeurs.copy()
                    vals[nouveau.strip()] = np.nan
                    crit = pd.concat([criteres, pd.DataFrame([{
                        "Critère": nouveau.strip(), "Sens": "+", **({"Poids": 0.0} if avec_poids else {}), "Unité": ""}])],
                        ignore_index=True)
                    _remplacer_base(cle, {"valeurs": vals, "criteres": crit})
                    st.rerun()
        with c2:
            if len(criteres) > 1:
                cible = st.selectbox("Critère à supprimer", list(criteres["Critère"]), key=f"{cle}__cible_{v}")
                if st.button("Supprimer le critère", key=f"{cle}__supprimer_{v}"):
                    j = list(criteres["Critère"]).index(cible)
                    vals = valeurs.drop(columns=valeurs.columns[j + 1])
                    crit = criteres.drop(index=criteres.index[j]).reset_index(drop=True)
                    _remplacer_base(cle, {"valeurs": vals, "criteres": crit})
                    st.rerun()

    try:
        M = _construire_matrice(valeurs, criteres, avec_poids)
    except ErreurDonnees as exc:
        st.error(f"Données invalides : {exc}")
        return None

    if avec_poids:
        somme = float(M.poids.sum())
        if abs(somme - 1.0) > 1e-3:
            st.warning(f"La somme des poids vaut {somme:.4f} au lieu de 1 (le cours impose Σ wⱼ = 1, p. 53).")
    st.download_button("Exporter la matrice (CSV)", matrice_vers_csv(M, inclure_poids=avec_poids),
                       file_name=nom_fichier, mime="text/csv", key=f"{cle}__export")
    return M


# --------------------------------------------------------------------------
# Éditeur de tableau générique (rangs, scores, évaluations linguistiques…)
# --------------------------------------------------------------------------
def editeur_tableau(cle: str, defaut: pd.DataFrame, *, titre: str | None = None, nom_fichier: str = "tableau.csv",
                    colonnes_attendues: list[str] | None = None, column_config: dict | None = None,
                    num_rows: str = "dynamic") -> pd.DataFrame:
    """Éditeur d'un tableau quelconque avec import/export CSV ; retourne le tableau édité."""
    colonnes = colonnes_attendues or list(defaut.columns)

    def convertir(contenu: bytes) -> dict:
        entete, lignes = lire_tableau_csv(contenu, colonnes)
        return {"tableau": _typer(pd.DataFrame(lignes, columns=entete))[colonnes]}

    if f"{cle}__base" not in st.session_state:
        _remplacer_base(cle, {"tableau": defaut.copy()})
    base = st.session_state[f"{cle}__base"]
    v = _version(cle)

    if titre:
        st.subheader(titre)
    with st.expander("Importer un CSV ou réinitialiser", expanded=False):
        st.caption(f"Colonnes attendues : {', '.join(colonnes)}")
        fichier = st.file_uploader("Importer un tableau (CSV)", type=["csv"], key=f"{cle}__upload")
        _importer_si_nouveau(cle, fichier, convertir)
        if f"{cle}__erreur_import" in st.session_state:
            st.error(f"Import impossible : {st.session_state[f'{cle}__erreur_import']}")
        if st.button("Réinitialiser avec les données par défaut", key=f"{cle}__reset_{v}"):
            st.session_state.pop(f"{cle}__empreinte", None)
            _remplacer_base(cle, {"tableau": defaut.copy()})
            st.rerun()

    tableau = st.data_editor(base["tableau"], key=f"{cle}__tableau_{v}", num_rows=num_rows,
                             hide_index=True, column_config=column_config)
    tableau = tableau.dropna(how="all").reset_index(drop=True)
    st.download_button("Exporter le tableau (CSV)",
                       tableau_vers_csv(tableau.columns, tableau.itertuples(index=False, name=None)),
                       file_name=nom_fichier, mime="text/csv", key=f"{cle}__export")
    return tableau
