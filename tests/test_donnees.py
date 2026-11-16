"""Jalon J0 : catalogue, transcription des datasets du cours, format CSV (aller-retour)."""

import json

import numpy as np
import pytest

from core.donnees import (
    ErreurDonnees,
    MatriceDecision,
    chemin,
    charger_catalogue,
    completer_reciproque,
    dataset,
    ecart_reciprocite,
    fichiers_du_dataset,
    lire_matrice_comparaison_csv,
    lire_matrice_csv,
    lire_nombre,
    lire_tableau_csv,
    matrice_comparaison_vers_csv,
    matrice_vers_csv,
)


# ---------------------------------------------------------------- catalogue
def test_catalogue_identifiants_uniques_et_fichiers_presents():
    catalogue = charger_catalogue()
    ids = [d["id"] for d in catalogue]
    assert len(ids) == len(set(ids))
    for d in catalogue:
        for relatif in fichiers_du_dataset(d):
            assert chemin(relatif).is_file(), f"{d['id']} : fichier absent {relatif}"


@pytest.mark.parametrize("d", [d for d in charger_catalogue() if d["type"] == "matrice"], ids=lambda d: d["id"])
def test_matrices_du_catalogue_se_chargent(d):
    M = lire_matrice_csv(chemin(d["fichiers"]["matrice"]))
    assert M.m >= 2 and M.n >= 2


# ------------------------------------------------- transcription du cours
def test_ahp_voiture_p32_33_sommes_de_colonnes():
    d = dataset("C1_ahp_voiture")
    _, A = lire_matrice_comparaison_csv(chemin(d["fichiers"]["criteres"]))
    np.testing.assert_allclose(A, [[1, 7, 3], [1 / 7, 1, 1 / 3], [1 / 3, 3, 1]])
    # Sommes de colonnes imprimées p. 33 (3 décimales -> tolérance 0,0005)
    attendues = {"Coût": [1.143, 8], "Confort": [6, 1.2], "Sécurité": [10, 1.111]}
    for critere, sommes in attendues.items():
        _, B = lire_matrice_comparaison_csv(chemin(d["fichiers"]["alternatives"][critere]))
        assert ecart_reciprocite(B) < 1e-12
        np.testing.assert_allclose(B.sum(axis=0), sommes, atol=5e-4)


def test_bwm_fournisseur_p37_controles_du_cours():
    d = dataset("C2_bwm_fournisseur")
    entete, lignes = lire_tableau_csv(chemin(d["fichiers"]["vecteurs"]), ["critere", "BO", "OW"])
    criteres = [l[0] for l in lignes]
    bo = [lire_nombre(l[1]) for l in lignes]
    ow = [lire_nombre(l[2]) for l in lignes]
    assert bo == [1, 3, 4, 8] and ow == [8, 5, 3, 1]
    b, w = criteres.index(d["best"]), criteres.index(d["worst"])
    assert bo[b] == 1 and ow[w] == 1                    # a_BB = a_WW = 1 (p. 36)
    assert criteres[int(np.argmax(bo))] == d["worst"]   # note maximale de BO sur W (p. 36)


@pytest.mark.parametrize("identifiant, valeurs, sens, poids", [
    ("C3_critic_machines_decoupe",
     [[30, 0.1, 1, 20], [100, 0.7, 1, 40], [50, 1, 2, 10], [300, 2, 3, 35]], [-1, -1, 1, -1], None),
    ("C4_waspas_machines",
     [[0.035, 847, 0.335, 1.760, 0.590], [0.027, 834, 0.335, 1.680, 0.665],
      [0.037, 808, 0.590, 2.400, 0.500], [0.028, 821, 0.500, 1.590, 0.410]],
     [-1, -1, 1, -1, 1], [0.331, 0.181, 0.369, 0.072, 0.047]),
    ("C5_topsis_voitures",
     [[7, 9, 9, 8], [8, 7, 8, 7], [9, 6, 8, 9], [6, 7, 8, 6]], [1, 1, 1, -1], [0.2, 0.1, 0.4, 0.3]),
    ("C6_vikor_machines", [[1, 3000], [2, 3750], [5, 4500]], [-1, 1], [0.5, 0.5]),
])
def test_matrices_du_cours_transcrites_exactement(identifiant, valeurs, sens, poids):
    M = lire_matrice_csv(chemin(dataset(identifiant)["fichiers"]["matrice"]))
    np.testing.assert_array_equal(M.valeurs, valeurs)
    np.testing.assert_array_equal(M.sens, sens)
    if poids is None:
        assert M.poids is None
    else:
        np.testing.assert_array_equal(M.poids, poids)


def test_portefeuille_vukovic_p18():
    M = lire_matrice_csv(chemin(dataset("C12_portefeuille_vukovic2020")["fichiers"]["matrice"]))
    assert (M.m, M.n) == (18, 9)
    np.testing.assert_array_equal(M.sens, [1, -1, 1, -1, 1, 1, -1, -1, 1])   # max/min de la p. 18
    np.testing.assert_allclose(M.poids.sum(), 0.9999, atol=1e-12)           # somme imprimée : 0,9999
    assert M.valeurs[13, 5] == -303.61 and M.alternatives[13] == "A14 OPTE"
    assert (M.valeurs < 0).any()


def test_spearman_rangs_p78():
    _, lignes = lire_tableau_csv(chemin(dataset("C7_spearman_rangs")["fichiers"]["rangs"]))
    rangs = np.array([[int(x) for x in l[1:]] for l in lignes])
    np.testing.assert_array_equal(rangs.T, [[2, 1, 3, 5, 4, 6], [1, 2, 4, 3, 5, 6], [2, 1, 3, 6, 4, 5]])


def test_decisions_passees_p85():
    _, lignes = lire_tableau_csv(chemin(dataset("C8_decisions_passees_6")["fichiers"]["tableau"]))
    assert [float(l[1]) for l in lignes] == [0.85, 0.78, 0.71, 0.69, 0.62, 0.54]
    assert [l[2] for l in lignes] == ["Accepter"] * 3 + ["Rejeter"] * 3


def test_chen2000_tableaux_1_2_3():
    d = dataset("C10_chen2000_recrutement")
    f = d["fichiers"]
    for cle in ("echelle_alternatives", "echelle_importance"):
        _, lignes = lire_tableau_csv(chemin(f[cle]), ["terme", "code", "a", "b", "c"])
        assert len(lignes) == 7
        for l in lignes:
            a, b, c = (lire_nombre(x) for x in l[2:])
            assert a <= b <= c                       # nombres flous triangulaires valides
    codes_alt = {l[1] for l in lire_tableau_csv(chemin(f["echelle_alternatives"]))[1]}
    codes_imp = {l[1] for l in lire_tableau_csv(chemin(f["echelle_importance"]))[1]}
    _, evals = lire_tableau_csv(chemin(f["evaluations"]), ["alternative", "decideur", "critere", "terme"])
    _, poids = lire_tableau_csv(chemin(f["poids"]), ["decideur", "critere", "terme"])
    assert len(evals) == 3 * 3 * 5 and len(poids) == 3 * 5
    assert {l[3] for l in evals} <= codes_alt and {l[2] for l in poids} <= codes_imp
    # Deux contrôles ponctuels du Tableau 2 et du Tableau 3 (p. 141)
    assert ["A3", "DM3", "C1 Stabilité émotionnelle", "F"] in evals
    assert ["DM1", "C5 Confiance en soi", "M"] in poids


def test_exemples_logique_floue_json():
    with chemin(dataset("C14_ensembles_flous_exemples")["fichiers"]["json"]).open(encoding="utf-8") as fh:
        ex = json.load(fh)
    assert ex["age"]["ensembles"]["Âge moyen"]["parametres"] == [25, 35, 55, 65]
    assert ex["average"]["parametres"] == [10, 15, 20]
    assert ex["moyenne_etudiant"]["parametres"] == [4, 5, 6, 7]
    v = ex["vote_intuitionniste"]
    assert v["pour"] + v["contre"] + v["absents"] == 10


# ------------------------------------------------------------ format CSV
def test_lire_nombre_fractions_et_erreurs():
    assert lire_nombre("1/7") == pytest.approx(1 / 7)
    assert lire_nombre(" −3 ") == -3
    for mauvais in ("", "abc", "1/0"):
        with pytest.raises(ErreurDonnees):
            lire_nombre(mauvais)


@pytest.mark.parametrize("identifiant", ["C4_waspas_machines", "C5_topsis_voitures", "C12_portefeuille_vukovic2020"])
def test_aller_retour_csv(identifiant):
    M = lire_matrice_csv(chemin(dataset(identifiant)["fichiers"]["matrice"]))
    N = lire_matrice_csv(matrice_vers_csv(M))
    assert N.alternatives == M.alternatives and N.criteres == M.criteres
    np.testing.assert_array_equal(N.valeurs, M.valeurs)
    np.testing.assert_array_equal(N.sens, M.sens)
    np.testing.assert_array_equal(N.poids, M.poids)


def test_valeurs_par_defaut_sens_et_poids():
    M = lire_matrice_csv("alternative,C1,C2\nA1,1,2\nA2,3,4\n")
    np.testing.assert_array_equal(M.sens, [1, 1])
    assert M.poids is None
    np.testing.assert_allclose(M.poids_ou_egaux(), [0.5, 0.5])


@pytest.mark.parametrize("texte, message", [
    ("alternative,C1,C2\nA1,1\n", "cellules"),
    ("alternative,C1\n@sens,x\nA1,1\n", "sens"),
    ("alternative,C1\nA1,1\nA1,2\n", "même nom"),
    ("alternative,C1\n@poids,-1\nA1,1\n", "poids"),
    ("alternative,C1\n@autre,1\nA1,1\n", "inconnue"),
])
def test_erreurs_de_format_en_francais(texte, message):
    with pytest.raises(ErreurDonnees, match=message):
        lire_matrice_csv(texte)


def test_matrice_comparaison_triangle_superieur_complete():
    etiquettes, A = lire_matrice_comparaison_csv(",C1,C2,C3\nC1,1,7,3\nC2,,1,1/3\nC3,,,1\n")
    assert etiquettes == ("C1", "C2", "C3")
    np.testing.assert_allclose(A, [[1, 7, 3], [1 / 7, 1, 1 / 3], [1 / 3, 3, 1]])
    E, B = lire_matrice_comparaison_csv(matrice_comparaison_vers_csv(etiquettes, A))
    np.testing.assert_allclose(B, A)
    assert "1/7" in matrice_comparaison_vers_csv(etiquettes, A)


def test_matrice_comparaison_cases_vides_symetriques_refusees():
    with pytest.raises(ErreurDonnees, match="vides"):
        completer_reciproque(np.array([[1, np.nan], [np.nan, 1]]))


def test_matrice_decision_refuse_valeur_manquante():
    with pytest.raises(ErreurDonnees, match="manquantes"):
        MatriceDecision(("A1",), ("C1",), np.array([[np.nan]]), np.array([1]))
