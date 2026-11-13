"""Chaque page s'exécute sans exception, pour chacun de ses jeux de données (streamlit.testing)."""

import re
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

RACINE = Path(__file__).resolve().parent.parent
PAGES = sorted((RACINE / "pages").glob("*.py"))
DELAI = 120


def _lancer(page: Path) -> AppTest:
    return AppTest.from_file(str(page), default_timeout=DELAI).run()


def _selecteur_dataset(at: AppTest):
    return next((s for s in at.selectbox if s.label == "Jeu de données"), None)


def test_navigation_reference_toutes_les_pages():
    source = (RACINE / "streamlit_app.py").read_text(encoding="utf-8")
    referencees = set(re.findall(r'"(pages/[\w]+\.py)"', source))
    assert referencees == {f"pages/{p.name}" for p in PAGES}


def test_application_demarre_sur_l_accueil():
    at = AppTest.from_file(str(RACINE / "streamlit_app.py"), default_timeout=DELAI).run()
    assert not at.exception, at.exception
    assert at.title[0].value.startswith("Aide à la décision multicritère")


@pytest.mark.parametrize("page", [p for p in PAGES if p.name != "accueil.py"], ids=lambda p: p.stem)
def test_page_sans_exception_ni_erreur_avec_les_donnees_par_defaut(page):
    at = _lancer(page)
    assert not at.exception, at.exception
    assert not at.error, [e.value for e in at.error]
    assert "en construction" not in " ".join(i.value for i in at.info)


@pytest.mark.parametrize("page", [p for p in PAGES if p.name != "accueil.py"], ids=lambda p: p.stem)
def test_page_avec_chaque_jeu_de_donnees(page):
    at = _lancer(page)
    selecteur = _selecteur_dataset(at)
    if selecteur is None:
        pytest.skip("page sans sélecteur de jeu de données")
    for option in selecteur.options[1:]:
        at = _lancer(page)
        _selecteur_dataset(at).set_value(option).run()
        assert not at.exception, (option, at.exception)


def _textes(at: AppTest) -> str:
    morceaux = [m.value for m in at.markdown] + [s.value for s in at.success] + [w.value for w in at.warning]
    morceaux += [str(m.value) for m in at.metric] + [c.value for c in at.caption]
    return "\n".join(morceaux)


def test_aucun_numero_de_page_affiche():
    for page in PAGES:
        if page.name == "accueil.py":
            at = AppTest.from_file(str(RACINE / "streamlit_app.py"), default_timeout=DELAI).run()
        else:
            at = _lancer(page)
        assert not re.search(r"\bp\. ?\d", _textes(at) + " ".join(t.value for t in at.title)), page.name


def test_ahp_affiche_le_cr_de_l_exercice():
    at = _lancer(RACINE / "pages" / "ponderation_ahp.py")
    assert "0.0093" in [m.value for m in at.metric]
    assert "Voiture 1" in _textes(at)


def test_bwm_affiche_xi_et_l_ecart_avec_les_poids_imprimes():
    at = _lancer(RACINE / "pages" / "ponderation_bwm.py")
    assert "0.0838" in _textes(at) and "0.552" in _textes(at)


def test_vikor_designe_a2():
    at = _lancer(RACINE / "pages" / "classement_vikor.py")
    assert "Solution de compromis : **A2**" in _textes(at)


def test_topsis_designe_m2():
    at = _lancer(RACINE / "pages" / "classement_topsis.py")
    assert "**M2** a le RC* le plus élevé (0.5678)" in _textes(at)


def test_seuil_reproduit_le_tableau_p88():
    at = _lancer(RACINE / "pages" / "unique_maut_seuil.py")
    tableau = next(df.value for df in at.dataframe if "TPR" in df.value.columns and "Cas" in df.value.columns)
    assert list(tableau["TPR"]) == [0.67, 1.0, 1.0] and list(tableau["J"]) == [0.67, 0.67, 0.33]


def test_flou_unique_reproduit_p134():
    at = _lancer(RACINE / "pages" / "mcdm_flou_unique.py")
    assert "0.8626" in [m.value for m in at.metric]
