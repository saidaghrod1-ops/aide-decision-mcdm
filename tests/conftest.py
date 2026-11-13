import pytest

from core.donnees import chemin, dataset, lire_matrice_csv


@pytest.fixture
def matrice():
    """Charge la matrice d'un dataset du catalogue."""
    def _charger(identifiant: str):
        return lire_matrice_csv(chemin(dataset(identifiant)["fichiers"]["matrice"]))
    return _charger
