import os
import pandas as pd
from pathlib import Path


def charger_donnees_depuis_bureau():
    # Importe les chemins #
    bureau = Path.home() / "Desktop"
    dossier_donnees = bureau / "donnees_formule_un"

    # Vérifie si le dossier existe #
    if not dossier_donnees.exists():
        print(f"Dossier non trouvé : {dossier_donnees}")
        print("Veuillez télécharger le dossier 'donnees_formule_un' sur votre Bureau.")
        return

    print(f" Dossier trouvé : {dossier_donnees}")