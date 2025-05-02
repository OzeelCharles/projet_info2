import os
import pandas as pd
from pathlib import Path
import builtins

def charger_donnees_depuis_bureau():
    # Importe les chemins
    bureau = Path.home() / "Desktop"
    dossier_donnees = bureau / "donnees_formule_un"

    # Vérifie si le dossier existe
    if not dossier_donnees.exists():
        print(f"Dossier non trouvé : {dossier_donnees}")
        print("Veuillez télécharger le dossier 'donnees_formule_un' sur votre Bureau.")
        return None
    print(f"Dossier trouvé : {dossier_donnees}")
   
    # Cherche tous les fichiers CSV dans le dossier
    fichiers_csv = list(dossier_donnees.glob("*.csv"))
    print(f"Fichiers CSV trouvés : {fichiers_csv}")  # Ajouter cette ligne pour vérifier les fichiers trouvés
    if not fichiers_csv:
        print("Aucun fichier CSV trouvé dans le dossier.")
        return None
    
    # Charge chaque fichier CSV et crée une variable avec le nom du fichier
    print("Pandas est importé sous le nom pd.")
    for fichier in fichiers_csv:
        nom_variable = fichier.stem  # nom du fichier sans extension
        try:
            df = pd.read_csv(fichier)
            builtins.__dict__[nom_variable] = df
            print(f"Données chargées dans la variable : {nom_variable}")
        except Exception as e:
            print(f"Erreur lors de la lecture du fichier {fichier.name}: {e}")
