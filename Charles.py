#Questions de Charles 
# coding: utf-8

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors
from src import import_data as i_d
from sklearn.linear_model import LinearRegression
import scipy.stats as stats

#Charges les données depuis le bureau#

i_d.charger_donnees_depuis_bureau()


def time_to_decimal(time_str: str) -> float:
    """
    Convertit une durée au format 'HH:MM:SS' en heures sous forme décimale.

    Cette fonction prend en entrée une chaîne de caractères représentant une durée
    au format 'HH:MM:SS', où `HH` représente les heures, `MM` les minutes, et `SS` les secondes.
    Elle retourne la durée totale en heures sous forme décimale.

    Arguments:
         time_str : str
                    La chaîne de caractères représentant la durée, au format 'HH:MM:SS'.

    Sortie:
        float: La durée convertie en heures sous forme décimale.

    Exceptions:
        ValueError
        Si la chaîne d'entrée n'est pas au format attendu 'HH:MM:SS'.

    Exemples:
        >>> time_to_decimal("2:30:45")
        2.5125
        >>> time_to_decimal("1:05:10")
        1.0861

    Remarques:
    - Si l'argument `time_str` est `NaN`, la fonction retournera `NaN`.
    - Cette fonction suppose que l'entrée est bien formatée sous la forme 'HH:MM:SS'.
    """
    if pd.isna(time_str):
        return np.nan
    if not map(int, time_str.split(":")):
        raise ValueError("La chaîne d'entrée n'est pas au format attendu 'HH:MM:SS'.")
    h, m, s = map(int, time_str.split(":"))
    return h + m / 60 + s / 3600


def horaire_moyen_run_f1(races: pd.DataFrame):
    """
    Calcule et affiche les horaires moyens de départ des différentes sessions de la course
    pour chaque emplacement géographique des Grands Prix, en fonction des longitudes des circuits.

    Cette fonction prend en entrée un DataFrame contenant des informations sur les courses de F1,
    et effectue plusieurs étapes de transformation et de nettoyage des données, notamment :
    - Vérification que les colonnes nécessaires sont présentes.
    - Conversion des colonnes de date et de temps au bon format.
    - Calcul des horaires moyens pour chaque longitude unique.
    - Visualisation géographique des résultats sous forme de carte.

    Arguments:
         races : pd.DataFrame
                 Un DataFrame contenant des informations 
                 sur les courses de F1, avec les colonnes suivantes
                 au minimum :
                 - 'raceId', 'year', 'round', 'circuitId', 'name', 'date', 'time',
                 - 'fp1_date', 'fp1_time', 'fp2_date', 'fp2_time', 'fp3_date', 'fp3_time',
                 - 'quali_date', 'quali_time', 'sprint_date', 'sprint_time'.
        
                 Le DataFrame doit avoir des valeurs de type `datetime` pour 
                 les colonnes de date et des chaînes
                 de caractères de type 'HH:MM:SS' pour les colonnes de temps.

    Exceptions:
       ValueError:
              Si les colonnes nécessaires ne sont pas présentes dans le DataFrame `races`.
              Si une des colonnes de dates ou de temps n'est pas au format attendu.
              Si le fichier 'f1_grands_prix_locations.csv' n'est pas disponible.

    Examples:
         >>> horaire_moyen_run_f1(races_df)
         Affiche une carte avec les horaires moyens pour chaque longitude des Grands Prix.

    Remarques:
    - La fonction utilise un fichier CSV externe ('f1_grands_prix_locations.csv') pour récupérer les
      informations de longitude et latitude des circuits.
    - La fonction effectue des conversions sur les colonnes de temps et de date en utilisant `pd.to_datetime`
      et une fonction auxiliaire `time_to_decimal`.
    - Si le format des colonnes n'est pas valide, une exception `ValueError` est levée.
    - La visualisation géographique est générée à l'aide de GeoPandas et Matplotlib.
    """
    import geopandas as gpd
    import os

    # Conditions initiales. Il faut que la table en entrée possède ces colonnes#
    if set(races.columns) in {
        "raceId",
        "year",
        "round",
        "circuitId",
        "name",
        "date",
        "time",
        "fp1_date",
        "fp1_time",
        "fp2_date",
        "fp2_time",
        "fp3_date",
        "fp3_time",
        "quali_date",
        "quali_time",
        "sprint_date",
        "sprint_time",
    }:
        raise ValueError(
            "la table ne reste pase les normes attendus."
            "la table doit au moins contenir les colonnes"
            "suivantes: [name',"
            "'date',"
            "'time',"
            "'fp1_date',"
            "'fp1_time',"
            "'fp2_date',"
            "'fp2_time',"
            "'fp3_date',"
            "'fp3_time',"
            "'quali_date',"
            "'quali_time',"
            "'sprint_date',"
            "'sprint_time']"
        )
    races = races.copy(deep=True)

    date = ["date", "fp1_date", "fp2_date", "fp3_date", "quali_date", "sprint_date"]

    time = ["time", "fp1_time", "fp2_time", "fp3_time", "quali_time", "sprint_time"]

    # Ici on remplace les \\N, valeurs manquantes, par des NaN#
    races.replace("\\N", np.nan, inplace=True)

    for _ in date:
        # Ici si la colonne n'est pas en date_time mais en str (donc convertible)#
        if races[_].dtype == object:

            # On vérifie alors qu'en cas de conversion, si jamais la conversion rend des NaN partout#
            # Alors c'est que le format n'est pas respecté #
            if pd.to_datetime(races[_], "coerce").isna().all():
                raise ValueError(
                    f"La colonne {_} n'est pas au bon format date time."
                    " Il faut que ça soit sous la forme 'YYYY-MM-DD HH:MM:SS'."
                )

        races[_] = pd.to_datetime(races[_])

    for _ in time:
        # même chose qu'au dessus#
        if races[_].dtype == object:
            if races[_].apply(time_to_decimal, "coerce").isna().all():
                raise ValueError(
                    f"{_} n'est pas au bon format."
                    "Elle doit être sous la forme 'HH:MM:SS'."
                )

        races[_] = races[_].apply(time_to_decimal)
    # ici si la table loca est présente on l'importe. #
    if not os.path.exists("f1_grands_prix_locations.csv"):
        raise ValueError("Le fichier 'f1_grands_prix_locations.csv' n'est pas présent.")

    loca = pd.read_csv("f1_grands_prix_locations.csv").copy(deep=True)

    loca.rename(columns={"Grand Prix": "name"}, inplace=True)

    races = pd.merge(races, loca, how="left")
    Long = races.Longitude.unique()
    # On s'assure de l'unicité des valeurs dans la table loca colonne Longitude#
    avg = []
    # Ici pour tout éléments dans dans les longitudes#
    # on sélectionne les valeurs qui ont cette position#
    # On calcul pour chaque sous table la moyenne de temps par course#
    for l in Long:
        a = races[races["Longitude"] == l]
        avg_time_A = [a[_].mean() for _ in time]
        avg.append(avg_time_A)

    # Ici on place le point sur une map géo isssus de géopandas#
    for j in range(len(avg[0])):
        data = [avg[i][j] for i in range(len(avg))]
        norm = mcolors.Normalize(
            vmin=min(data), vmax=max(data)
        )  # Normalisation des valeurs
        cmap = plt.cm.plasma
        shapefile_path = "naturalearth_lowres"
        gdf = gpd.read_file(shapefile_path)
        fig, ax = plt.subplots(figsize=(10, 8))
        gdf.plot(ax=ax)
        scatter = ax.scatter(
            races["Longitude"].unique(),
            races["Latitude"].unique(),
            c=data,
            cmap=cmap,
            s=50,
            marker="o",
            edgecolor="k",
        )
        fig.colorbar(
            cm.ScalarMappable(norm=norm, cmap=cmap),
            ax=ax,
            orientation="horizontal",
            label="Moyenne des temps",
        )
        fig.suptitle(f"Horaire moyen de début de {time[j]}")
        plt.show()


def min_max_pit_stop_drivers(
    nom_pilote: str,
    pit_stops: pd.DataFrame,
    drivers: pd.DataFrame,
    param: {"min", "max"},
):
    """
    Retourne la moyenne des temps d'arrêt aux stands (pit stops) minimum ou maximum 
    pour un pilote donné, calculée sur l'ensemble des courses qu'il a joué.

    Arguments:
         nom_pilote : str
                      Identifiant du pilote 
                      (champ `driverRef` dans le DataFrame `drivers`).
    
         pit_stops : pd.DataFrame
                     DataFrame contenant les informations sur les arrêts aux stands. 
                     Doit inclure les colonnes : `raceId`, `driverId`, `milliseconds`.
    
         drivers : pd.DataFrame
                   DataFrame contenant les informations sur les pilotes.
                   Doit inclure les colonnes : `driverId`, `driverRef`.

         param : {'min', 'max'}
                 Spécifie le type de temps d'arrêt à retourner :
                 - 'min' : temps d'arrêt minimum moyen par course
                 - 'max' : temps d'arrêt maximum moyen par course
        
         Sortie:
             pd.DataFrame: Un DataFrame avec une seule ligne contenant la moyenne 
                           des temps d'arrêt correspondants (min ou max) pour 
                           le pilote spécifié. La colonne retournée sera `min_pit_stop`
                           ou `max_pit_stop` selon la valeur de `param`.

    Remarques:
    - Les temps d'arrêt sont convertis de millisecondes en secondes.
    - Seules les courses où des données valides existent pour le pilote sont prises en compte.
    """
    pit_stops = pit_stops.copy(deep=True)
    pit_stops["duration"] = pit_stops["milliseconds"] * (10 ** -3)
    pit_stop_min = (
        pit_stops.groupby(["raceId", "driverId"]).agg({"duration": "min"}).reset_index()
    )
    pit_stop_max = (
        pit_stops.groupby(["raceId", "driverId"]).agg({"duration": "max"}).reset_index()
    )
    a = pd.merge(drivers, pit_stop_min, on="driverId", how="left")
    a.rename(columns={"duration": "min_pit_stop"}, inplace=True)
    a = pd.merge(a, pit_stop_max, on=["driverId", "raceId"], how="left")
    a.rename(columns={"duration": "max_pit_stop"}, inplace=True)
    a["min_pit_stop"] = a["min_pit_stop"].astype(float)
    a["max_pit_stop"] = a["max_pit_stop"].astype(float)
    a = a[~a["min_pit_stop"].isna()]
    a = a[~a["max_pit_stop"].isna()]
    drivers_mean_min_pit_stop = (
        a.groupby("driverRef").agg({"min_pit_stop": "mean"}).reset_index()
    )
    drivers_mean_max_pit_stop = (
        a.groupby("driverRef").agg({"max_pit_stop": "mean"}).reset_index()
    )
    if param == "min":
        return drivers_mean_min_pit_stop[
            drivers_mean_min_pit_stop["driverRef"] == nom_pilote
        ]
    else:
        return drivers_mean_max_pit_stop[
            drivers_mean_max_pit_stop["driverRef"] == nom_pilote
        ]

def generer_table_fichier(nom_fichier_recherche):
    """
    Génère ligne par ligne le contenu d’un fichier texte situé dans un dossier spécifique sur le bureau.

    Le fichier recherché doit se trouver dans le dossier :
    ~/Desktop/donnees_formule_un/

    Argument:
          nom_fichier_recherche : str
                                  Le nom du fichier (sans extension) à 
                                  rechercher dans le dossier.

    Yields
    ------
    str
        Chaque ligne du fichier, sans les sauts de ligne finaux.

    Exceptions:
        ValueError :
              Si le dossier 'donnees_formule_un'
              n'existe pas sur le bureau de l'utilisateur.

    Remarques:
    - Le fichier doit être un fichier texte (.txt, .csv, etc.).
    - La recherche du fichier se fait en comparant le nom sans extension.
    """
    import os

    dossier = os.path.join(os.path.expanduser("~"), "Desktop", "donnees_formule_un")
    if not os.path.isdir(dossier):
        raise ValueError(
            "Le fichier 'donnees_formule_un' est introuvable sur votre bureau."
        )
        print("Le dossier n'existe pas :", dossier)
        return
    for nom_fichier in os.listdir(dossier):
        chemin_fichier = os.path.join(dossier, nom_fichier)
        if os.path.isfile(chemin_fichier):
            nom_sans_ext = os.path.splitext(nom_fichier)[0]
            if nom_sans_ext == nom_fichier_recherche:
                with open(chemin_fichier, "r", encoding="utf-8") as f:
                    for ligne in f:
                        yield ligne.strip()


def nbr_victoire_joueurs():
    """
    Calcule le nombre total de victoires pour chaque pilote à partir du fichier 'driver_standings'.

    Le fichier est lu ligne par ligne à l'aide de la fonction `generer_table_fichier`.  
    La fonction extrait le nom ou l'identifiant du pilote (clé) ainsi que le nombre de victoires associé
    à chaque ligne, puis cumule les victoires dans un dictionnaire.

    Sortie:
        dict: Un dictionnaire où les clés sont les identifiants des pilotes
              et les valeurs sont le total de leurs victoires.

    Remarques:
    - Ignore la première ligne (header) du fichier.
    - Suppose que le fichier contient des lignes CSV où le 3e champ est
      l'identifiant du pilote et le dernier champ est le nombre de victoires.
    """
    count = 0
    joueurs_points = dict()
    begin = 0
    end = 0
    for ligne in generer_table_fichier("driver_standings"):
        if count != 0:
            count_coma = 0
            for i in range(len(ligne)):
                if ligne[i] == ",":
                    count_coma += 1
                    if count_coma == 2:
                        begin = i
                    if count_coma == 3:
                        end = i
                        break
            keys = ligne[begin + 1 : end]
            victory = ligne[-1]
            if keys in joueurs_points:
                joueurs_points[keys] = joueurs_points[keys] + int(victory)
            else:
                joueurs_points[keys] = int(victory)
        count += 1
    return joueurs_points


def nbr_points_joueurs():
    """
    Calcule le nombre total de points accumulés par chaque pilote à partir du fichier 'driver_standings'.

    Le fichier est lu ligne par ligne à l’aide de la fonction `generer_table_fichier`.
    La fonction extrait l'identifiant du pilote (champ 3) et le nombre de points (champ 4) de chaque ligne,
    puis additionne les points pour chaque pilote dans un dictionnaire.

    Sortie:
        dict: Un dictionnaire dont les clés sont les identifiants des pilotes,
              et les valeurs sont le total de leurs points (float).

    Remarques:
    - Ignore la première ligne (header).
    - Suppose que chaque ligne contient au moins 5 champs séparés par des virgules.
    - Le 3e champ est l'identifiant du pilote, le 4e champ correspond aux points gagnés.
    """
    count = 0
    joueurs_points = dict()
    begin_name = 0
    end_name = 0
    begin_points = 0
    end_points = 0
    for ligne in generer_table_fichier("driver_standings"):
        if count != 0:
            count_coma = 0
            for i in range(len(ligne)):
                if ligne[i] == ",":
                    count_coma += 1
                    if count_coma == 2:
                        begin_name = i
                    if count_coma == 3:
                        end_name = i
                        begin_points = i
                    if count_coma == 4:
                        end_points = i
                        break
            keys = ligne[begin_name + 1 : end_name]
            points = ligne[begin_points + 1 : end_points]
            if keys in joueurs_points:
                joueurs_points[keys] = joueurs_points[keys] + float(points)
            else:
                joueurs_points[keys] = float(points)
        count += 1
    return joueurs_points



def nom_joueurs():
    """
    Calcule le nombre total de points accumulés par chaque pilote à partir du fichier 'driver_standings'.

    Le fichier est lu ligne par ligne à l’aide de la fonction `generer_table_fichier`.
    La fonction extrait l'identifiant du pilote (champ 3) et le nombre de points (champ 4) de chaque ligne,
    puis additionne les points pour chaque pilote dans un dictionnaire.

    Sortie:
        dict: Un dictionnaire dont les clés sont les identifiants des pilotes,
              et les valeurs sont le total de leurs points (float).

    Remarques:
    - Ignore la première ligne (header).
    - Suppose que chaque ligne contient au moins 5 champs séparés par des virgules.
    - Le 3e champ est l'identifiant du pilote, le 4e champ correspond aux points gagnés.
    """
    count = 0
    joueurs_nom = dict()
    begin = 0
    end = 0
    for ligne in generer_table_fichier("drivers"):
        if count != 0:
            count_coma = 0
            for i in range(len(ligne)):
                if ligne[i] == ",":
                    count_coma += 1
                    if count_coma == 1:
                        begin = i
                    if count_coma == 2:
                        end = i
                        break
            keys = ligne[:begin]
            name = ligne[begin + 2 : end - 1]
            joueurs_nom[keys] = name
        count += 1
    return joueurs_nom


def nbr_victoire_ttal_pilote():
    """
    Associe le nombre total de victoires à chaque pilote en utilisant leur nom lisible.

    Cette fonction croise deux sources :
    - `nbr_victoire_joueurs()` : fournit un dictionnaire {id_pilote: nb_victoires}
    - `nom_joueurs()` : fournit un dictionnaire {id_pilote: nom_pilote}
    
    Sortie:
        dict: Un dictionnaire avec les noms des pilotes 
              comme clés et leur nombre total de victoires comme valeurs.

    Remarques:
    - Les identifiants de pilotes présents dans `score` mais absents de `pilotes` sont ignorés.
    - Si deux pilotes ont le même nom, seul l’identifiant dans `score` est pris en compte.
    """
    score = nbr_victoire_joueurs()
    pilotes = nom_joueurs()
    return {pilotes[key]: score[key] for key in score if key in pilotes}


def nbr_points_ttal_pilote():
    """
    Cette fonction récupère les scores des joueurs et les associe à leurs noms respectifs.

    Elle utilise la fonction `nbr_points_joueurs()` pour obtenir les scores des joueurs 
    et la fonction `nom_joueurs()` pour obtenir les noms des joueurs. Elle retourne 
    un dictionnaire où les clés sont les noms des joueurs et les valeurs sont leurs 
    scores respectifs.

    Sortie:
        dict: Un dictionnaire associant les noms des joueurs (tirés de `nom_joueurs()`) 
              à leurs scores respectifs (tirés de `nbr_points_joueurs()`), mais seulement 
              pour les joueurs dont le nom et le score existent dans les deux sources.

    Exemple:
        Si `nom_joueurs()` retourne ['Alice', 'Bob', 'Charlie'] et `nbr_points_joueurs()` 
        retourne {0: 10, 1: 15, 2: 20}, la fonction renverra:
        {'Alice': 10, 'Bob': 15, 'Charlie': 20}
    """
    score = nbr_points_joueurs()
    pilotes = nom_joueurs()
    return {pilotes[key]: score[key] for key in score if key in pilotes}


def classement_absolu_pilote(pilote: str) -> int:
    """
    Cette fonction calcule et affiche le classement absolu d'un pilote en fonction de 
    ses victoires et de son score total de points.

    Elle vérifie si le pilote est bien référencé dans la liste des pilotes, puis utilise 
    les fonctions `nbr_victoire_ttal_pilote()` et `nbr_points_ttal_pilote()` pour récupérer 
    respectivement les victoires et les points totaux du pilote. Elle trie ensuite tous les pilotes 
    en fonction de leurs victoires, puis génère un classement absolu, où les pilotes sont classés 
    du plus grand au plus petit nombre de victoires. 

    Le classement du pilote donné en paramètre est affiché. Si le pilote n'a pas de victoires, 
    un message spécifique indique qu'il n'a pas de classement.

    Arguments:
        pilote (str): Le nom du pilote dont on souhaite connaître le classement.

    Sortie:
        int: Le nombre de victoires du pilote dans le classement absolu.

    Exceptions:
        ValueError: Si le pilote spécifié n'est pas référencé dans la liste des pilotes.

    Exemple:
        Si 'Alice' a 5 victoires et 120 points, et que son classement est le 1er dans 
        les victoires, la fonction affichera:
        "Le pilote 'Alice' est arrivé premier avec 5 victoires de course tout au long de sa carrière."
        "Il aura marqué au total 120 points."
    """
    drivers = nom_joueurs()
    if pilote not in drivers.values():
        raise ValueError(f"{pilote} n'est pas un pilote référencé")
    score1 = nbr_victoire_ttal_pilote()
    score2 = nbr_points_ttal_pilote()
    res = score1[pilote]
    classement = sorted(score1.items(), key=lambda x: x[1], reverse=True)
    resultat = {}
    for i in range(len(classement)):
        resultat[classement[i][0]] = i + 1
    if res != 0:
        print(
            f"Le pilote '{pilote}' est arrivé premier "
            f"avec {res} victoires de course "
            "tout au long de sa carrière."
            f"Il aura marqué au total {score2[pilote]}"
        )
    else:
        print(
            f"Le pilote {pilote} n'a pas de classement "
            "car il n'a aucune victoire de course sur "
            "toute sa carrière."
            f" Il aura marqué au total {score2[pilote]} points."
        )

    return res


def ttal_vict_pts_pilote(pilote: str) -> list:
    """
    Cette fonction retourne une liste contenant les statistiques de victoire et de points d'un pilote.

    Elle utilise les fonctions `nbr_victoire_ttal_pilote()` et `nbr_points_ttal_pilote()` pour récupérer 
    respectivement les victoires et les points totaux des pilotes. Si le pilote est référencé, la fonction 
    retourne une liste contenant le nombre de victoires et le nombre de points pour ce pilote spécifique.

    Arguments:
        pilote (str): Le nom du pilote dont on souhaite connaître les statistiques.

    Sorties:
        list: Une liste contenant deux éléments :
              - Le nombre de victoires du pilote.
              - Le nombre de points du pilote.

    Exceptions:
        ValueError: Si le pilote spécifié n'est pas référencé dans les listes de victoires et de points.

    Exemple:
        Si 'Alice' a 5 victoires et 120 points, la fonction renverra :
        [5, 120]
    """
    victory = nbr_victoire_ttal_pilote()
    points = nbr_points_ttal_pilote()
    if pilote not in victory.keys():
        raise ValueError(f"{pilote} n'est pas un pilote référencé")
    table = {key: [victory[key], points[key]] for key in victory}
    return table[pilote]


victory_ = list(nbr_victoire_ttal_pilote().values())
points_ = list(nbr_points_ttal_pilote().values())


def plot_relation_victoires_points(victory_, points_):
    """
    Cette fonction génère un graphique de dispersion montrant la relation entre
    le nombre de victoires et le total de points en F1.

    Arguments:
        victory_ (array-like): Le tableau des nombres de victoires.
        points_ (array-like): Le tableau des points associés.
    """
    plt.figure(figsize=(8, 6))
    plt.scatter(victory_, points_, color="royalblue", edgecolors="k", s=100, alpha=0.8)
    plt.title(
        "Relation entre nombre de victoires et points en F1",
        fontsize=14,
        fontweight="bold",
    )
    plt.xlabel("Nombre de victoires", fontsize=12)
    plt.ylabel("Total de points", fontsize=12)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.show()


def regression_lineaire(victory_, points_):
    """
    Cette fonction effectue une régression linéaire entre les données de victoires et de points.
    Elle affiche plusieurs graphiques pour évaluer la qualité du modèle de régression, 
    ainsi que les statistiques associées.

    Arguments:
        victory_ (array-like): Le tableau des nombres de victoires (pour la régression).
        points_ (array-like): Le tableau des points associés (pour la régression).
    """
    from scipy import stats
    from sklearn.linear_model import LinearRegression

    victory_ = np.array(victory_).reshape(-1, 1)
    model = LinearRegression()
    model.fit(victory_, points_)
    score = model.score(victory_, points_)
    print(f"Coefficient (slope) : {model.coef_[0]:.2f}")
    print(f"Ordonnée à l'origine (intercept) : {model.intercept_:.2f}")
    print(f"Score R² du modèle : {score:.4f}")
    x_range = np.linspace(min(victory_), max(victory_), 100).reshape(-1, 1)
    y_pred = model.predict(x_range)
    plt.figure(figsize=(8, 6))
    plt.scatter(
        victory_,
        points_,
        color="royalblue",
        edgecolors="k",
        s=100,
        alpha=0.8,
        label="Données réelles",
    )
    plt.plot(x_range, y_pred, color="crimson", linewidth=2, label="Régression linéaire")
    plt.title("Régression : Points vs Victoires", fontsize=14, fontweight="bold")
    plt.xlabel("Nombre de victoires", fontsize=12)
    plt.ylabel("Total de points", fontsize=12)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend()
    plt.tight_layout()
    plt.show()

    # Calcul des résidus
    y_pred = model.predict(victory_)
    residuals = points_ - y_pred

    # QQ plot des résidus pour tester la normalité
    plt.figure(figsize=(6, 6))
    stats.probplot(residuals, dist="norm", plot=plt)
    plt.title("QQ Plot des résidus", fontsize=14, fontweight="bold")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.show()

    # Visualisation des résidus vs valeurs prédites
    plt.scatter(y_pred, residuals, alpha=0.7)
    plt.axhline(0, color="red", linestyle="--")
    plt.xlabel("Valeurs prédites")
    plt.ylabel("Résidus")
    plt.title("Résidus vs. valeurs prédites")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # Histogramme des résidus pour vérifier leur distribution
    plt.hist(residuals, bins=15, edgecolor="black", alpha=0.7)
    plt.title("Distribution des résidus")
    plt.xlabel("Résidu")
    plt.ylabel("Fréquence")
    plt.grid(True)
    plt.tight_layout()
    plt.show()
