# Fonctions pour les:
# Q1: Où se place sur une map les grands prix ?
# Q2: Comment récupérier et comparer les pilotes selons le nombre de point, de victoire et taux de victoire ?
# Q3: Comment se répartie le temps au stand par pilote ?
# Régression linéaire entre nombre de point et nombre de victoire.
# coding: utf-8

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def mappage_courses(circuits: pd.DataFrame, display=False):
    """
    Affiche les circuits des courses de F1 sur une carte du monde (statique), sans les noms des circuits visibles.

    Paramètres :
    - circuits : DataFrame contenant les coordonnées des circuits ('lat', 'lng', 'name')
    - display : True ou False, si True affiche les noms sur chaque point sinon par défaut ne le fait pas
    """
    import geopandas as gpd

    circuits = circuits.copy()
    circuits = circuits.drop_duplicates(subset=["circuitId"])
    geo_df = gpd.GeoDataFrame(
        circuits,
        geometry=gpd.points_from_xy(circuits["lng"], circuits["lat"]),
        crs="EPSG:4326",
    )
    world = gpd.read_file("naturalearth_lowres/naturalearth_lowres.shp")
    fig, ax = plt.subplots(figsize=(15, 10))
    world.plot(ax=ax, color="lightgray", edgecolor="white")
    geo_df.plot(ax=ax, color="midnightblue", markersize=50)
    if display:
        for x, y, name in zip(
            geo_df.geometry.x, geo_df.geometry.y, geo_df["circuitRef"]
        ):
            ax.text(x + 1, y + 0.5, name, fontsize=7, ha="left", va="bottom")
    plt.title("Circuits de F1 dans le monde", fontsize=16)
    plt.show()


def table_stat_stop_drivers(pit_stops: pd.DataFrame, drivers: pd.DataFrame):
    """
        Calcule les statistiques moyennes des durées d'arrêts aux stands (pit stops)
    pour un pilote donné à partir des données d'une tabel pit_stops et des drivers.

    Paramètres :
    ------------
    nom_pilote : str
        Le nom de référence du pilote (`driverRef`) pour lequel on veut les statistiques.
    pit_stops : pd.DataFrame
        Un DataFrame contenant les informations sur les arrêts aux stands, avec les colonnes
        requises : "raceId", "driverId", "milliseconds".
    drivers : pd.DataFrame
        Un DataFrame contenant les informations des pilotes, avec les colonnes requises :
        "driverId", "driverRef".

    Retour :
    --------
    pd.DataFrame
        Un DataFrame contenant les moyennes suivantes pour le pilote sélectionné :
        - driverRef : Nom du pilote renseigné en entrée
        - duration_min : moyenne des durées minimales par course
        - duration_max : moyenne des durées maximales par course
        - duration_mean : moyenne des durées moyenne par course
        - duration_std : moyenne des écarts-types des durées par course

    Lève :
    ------
    ValueError
        Si les colonnes nécessaires sont absentes ou si les données ne sont pas convertibles
        en valeurs numériques.
    """
    # Ici on ajoute la condition que pit_stops doit posséder ces colonnes#
    colonnes_stops = {"raceId", "driverId", "milliseconds"}
    colonnes_drivers = {"driverId", "driverRef"}
    if not all(col in pit_stops.columns for col in colonnes_stops):
        raise ValueError(
            f"La table pit_stops ne respecte pas les normes attendus,"
            f" elle doit posséder les colonnes {colonnes_stops}."
        )
    if not all(col in drivers.columns for col in colonnes_drivers):
        raise ValueError(
            f"La table drivers ne respecte pas les normes attendus,"
            f" elle doit posséder les colonnes {colonnes_drivers}."
        )
    pit_stops = pit_stops.copy(deep=True)
    # On renvoie une erreure si jamais la colonne "milliseconds"#
    # n'est pas en float type ou pas convertible en float type#
    if pit_stops["milliseconds"].dtype == object:
        if pd.to_numeric(pit_stops["milliseconds"], errors="coerce").isna().all():
            raise ValueError(
                f"La colonne 'milliseconds' de la table pit_stops n'est pas formatable"
                f"en valeurs numériques. Corrigez la table"
            )
    pit_stops["duration"] = pit_stops["milliseconds"] * (10**-3)
    pit_stops = (
        pit_stops.groupby(["raceId", "driverId"])
        .agg(
            duration_min=("duration", "min"),
            duration_max=("duration", "max"),
            duration_mean=("duration", "mean"),
            duration_std=("duration", "std"),
        )
        .reset_index()
    )
    pit_stops = pd.merge(drivers, pit_stops, on="driverId", how="right")
    pit_stops = (
        pit_stops.groupby("driverRef")
        .agg(
            {
                "duration_min": "mean",
                "duration_max": "mean",
                "duration_std": "mean",
                "duration_mean": "mean",
            }
        )
        .reset_index()
    )
    return pit_stops


def stat_mean_stop_drivers(
    nom_pilote: str, pit_stops: pd.DataFrame, drivers: pd.DataFrame
):
    pit_stop = table_stat_stop_drivers(pit_stops, drivers)
    pit_stop_names = list(pit_stop["driverRef"])
    if nom_pilote not in pit_stop_names:
        raise ValueError(
            f"{nom_pilote} est incorrecte,"
            " ou il y a des valeurs manquantes "
            "liées au pilote recherché."
        )
    return pit_stop[pit_stop["driverRef"] == nom_pilote]


def classement_mean_stop_drivers(pit_stops: pd.DataFrame, drivers: pd.DataFrame):
    """
    Génère un classement des pilotes en fonction de la durée moyenne de leurs arrêts aux stands.

    Cette fonction utilise les données des pit stops et des pilotes pour calculer les durées moyennes
    d'arrêt par pilote à l'aide de la fonction `table_stat_stop_drivers`. Le résultat est un DataFrame
    trié par ordre croissant de la durée moyenne d'arrêt, affichant pour chaque pilote son identifiant
    (`driverRef`) et sa durée moyenne (`duration_mean`).

    Paramètres
    ----------
    pit_stops : pd.DataFrame
        Le DataFrame contenant les informations sur les arrêts aux stands.
    drivers : pd.DataFrame
        Le DataFrame contenant les informations sur les pilotes.

    Retourne
    -------
    pd.DataFrame
        Un DataFrame trié par durée moyenne d'arrêt croissante, avec les colonnes :
        - 'driverRef' : identifiant du pilote
        - 'duration_mean' : durée moyenne d'arrêt aux stands
    """
    data = table_stat_stop_drivers(pit_stops, drivers)
    data = data.sort_values(by="duration_mean")
    return data[["driverRef", "duration_mean"]]


def plot_duration_mean_distribution(
    pit_stops: pd.DataFrame, drivers: pd.DataFrame, bins: int = 5
):
    """
    Affiche un histogramme en barres des fréquences par intervalle de durée moyenne d'arrêts aux stands,
    afin de visualiser une éventuelle allure de loi de Poisson.

    Paramètres
    ----------
    pit_stops : pd.DataFrame
        Le DataFrame contenant les informations sur les arrêts aux stands.
    drivers : pd.DataFrame
        Le DataFrame contenant les informations sur les pilotes.
    bins : int
        Le nombre d'intervalles à créer pour regrouper les durées moyennes.
    """
    df = classement_mean_stop_drivers(pit_stops, drivers)[
        ["driverRef", "duration_mean"]
    ]
    df["interval"] = pd.cut(df["duration_mean"], bins=bins)
    interval_counts = df["interval"].value_counts(normalize=True).sort_index()
    plt.figure(figsize=(12, 6))
    plt.bar(
        interval_counts.index.astype(str),
        interval_counts.values,
        color="skyblue",
        edgecolor="black",
    )
    plt.title(
        "Fréquences relatives par intervalle de durée moyenne des arrêts aux stands"
    )
    plt.xlabel("Intervalle de durée moyenne (ms)")
    plt.ylabel("Fréquence relative")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()


def generer_table_fichier(nom_fichier_recherche: str):
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
    for nom_fichier in os.listdir(dossier):
        chemin_fichier = os.path.join(dossier, nom_fichier)
        if os.path.isfile(chemin_fichier):
            nom_sans_ext = os.path.splitext(nom_fichier)[0]
            if nom_sans_ext == nom_fichier_recherche:
                with open(chemin_fichier, "r", encoding="utf-8") as f:
                    for ligne in f:
                        yield ligne.strip()


def nbr_victoire_joueurs_pure():
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
    for ligne in generer_table_fichier("results"):
        if count != 0:
            count_coma = 0
            for i in range(len(ligne)):
                if ligne[i] == ",":
                    count_coma += 1
                    if count_coma == 2:
                        begin_name = i
                    if count_coma == 3:
                        end_name = i
                    if count_coma == 8:
                        begin_position = i
                    if count_coma == 9:
                        end_position = i
                        break
            keys = ligne[begin_name + 1 : end_name]
            # ici False + assure d'avoir une valeur numérique à la fin (1 ou 0)#
            victory = False + (ligne[begin_position + 1 : end_position] == "1")
            if keys in joueurs_points:
                joueurs_points[keys] = joueurs_points[keys] + victory
            else:
                joueurs_points[keys] = victory
        count += 1
    return joueurs_points


def nbr_points_joueurs_pure():
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
    for ligne in generer_table_fichier("results"):
        if count != 0:
            count_coma = 0
            for i in range(len(ligne)):
                if ligne[i] == ",":
                    count_coma += 1
                    if count_coma == 2:
                        begin_name = i
                    if count_coma == 3:
                        end_name = i
                    if count_coma == 9:
                        begin_points = i
                    if count_coma == 10:
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


def nbr_courses_joueurs_pure():
    """
     Calcule le nombre total de courses disputées par chaque pilote à partir du fichier 'results'.

    La fonction lit ligne par ligne un fichier brut (type CSV), extrait les identifiants des pilotes
    et les identifiants de course, puis compte combien de fois chaque pilote apparaît (c'est-à-dire
    combien de courses il a disputées).

    Retour
    ------
    dict
        Un dictionnaire où chaque clé est l'identifiant d'un pilote (str), et chaque valeur est
        le nombre total de courses qu'il a disputées (int).
    """
    count = 0
    joueurs_points = dict()
    begin_name = 0
    end_name = 0
    begin_race = 0
    end_race = 0
    for ligne in generer_table_fichier("results"):
        if count != 0:
            count_coma = 0
            for i in range(len(ligne)):
                if ligne[i] == ",":
                    count_coma += 1
                    if count_coma == 1:
                        begin_race = i
                    if count_coma == 2:
                        end_race = i
                        begin_name = i
                    if count_coma == 3:
                        end_name = i
                        break
            keys = ligne[begin_name + 1 : end_name]
            race = ligne[begin_race + 1 : end_race]
            if keys in joueurs_points:
                joueurs_points[keys] = joueurs_points[keys] + [race]
            else:
                joueurs_points[keys] = [race]
        count += 1
    for keys in joueurs_points.keys():
        joueurs_points[keys] = len(joueurs_points[keys])
    return joueurs_points


def nom_joueurs_pure():
    """
    Extrait un dictionnaire des identifiants des pilotes associés à leurs noms à partir du fichier 'drivers'.

    La fonction parcourt ligne par ligne un fichier brut (probablement CSV sans en-tête explicite au bon format),
    et extrait, pour chaque ligne sauf la première, l'identifiant (`driverId`) et le nom (`driverRef`) ou nom court
    du pilote. Elle utilise des indices de virgule pour découper manuellement chaque ligne.

    Retour
    ------
    dict
        Un dictionnaire où chaque clé est un identifiant de pilote (str) et chaque valeur est le nom ou identifiant texte du pilote (str).
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


def nbr_victoire_ttal_pilote_pure():
    """
    Associe le nombre total de victoires à chaque pilote en utilisant leur nom lisible.

    Cette fonction croise deux sources :
    - `nbr_victoire_joueurs_pure()` : fournit un dictionnaire {id_pilote: nb_victoires}
    - `nom_joueurs_pure()` : fournit un dictionnaire {id_pilote: nom_pilote}

    Sortie:
        dict: Un dictionnaire avec les noms des pilotes
              comme clés et leur nombre total de victoires comme valeurs.

    Remarques:
    - Les identifiants de pilotes présents dans `score` mais absents de `pilotes` sont ignorés.
    - Si deux pilotes ont le même nom, seul l’identifiant dans `score` est pris en compte.
    """
    score = nbr_victoire_joueurs_pure()
    pilotes = nom_joueurs_pure()
    return {pilotes[key]: score[key] for key in score}


def nbr_course_ttal_pilote_pure():
    """
    Retourne un dictionnaire associant chaque pilote à son nombre total de courses disputées.

    Cette fonction combine les résultats de deux fonctions :
    - `nbr_courses_joueurs_pure()` : fournit un dictionnaire avec le nombre de courses par identifiant pilote.
    - `nom_joueurs_pure()` : fournit un dictionnaire de correspondance entre identifiants pilotes et noms.

    Retour
    ------
    dict
        Un dictionnaire où chaque clé est le nom d’un pilote (str), et chaque valeur est le nombre de
        courses qu’il a disputées (int).
    """
    course = nbr_courses_joueurs_pure()
    pilotes = nom_joueurs_pure()
    return {pilotes[key]: course[key] for key in course}


def nbr_points_ttal_pilote_pure():
    """
    Cette fonction récupère les scores des joueurs et les associe à leurs noms respectifs.

    Elle utilise la fonction `nbr_points_joueurs_pure()` pour obtenir les scores des joueurs
    et la fonction `nom_joueurs_pure()` pour obtenir les noms des joueurs. Elle retourne
    un dictionnaire où les clés sont les noms des joueurs et les valeurs sont leurs
    scores respectifs.

    Sortie:
        dict: Un dictionnaire associant les noms des joueurs (tirés de `nom_joueurs_pure()`)
              à leurs scores respectifs (tirés de `nbr_points_joueurs_pure()`), mais seulement
              pour les joueurs dont le nom et le score existent dans les deux sources.

    Exemple:
        Si `nom_joueur_pure()` retourne ['Alice', 'Bob', 'Charlie'] et `nbr_points_joueurs_pure()`
        retourne {0: 10, 1: 15, 2: 20}, la fonction renverra:
        {'Alice': 10, 'Bob': 15, 'Charlie': 20}
    """
    score = nbr_points_joueurs_pure()
    pilotes = nom_joueurs_pure()
    return {pilotes[key]: score[key] for key in score}


def rate_victoire_absolu_pilote_pure():
    """
       Calcule le taux de victoire absolu de chaque pilote, c'est-à-dire le ratio du nombre de victoires
    sur le nombre total de courses disputées.

    Cette fonction utilise les résultats de deux fonctions :
    - `nbr_course_ttal_pilote_pure()` : fournit un dictionnaire avec le nombre total de courses disputées par pilote.
    - `nbr_victoire_ttal_pilote_pure()` : fournit un dictionnaire avec le nombre total de victoires par pilote.

    Le taux de victoire est calculé pour chaque pilote comme suit :
    - Taux de victoire = Nombre de victoires / Nombre total de courses

    Retour
    ------
    dict
        Un dictionnaire où chaque clé est l'identifiant d'un pilote (str), et chaque valeur est son taux de victoire
        arrondi à trois décimales (float).
    """
    course = nbr_course_ttal_pilote_pure()
    victory = nbr_victoire_ttal_pilote_pure()
    return {key: round(victory[key] / course[key], 3) for key in course}


def ttal_vict_pts_pilote_table():
    """
    Calcule et retourne une table des statistiques de performance pour chaque pilote.

    Cette fonction utilise les résultats de plusieurs fonctions pour générer une table avec les statistiques
    suivantes pour chaque pilote :
    - Nombre total de victoires
    - Nombre total de points
    - Nombre total de courses disputées
    - Taux de victoire (rapport entre victoires et courses disputées)

    Les fonctions utilisées sont les suivantes :
    - `nbr_victoire_ttal_pilote_pure()` : nombre total de victoires par pilote.
    - `nbr_points_ttal_pilote_pure()` : nombre total de points par pilote.
    - `nbr_course_ttal_pilote_pure()` : nombre total de courses disputées par pilote.
    - `rate_victoire_absolu_pilote_pure()` : taux de victoire par pilote.

    Retour
    ------
    dict
        Un dictionnaire où chaque clé est l'identifiant d'un pilote (str), et chaque valeur est une liste de statistiques
        [victoires, points, courses, taux de victoire] pour ce pilote. Ces statistiques sont dans l'ordre suivant :
        - Nombre de victoires (int)
        - Nombre de points (int)
        - Nombre de courses disputées (int)
        - Taux de victoire (float, arrondi à 3 décimales)
    """
    victory = nbr_victoire_ttal_pilote_pure()
    points = nbr_points_ttal_pilote_pure()
    course = nbr_course_ttal_pilote_pure()
    score = rate_victoire_absolu_pilote_pure()
    table = {
        key: [victory[key], points[key], course[key], score[key]] for key in victory
    }
    return table


def ttal_vict_pts_pilote_pure(pilote: str):
    """
       Retourne les statistiques complètes d'un pilote donné sous forme de liste.

    La fonction extrait les statistiques du pilote en utilisant la fonction `ttal_vict_pts_pilote_table()`,
    qui calcule le nombre de victoires, de points, le nombre de courses et le taux de victoire pour tous les pilotes.
    Elle vérifie ensuite si le pilote spécifié existe dans la table. Si le pilote est présent, elle retourne les statistiques,
    sinon elle lève une exception `ValueError`.

    Paramètres
    ----------
    pilote : str
        L'identifiant du pilote (par exemple, "Hamilton") pour lequel les statistiques doivent être retournées.

    Retour
    ------
    list
        Une liste contenant les statistiques du pilote spécifié dans l'ordre suivant :
        - Nombre de victoires (int)
        - Nombre de points (int)
        - Nombre de courses disputées (int)
        - Taux de victoire (float, arrondi à 3 décimales)

    Exceptions
    ----------
    ValueError
        Si le pilote spécifié n'est pas trouvé dans les données.
    """
    table = ttal_vict_pts_pilote_table()
    if pilote not in table.keys():
        raise ValueError(f"{pilote} n'est pas un pilote référencé")
    return table[pilote]


# On refait cette fonction mais cette fois avec pandas#

def ttal_vict_pts_table(results: pd.DataFrame, drivers= pd.DataFrame):
    """
    Calcule les statistiques de performance des pilotes à partir des résultats de course.

    Cette fonction fusionne les données de résultats de course avec celles des pilotes,
    puis calcule pour chaque pilote :
        - le total de points marqués,
        - le nombre de victoires (positionText == "1"),
        - le nombre total de courses disputées,
        - le taux de victoire (nombre de victoires divisé par le nombre de courses).

    Paramètres
    ----------
    results : pd.DataFrame
        DataFrame contenant les résultats des courses (doit inclure 'driverId', 'positionText', 'points', 'raceId').
    
    drivers : pd.DataFrame, optional
        DataFrame contenant les informations des pilotes (doit inclure 'driverId' et 'driverRef').
        Par défaut, un DataFrame vide.

    Retour
    ------
    pd.DataFrame
        DataFrame contenant, pour chaque pilote (driverRef) :
        - 'points' : total des points marqués,
        - 'win' : nombre de victoires,
        - 'raceId' : nombre total de courses,
        - 'victory_rate' : taux de victoire (victoires / courses).
    """
    data = pd.merge(results, drivers, on="driverId", how="right")
    data["win"] = data["positionText"].apply(lambda x: 1 if x == "1" else 0)
    data["nbr_course"] = data.groupby("driverRef").agg({"raceId": "count"})
    res = (
        data.groupby("driverRef")
        .agg({"points": "sum", "win": "sum", "raceId": "count"})
        .reset_index()
    )
    res["victory_rate"] = round(res["win"] / res["raceId"], 3)
    return res


def ttal_vict_pts_pilote(
    pilote: str, results: pd.DataFrame, drivers: pd.DataFrame
) -> list:
    """
    Calcule les statistiques de performance d'un pilote de Formule 1 donné : total de points,
    nombre de victoires, nombre de courses disputées, et taux de victoire.

    Paramètres
    ----------
    pilote : str
        Identifiant du pilote (driverRef) pour lequel on souhaite extraire les statistiques.
    results : pd.DataFrame
        DataFrame contenant les résultats des courses (doit inclure les colonnes 'driverId',
        'raceId', 'positionText', et 'points').
    drivers : pd.DataFrame
        DataFrame contenant les informations sur les pilotes (doit inclure 'driverId' et 'driverRef').

    Retour
    ------
    list
        Liste contenant les statistiques du pilote, dans l'ordre suivant :
        - Total des points (float)
        - Nombre de victoires (int)
        - Nombre de courses disputées (int)
        - Taux de victoire arrondi à 3 décimales (float)
    """
    res = ttal_vict_pts_table(results, drivers)
    res = res[res["driverRef"] == pilote]
    return res.iloc[0].tolist()[1:]


def plot_relation_wins(victory_: list, points_: list):
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

def plot_relation_pitstops(pit_stop_: pd.DataFrame, points_: pd.DataFrame,):
    """
    Cette fonction génère un graphique de dispersion montrant la relation entre
    le temps d'arrêt au stand et le total de points en F1.

    Arguments:
        pit_stop_ (array-like): Le tableau des nombres de victoires.
        points_ (array-like): Le tableau des points associés.
    """
    plt.figure(figsize=(8, 6))
    plt.scatter(points_, pit_stop_, color="royalblue", edgecolors="k", s=100, alpha=0.8)
    plt.title(
        "Relation entre nombre de points et temps au stand moyen",
        fontsize=14,
        fontweight="bold",
    )
    plt.ylabel("Nombre de points", fontsize=12)
    plt.xlabel("durée au stand (en ms)", fontsize=12)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.show()


def regression_lineaire_wins(victory_: list, points_: list):
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




def regression_lineaire_pitstops(pit_stop_: list, points_: list):
    """
    Cette fonction effectue une régression linéaire entre les données de tempsd d'arrêt au stand et de points.
    Elle affiche plusieurs graphiques pour évaluer la qualité du modèle de régression,
    ainsi que les statistiques associées.

    Arguments:
        pit_stop_ (array-like): Le tableau du temps moyens par pilote au stand (pour la régression).
        points_ (array-like): Le tableau des points associés (pour la régression).
    """
    from scipy import stats
    from sklearn.linear_model import LinearRegression

    pit_stop_ = np.array(pit_stop_).reshape(-1, 1)
    model = LinearRegression()
    model.fit(pit_stop_, points_)
    score = model.score(pit_stop_, points_)
    print(f"Coefficient (slope) : {model.coef_[0]:.2f}")
    print(f"Ordonnée à l'origine (intercept) : {model.intercept_:.2f}")
    print(f"Score R² du modèle : {score:.4f}")
    x_range = np.linspace(min(pit_stop_), max(pit_stop_), 100).reshape(-1, 1)
    y_pred = model.predict(x_range)
    plt.figure(figsize=(8, 6))
    plt.scatter(
        pit_stop_,
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
    y_pred = model.predict(pit_stop_)
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
