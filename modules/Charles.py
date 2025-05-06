# Question et Data-Mining de Charles OZEEL
# coding: utf-8

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors

def mappage_courses_sans_noms(circuits: pd.DataFrame, display = False):
    """
    Affiche les circuits des courses de F1 sur une carte du monde (statique), sans les noms des circuits visibles.
    
    Paramètres :
    - circuits : DataFrame contenant les coordonnées des circuits ('lat', 'lng', 'name')
    - display : True ou False, si True affiche les noms sur chaque point sinon par défaut ne le fait pas
    """
    import geopandas as gpd
    circuits = circuits.copy()
    circuits = circuits.drop_duplicates(subset=["circuitId"])
    geo_df = gpd.GeoDataFrame(circuits,
                               geometry=gpd.points_from_xy(circuits['lng'], circuits['lat']),
                               crs="EPSG:4326")
    world = gpd.read_file("naturalearth_lowres/naturalearth_lowres.shp")
    fig, ax = plt.subplots(figsize=(15, 10))
    world.plot(ax=ax, color='lightgray', edgecolor='white')
    geo_df.plot(ax=ax, color='midnightblue', markersize=50)
    if display:
        for x, y, name in zip(geo_df.geometry.x, geo_df.geometry.y, geo_df['circuitRef']):
            ax.text(x + 1, y + 0.5, name, fontsize=7, ha='left', va='bottom')
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
    pit_stops = pit_stops.groupby(["raceId", "driverId"]).agg(duration_min=("duration", "min"),
                                                               duration_max=("duration", "max"),
                                                               duration_mean = ("duration", "mean"),
                                                               duration_std=("duration", "std")).reset_index()
    pit_stops = pd.merge(drivers, pit_stops, on ="driverId", how ="right")
    pit_stops = pit_stops.groupby("driverRef").agg({"duration_min": "mean",
                                                  "duration_max": "mean",
                                                  "duration_std": "mean", 
                                                  "duration_mean": "mean"}).reset_index()
    return pit_stops


def stat_mean_stop_drivers(nom_pilote: str, pit_stops: pd.DataFrame, drivers: pd.DataFrame):
    pit_stop = table_stat_stop_drivers(pit_stops, drivers)
    pit_stop_names = list(pit_stop["driverRef"])
    if nom_pilote not in pit_stop_names:
        raise ValueError(f"{nom_pilote} est incorrecte,"
                         " ou il y a des valeurs manquantes "
                         "liées au pilote recherché.")
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
    data = data.sort_values(by = "duration_mean")
    return data[["driverRef", "duration_mean"]]


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
            "Le fichier 'donnees_formule_un' est introuvable sur votre bureau.")
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
            #ici False + assure d'avoir une valeur numérique à la fin (1 ou 0)#
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


def nom_joueurs_pure():
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
    return {pilotes[key]: score[key] for key in score if key in pilotes}


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
    return {pilotes[key]: score[key] for key in score if key in pilotes}


def classement_absolu_pilote_pure(pilote: str) -> int:
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
    drivers = nom_joueurs_pure()
    if pilote not in drivers.values():
        raise ValueError(f"{pilote} n'est pas un pilote référencé")
    score1 = nbr_victoire_ttal_pilote_pure()
    score2 = nbr_points_ttal_pilote_pure()
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


def ttal_vict_pts_pilote_pure(pilote: str) -> list:
    """
    Cette fonction retourne une liste contenant les statistiques de victoire et de points d'un pilote.

    Elle utilise les fonctions `nbr_victoire_ttal_pilote_pure()` et `nbr_points_ttal_pilote_pure()` pour récupérer
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
    victory = nbr_victoire_ttal_pilote_pure()
    points = nbr_points_ttal_pilote_pure()
    if pilote not in victory.keys():
        raise ValueError(f"{pilote} n'est pas un pilote référencé")
    table = {key: [victory[key], points[key]] for key in victory}
    return table[pilote]


def plot_relation_victoires_points(victory_: list, points_: list):
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


def regression_lineaire(victory_: list, points_: list):
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
