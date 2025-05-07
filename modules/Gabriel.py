import numpy as np
import pandas as pd
import copy as cp
from src import import_data as i_d

# importation des données

i_d.charger_donnees_depuis_bureau()


"""
1) Quelle écurie a gagné le plus de courses en cumulé des saisons ?(faire un classement des écuries)
2) Quelle écurie a remporté le plus de saisons ?(faire un classement)
3) Question en python base:
"""


def rm_na(l):
    """
    cette fonction prend pour objet un tableau panda et
    retourne une table dont les \\N ont été remplacées
    par des NaN plus facilement traitables
    ------------------------------------------------------------------------------------
    arguments:

    l : un dataframe pandas
    ------------------------------------------------------------------------------------
    renvoit:

    un dataframe pandas dont les \\N ont été remplacées par des NaN
    """
    if not isinstance(l, pd.DataFrame):
        raise TypeError(f"{l} doit être un dataframe pandas")
    table_corrigée = cp.deepcopy(l)
    return table_corrigée.replace("\\N", np.nan)


def classement_victoires_écuries():
    """
    cette méthode renvoit un dataframe pandas qui condense le classement des écuries
    selon le nombre de victoires cumulées
    Returns:
    -------
    un dataframe pandas qui condense le classement des écuries
    selon le nombre de victoires cumulées
    """
    constructor_gagnants = constructor_results.loc[
        constructor_results["points"]
        == constructor_results.groupby("raceId")["points"].transform("max")
    ]
    constructor_gagnants = constructor_gagnants[["raceId", "constructorId"]]
    constructor_gagnants = constructor_gagnants.groupby("constructorId")[
        "raceId"
    ].count()
    constructor_gagnants = constructor_gagnants.sort_values(ascending=False)
    constructor_gagnants = pd.merge(
        constructor_gagnants, constructors, on="constructorId", how="right"
    )
    constructor_gagnants = constructor_gagnants[["name", "raceId"]].sort_values(
        by="raceId", ascending=False
    )
    constructor_gagnants.to_csv(
        "resultats/Q4-pandas.csv", index=False, encoding="utf-8"
    )
    return constructor_gagnants


# 2) quel constructeur a remporté le plus de saisons ? Faire un classement


def classement_courses_points():
    """
    Renvoit une table
    qui fait correspondre à chaque écurie le nombre de points remportés par course

    Return
    ------
    pandas.csv
        Le nombre de points par écurie pour chaque course de la base de donnée
    """
    # on fait correspondre les tables races et constructor_results pour obtenir
    # en une seule table les variables "constructorId", "year" et "points"
    total_course = pd.merge(races, constructor_results, how="inner")
    total_course = total_course.loc[:, ["year", "constructorId", "points"]]
    return total_course


def points_construct_annee():
    """
    Renvoit une tabl qui fait correspondre pour chaque année
    et chaque constructeur le nombre de points accumulés

    Return
    ------
    pandas.csv


    """
    total_course = classement_courses_points()
    # on exprime le nombre de points cumulés par constructeur pour une année donnée
    total_annee = (
        total_course.groupby(["constructorId", "year"])["points"].sum().reset_index()
    )
    return total_annee


def max_point_construct_annee():
    """
    Renvoit une table qui fait correspondre pour chaque année l'écurie
    qui a remporté le plus de points et son nombre de point cumulé

    Return
    ------
    pandas.csv
    """
    total_annee = points_construct_annee()
    # on ne garde dans la table que les points max pour une année donnée
    max_points_par_annee = total_annee.loc[
        total_annee["points"] == total_annee.groupby("year")["points"].transform("max")
    ]
    max_points_par_annee = max_points_par_annee.sort_values(by="year")
    return max_points_par_annee


def reponseQ5():
    """
    cette méthode renvoit le classement des écurie selon le nombre de saisons
    qu'elles ont remporté, sous la forme d'une table
    """
    max_points_par_annee = max_point_construct_annee()
    # on compte le nombre d'années remportées par constructeur
    max_points_par_annee = max_points_par_annee.groupby("constructorId")["year"].size()
    # on récupère le nom du constructeur à partir de constructorId
    max_points_par_annee = pd.merge(
        max_points_par_annee, constructors, on=["constructorId"], how="inner"
    )
    # on simplifie l'affichage
    max_points_par_annee = max_points_par_annee.rename(
        columns={"year": "nombre_saisons"}
    )
    max_points_par_annee.to_csv(
        "resultats/Q5-pandas.csv", index=False, encoding="utf-8"
    )
    # on retourne le classement
    return max_points_par_annee.sort_values(by="nombre_saisons", ascending=False)


# 3) Question python base: faire le classement des écuries selon le nombre de courses remportées


import csv


def Liste_constructor(path: str):
    """
    Prend une table et condense les valeurs d'une colonne donnée

    Parameters
    ----------
    path: str
        le chemin d'accès au fichier csv

    Return
    ------
    list
        list qui recense les valeurs de la colonne
    """
    liste = []
    with open(f"{path}", mode="r", encoding="utf-8", newline="") as results:
        results = csv.reader(results)
        for row in results:
            liste.append(row)
    return liste


def select_element(l: list, indices: list):
    """Simplifie une liste

    Parameters:
    ----------
    liste: list
        liste de listes à simplifier
    indices: list
        indices des éléments à conserver

    Return:
    -------
    list
        liste simplifiée

    """
    if not isinstance(l, list):
        raise TypeError(f"{l} doit être une liste")
    if not isinstance(indices, list):
        raise TypeError(f"{indices} doit être une liste")
    if len(l) == 0:
        raise ValueError("la liste est vide")
    if len(indices) == 0:
        raise ValueError("la liste d'indices est vide")
    if len(indices) > len(l[0]):
        raise ValueError("les indices excèdent la taille de la liste")
    liste = []
    liste_inter = []
    for i in range(len(l)):
        for j in indices:
            liste_inter.append(l[i][j])
        liste.append(liste_inter)
        liste_inter = []

    return liste


def points_courses(path):
    """
    Liste des points accordés à chaque écurie pour chaque course (raceId)

    Parameters
    ----------
    path : str
        Chemin vers le fichier de données

    Returns
    -------
    list of list of float
        Liste contenant, pour chaque course (raceId), une sous-liste de points
    """
    L = Liste_constructor(path)
    L = select_element(
        L, [1, 2, 3]
    )  # Assure-toi que ça donne [raceId, constructorId, points]
    L = L[1:]  # Ignore l'en-tête
    print(L)
    Liste_course = []
    course_id_courant = L[0][0]
    print(course_id_courant)
    points_course_courante = []

    for ligne in L:
        race_id = ligne[0]

        if race_id == course_id_courant:
            points_course_courante.append(float(ligne[2]))

        elif race_id != course_id_courant:
            Liste_course.append(points_course_courante)
            points_course_courante = []
            course_id_courant = ligne[0]

    # ajout de la dernière course
    Liste_course.append(points_course_courante)

    return Liste_course


def participants_courses(path):
    """
    Condense les écuries pour une course donnée
    Parameters
    ----------
    path : str
        chemin d'accès à la table constructor_results
    Return
    ------
    list
        liste qui rencense les écuries pour une course donnée
    """

    Liste_course_inter = []
    Liste_course = []
    L = Liste_constructor(path)
    L = select_element(L, [1, 2, 3])
    L = L[1:]  # Ignore l'en-tête
    Liste_course = []
    course_id_courant = L[0][0]
    points_course_courante = []
    for ligne in L:
        race_id = ligne[0]
        if race_id == course_id_courant:
            points_course_courante.append(float(ligne[1]))
        elif race_id != course_id_courant:
            Liste_course.append(points_course_courante)
            points_course_courante = []
            course_id_courant = ligne[0]
    # ajout de la dernière course
    Liste_course.append(points_course_courante)

    return Liste_course


def longueur_sous_liste(l: list):
    """
    Renvoit la liste des longueurs uniques des sous listes d'une liste

    Parameters
    ----------
    l: list
        liste à traiter

    Return
    ------
    list
        liste des longueurs des sous listes
    """
    if not isinstance(l, list):
        raise TypeError(f"{l} doit être une liste")
    if len(l) == 0:
        raise ValueError("la liste est vide")
    liste_unique = []
    for i in l:
        if len(i) not in liste_unique:
            liste_unique.append(len(i))
    return liste_unique


# on décide ici de supprimer les courses où il n'y a eu qu'un seul participant


def supp_in_list(l: list, v: int):
    """
    Supprime les éléments d'une liste dont la longueur est égale à v

    Parameters
    ----------
    l: list
        liste à traiter
    v: int
        longueur sous-liste à supprimer

    Return
    ------
    list
        liste sans les sous listes de longueur v
    """
    if not isinstance(l, list):
        raise TypeError(f"{l} doit être une liste")
    if len(l) == 0:
        raise ValueError("la liste est vide")
    liste = []
    for i in range(len(l)):
        if len(l[i]) == v:
            liste.append(i)
    for i in liste:
        del l[i]
    return l


def gagnants(path):
    """
    Renvoie la liste des gagnants de chaque course
    Parameter
    ---------
    path: str
        chemin d'accès vers la table constructorId

    Return
    ------
    list
        liste du gagnant pour chaque course
    """
    # supprime les courses avec un seul participant

    points = supp_in_list(points_courses(path), 1)
    participants = supp_in_list(participants_courses(path), 1)  # idem
    gagnants = []
    for i in range(len(points)):
        if len(points[i]) != len(participants[i]):
            print(
                f"Incohérence à l'index {i} : {len(points[i])} points vs {len(participants[i])} participants"
            )
            continue
        else:
            for j in range(len(points[i])):
                if points[i][j] == max(points[i]):
                    gagnants.append(participants[i][j])
    return gagnants


def classement_id(path):
    """
    Classement des id des écuries selon le nombre de victoires cumulées
    Parameter:
    ----------
    path: str
        chemin d'accès vers constructor_results

    Return
    ------
    list
        liste par ordre décroissant des écuries avec leur nombre de victoire associé

    """
    winner = gagnants(path)
    classement = []
    unique = []
    for ecurie in winner:
        if ecurie not in unique:
            unique.append(ecurie)
            classement.append([ecurie, 0])
    for e1 in winner:
        for e2 in classement:
            if e2[0] == e1:
                e2[1] += 1
    classement = sorted(classement, key=lambda x: x[1], reverse=True)
    return classement


def classement_nom(path1, path2):
    """
    Renvoie le classement des écuries selon le nombre de victoires cumulées

    Parameters:
    -----------
    path1: str
        chemin d'accès vers constructor_results
    path2: str
        chemin d'accès vers constructor

    Return
    ------
    list
        liste par ordre décroissant des écuries avec leur nombre de victoire associé
    """
    noms = Liste_constructor(path2)
    noms = select_element(noms, [0, 2])
    classement = classement_id(path1)
    unique = []
    for couple in noms:
        if couple not in unique:
            unique.append(couple)
    for couple in classement:
        for écurie in unique:
            if couple[0] == écurie[0]:
                couple[0] = écurie[1]

    with open("resultats/Q4-python_base.txt", "a", encoding="utf-8") as f:
        f.write("Classement des écuries par nombre de victoires :\n")
        for position, (nom, nb_victoires) in enumerate(classement, 1):
            f.write(f"{position}. {nom} - {nb_victoires} victoires\n")
        f.write("\n" + "-" * 50 + "\n\n")

    return classement
