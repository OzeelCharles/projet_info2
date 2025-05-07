# Questions de Jules
# coding: utf-8

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.nonparametric.smoothers_lowess import lowess

# # Fonctions Question 1


def filtrer_vainqueurs(results: pd.DataFrame) -> pd.DataFrame:
    """
    Garde uniquement les résultats des pilotes ayant terminé premiers.

    Parameters
    ----------
    results : pd.DataFrame
        Table des résultats des courses.

    Returns
    -------
    pd.DataFrame
        Résultats filtrés pour les vainqueurs.
    """
    return results[results["positionOrder"] == 1]


def compter_victoires(
    winners: pd.DataFrame, drivers: pd.DataFrame, seuil: int = 30
) -> pd.DataFrame:
    """
    Compte le nombre de victoires par pilote et filtre ceux avec au moins un seuil donné.

    Parameters
    ----------
    winners : pd.DataFrame
        Résultats filtrés pour les vainqueurs.
    drivers : pd.DataFrame
        Données des pilotes.
    seuil : int, optional
        Seuil minimal de victoires (default = 30).

    Returns
    -------
    pd.DataFrame
        Table des pilotes avec leur nombre de victoires.
    """
    victory_counts = winners["driverId"].value_counts()
    victory_counts = victory_counts[victory_counts >= seuil]
    top_winners = pd.DataFrame(
        {"driverId": victory_counts.index, "wins": victory_counts.values}
    )
    top_winners = top_winners.merge(drivers, on="driverId")
    return top_winners


def plot_victoires(top_winners: pd.DataFrame) -> None:
    """
    Affiche un graphique du nombre de victoires des pilotes sélectionnés.

    Parameters
    ----------
    top_winners : pd.DataFrame
        Table des pilotes avec leur nombre de victoires.

    Returns
    -------
    None
    """
    plt.figure(figsize=(12, 7))
    sns.set(style="whitegrid")
    sorted_data = top_winners.sort_values("wins", ascending=False)
    sns.barplot(x="wins", y="surname", data=sorted_data, palette="viridis")
    plt.title("Pilotes ayant remporté au moins 30 Grands Prix", fontsize=16)
    plt.xlabel("Nombre de victoires")
    plt.ylabel("Pilote")
    plt.show()


def stats_victoires(top_winners: pd.DataFrame) -> None:
    """
    Affiche les statistiques descriptives sur les victoires.

    Parameters
    ----------
    top_winners : pd.DataFrame
        Table des pilotes avec leur nombre de victoires.

    Returns
    -------
    None
    """
    print(top_winners["wins"].describe())
    print("\nMédiane :", top_winners["wins"].median())
    print("Écart-type :", top_winners["wins"].std())


def lire_drivers_depuis_table(table: list[dict]) -> dict:
    """
    Extrait un dictionnaire {driverId: "Nom Prénom"} depuis une table déjà chargée.

    Parameters
    ----------
    table : list of dict
        Table contenant les colonnes 'driverId', 'forename', 'surname'.

    Returns
    -------
    dict
        Dictionnaire des noms complets indexés par driverId.
    """
    drivers = {}
    for row in table:
        drivers[str(row["driverId"])] = row["forename"] + " " + row["surname"]
    return drivers


def compter_victoires_depuis_table(table: list[dict]) -> dict:
    """
    Compte les victoires (positionOrder = 1) depuis une table déjà chargée.

    Parameters
    ----------
    table : list of dict
        Résultats complets avec champs 'driverId' et 'positionOrder'.

    Returns
    -------
    dict
        Dictionnaire {driverId: nombre_de_victoires}
    """
    counts = {}
    for row in table:
        if str(row["positionOrder"]) == "1":
            driver_id = str(row["driverId"])
            counts[driver_id] = counts.get(driver_id, 0) + 1
    return counts


def filtrer_pilotes(victory_counts: dict, drivers: dict, seuil: int = 30) -> list:
    """
    Filtre les pilotes ayant atteint au moins un certain nombre de victoires.

    Parameters
    ----------
    victory_counts : dict
        Nombre de victoires par driverId.
    drivers : dict
        Correspondance driverId -> nom complet.
    seuil : int
        Nombre minimal de victoires.

    Returns
    -------
    list of tuple
        Liste triée de (nom, victoires) pour les pilotes ayant atteint le seuil.
    """
    filtered = []
    for driver_id, wins in victory_counts.items():
        if wins >= seuil and driver_id in drivers:
            filtered.append((drivers[driver_id], wins))
    filtered.sort(key=lambda x: x[1], reverse=True)
    return filtered


# # Fonctions Question vitesse


def compute_speed_per_year(results: pd.DataFrame, races: pd.DataFrame) -> pd.DataFrame:
    """
    Calcule la vitesse moyenne des vainqueurs de F1 par année.

    Parameters
    ----------
    results : pd.DataFrame
        Résultats des pilotes.
    races : pd.DataFrame
        Informations sur les courses.

    Returns
    -------
    pd.DataFrame
        Moyenne annuelle des vitesses (km/h).
    """
    winners = results[results["positionOrder"] == 1]
    merged = winners.merge(races, on="raceId")
    merged["milliseconds"] = pd.to_numeric(merged["milliseconds"], errors="coerce")
    merged = merged.dropna(subset=["milliseconds"])
    merged["hours"] = merged["milliseconds"] / (1000 * 60 * 60)
    merged["speed_kmh"] = 305 / merged["hours"]
    merged = merged[merged["speed_kmh"] <= 350]
    return merged.groupby("year")["speed_kmh"].mean().reset_index()


def plot_speed_evolution_improved(df_speed: pd.DataFrame) -> None:
    """
    Affiche l'évolution avec régression linéaire + intervalle de confiance.

    Parameters
    ----------
    df_speed : pd.DataFrame
        Contient les colonnes 'year' et 'speed_kmh'.

    Returns
    -------
    None
    """
    sns.set(style="whitegrid")
    plt.figure(figsize=(14, 7))
    x = df_speed["year"].values
    y = df_speed["speed_kmh"].values
    coef = np.polyfit(x, y, 1)
    trend = np.poly1d(coef)
    y_pred = trend(x)
    residuals = y - y_pred
    std_error = np.std(residuals)
    sns.lineplot(x=x, y=y, label="Vitesse moyenne", marker="o")
    plt.plot(x, y_pred, label="Régression (linéaire)", color="red", linewidth=2)
    plt.fill_between(
        x,
        y_pred - std_error,
        y_pred + std_error,
        color="red",
        alpha=0.2,
        label="Intervalle de confiance",
    )
    plt.title("Évolution de la vitesse moyenne des vainqueurs de F1", fontsize=16)
    plt.xlabel("Année")
    plt.ylabel("Vitesse (km/h)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def plot_speed_evolution_lowess(df_speed, frac=0.25, smoother=None):
    """
    Affiche un lissage LOWESS sur l'évolution de la vitesse moyenne des vainqueurs.

    Parameters
    ----------
    df_speed : pandas.DataFrame
        Tableau avec les colonnes 'year' et 'speed_kmh'.
    frac : float
        Fraction de données utilisées pour le lissage.
    smoother : fonction
        La fonction LOWESS à utiliser (doit être passée depuis le notebook).

    Returns
    -------
    None
    """
    if smoother is None:
        raise ValueError("Veuillez fournir la fonction 'lowess' via le paramètre 'smoother'.")

    x = df_speed["year"].values
    y = df_speed["speed_kmh"].values

    smoothed = smoother(endog=y, exog=x, frac=frac, return_sorted=True)

    plt.figure(figsize=(12, 6))
    plt.plot(x, y, "o", label="Données brutes", alpha=0.6)
    plt.plot(smoothed[:, 0], smoothed[:, 1], color="red", label="LOWESS")
    plt.title("Lissage LOWESS de la vitesse moyenne")
    plt.xlabel("Année")
    plt.ylabel("Vitesse moyenne (km/h)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def reponseQ6(results, drivers):
    """
    Affiche le classement des pilotes ayant remporté le plus de Grands Prix,
    à l'aide d'une version avec pandas, puis d'une version sans pandas.

    Parameters
    ----------
    results : pandas.DataFrame
        Données des résultats de courses, incluant la position d'arrivée et les identifiants des pilotes.
    drivers : pandas.DataFrame
        Données sur les pilotes, incluant leurs identifiants, prénoms et noms.

    Returns
    -------
    None
        Affiche le tableau des pilotes ayant au moins 30 victoires (avec pandas),
        le graphique associé, des statistiques descriptives, puis une version texte
        équivalente construite sans utiliser pandas.
    """
    print("→ Classement des pilotes ayant remporté au moins 30 courses (avec pandas) :\n")
    winners = filtrer_vainqueurs(results)
    top_winners = compter_victoires(winners, drivers)
    top_winners = top_winners.sort_values("wins", ascending=False).loc[:, ["surname", "wins"]]
    print(top_winners)
    plot_victoires(top_winners)
    stats_victoires(top_winners)

    print("\n→ Classement (version sans pandas) :\n")
    drivers_dict = lire_drivers_depuis_table(drivers.to_dict("records"))
    victory_counts = compter_victoires_depuis_table(results.to_dict("records"))
    top_pilotes = filtrer_pilotes(victory_counts, drivers_dict)

    for nom, victoires in top_pilotes:
        print(f"{nom} : {victoires} victoires")


def reponseQ7(results, races):
    """
    Calcule et affiche l'évolution de la vitesse moyenne des vainqueurs de Grands Prix par année.
    Deux visualisations sont générées : une régression linéaire et un lissage LOWESS.

    Parameters
    ----------
    results : pandas.DataFrame
        Résultats des courses, incluant le temps de course (en millisecondes) et les identifiants de course.
    races : pandas.DataFrame
        Données sur les courses, incluant les années et les identifiants de course.

    Returns
    -------
    None
        Affiche un aperçu des vitesses moyennes par année et produit deux graphiques montrant
        l'évolution temporelle (régression linéaire + lissage non paramétrique).
    """
    df_speed = compute_speed_per_year(results, races)
    print(df_speed.head())
    print("\n→ Graphique avec régression linéaire :")
    plot_speed_evolution_improved(df_speed)
    print("\n→ Graphique avec lissage LOWESS :")
    plot_speed_evolution_lowess(df_speed, smoother=lowess)
