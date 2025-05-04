# Questions de Jules
# coding: utf-8

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

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


def plot_speed_evolution_lowess(df_speed: pd.DataFrame, frac: float = 0.2) -> None:
    """
    Affiche l'évolution de la vitesse moyenne des vainqueurs avec un lissage LOWESS.

    Parameters
    ----------
    df_speed : pd.DataFrame
        Données contenant 'year' et 'speed_kmh'.
    frac : float
        Proportion des données utilisées pour chaque régression locale (entre 0 et 1).

    Returns
    -------
    None
    """
    sns.set(style="whitegrid")
    plt.figure(figsize=(14, 7))

    x = df_speed["year"].values
    y = df_speed["speed_kmh"].values

    # Application du lissage LOWESS
    smoothed = lowess(endog=y, exog=x, frac=frac, return_sorted=True)

    # Tracés
    plt.plot(x, y, "o", label="Données brutes", alpha=0.6)
    plt.plot(
        smoothed[:, 0],
        smoothed[:, 1],
        color="green",
        linewidth=2.5,
        label="LOWESS (lissage local)",
    )

    plt.title(
        "Évolution lissée de la vitesse moyenne des vainqueurs de F1", fontsize=16
    )
    plt.xlabel("Année")
    plt.ylabel("Vitesse moyenne (km/h)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
