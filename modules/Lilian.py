import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from pathlib import Path
from datetime import datetime
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score


def sauvegarder_figure(prefix="graph"):
    desktop = Path.home() / "Desktop"
    dossier_resultat = desktop / "résultat"
    dossier_resultat.mkdir(exist_ok=True)
    horodatage = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    nom_fichier = dossier_resultat / f"{prefix}_{horodatage}.png"
    plt.savefig(nom_fichier)
    plt.show()
    print(f"Graphique sauvegardé : {nom_fichier}")


def sauvegarder_tableau_csv(df, prefix="tableau"):
    desktop = Path.home() / "Desktop"
    dossier_resultat = desktop / "résultat"
    dossier_resultat.mkdir(exist_ok=True)
    horodatage = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    nom_fichier = dossier_resultat / f"{prefix}_{horodatage}.csv"
    df.to_csv(nom_fichier, index=False)
    print(f"Tableau sauvegardé : {nom_fichier}")


def charger_donnees(chemin_races, chemin_resultats, chemin_pilotes, chemin_ecuries, chemin_statuts):
    races = pd.read_csv(chemin_races)
    resultats = pd.read_csv(chemin_resultats)
    pilotes = pd.read_csv(chemin_pilotes)
    ecuries = pd.read_csv(chemin_ecuries)
    statuts = pd.read_csv(chemin_statuts)
    return races, resultats, pilotes, ecuries, statuts


def construire_base_multi_saisons(races, resultats, pilotes, statuts, annee_debut=2013, annee_fin=2023):
    courses_filtrees = races[races["year"].between(annee_debut, annee_fin)]
    resultats_filtrees = resultats[resultats["raceId"].isin(courses_filtrees["raceId"])].copy()
    resultats_filtrees = resultats_filtrees.merge(courses_filtrees[["raceId", "year"]], on="raceId", how="left")
    mots_cles_abandon = ["Accident", "Collision", "Engine", "Gearbox", "Retired", "Fire", "Electrical",
                         "Suspension", "Overheating", "Clutch", "Hydraulics", "Brake", "Oil", "Fuel"]
    ids_abandon = statuts[statuts["status"].str.contains('|'.join(mots_cles_abandon), case=False)]["statusId"].astype(str).tolist()
    resultats_filtrees["positionOrder"] = pd.to_numeric(resultats_filtrees["positionOrder"], errors="coerce")
    points = resultats_filtrees.groupby(["year", "driverId"])["points"].sum().reset_index(name="points_totaux")
    nb_courses = resultats_filtrees.groupby(["year", "driverId"]).size().reset_index(name="nb_courses")
    nb_abandons = resultats_filtrees[resultats_filtrees["statusId"].astype(str).isin(ids_abandon)]         .groupby(["year", "driverId"]).size().reset_index(name="nb_abandons")
    position_moyenne = resultats_filtrees.groupby(["year", "driverId"])["positionOrder"].mean().reset_index(name="position_moyenne")
    compte_positions = resultats_filtrees.pivot_table(index=["year", "driverId"], columns="positionOrder",
                                                      aggfunc="size", fill_value=0)
    compte_positions.columns = [f"pos_{int(col)}" for col in compte_positions.columns]
    base = points.merge(nb_courses, on=["year", "driverId"], how="left")                  .merge(nb_abandons, on=["year", "driverId"], how="left")                  .merge(position_moyenne, on=["year", "driverId"], how="left")                  .merge(compte_positions, on=["year", "driverId"], how="left")                  .merge(pilotes[["driverId", "forename", "surname"]], on="driverId", how="left")
    base["nb_abandons"] = base["nb_abandons"].fillna(0)
    base["NomComplet"] = base["forename"] + " " + base["surname"] + " (" + base["year"].astype(str) + ")"
    return base


def entrainer_modele_regression(base, variables_explicatives=None):
    if variables_explicatives is None:
        variables_explicatives = ["pos_1", "pos_2", "pos_3"]
    X = base[variables_explicatives]
    y = base["points_totaux"]
    noms = base["NomComplet"]
    X_train, X_test, y_train, y_test, noms_train, noms_test = train_test_split(
        X, y, noms, test_size=0.2, random_state=42
    )
    modele = LinearRegression()
    modele.fit(X_train, y_train)
    y_pred = modele.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    comparaison = pd.DataFrame({
        "NomComplet": noms_test.values,
        "Réel": y_test.values,
        "Prédit": y_pred.round(1)
    })
    return modele, mse, r2, comparaison


def afficher_scatter_comparaison(comparaison, titre="Prédictions vs Réalité", seuil_filtrage=None):
    if seuil_filtrage is not None:
        comparaison = comparaison[comparaison["Réel"] > seuil_filtrage]
    plt.figure(figsize=(14, 10))
    plt.scatter(comparaison["Réel"], comparaison["Prédit"], alpha=0.7)
    for i in range(len(comparaison)):
        plt.text(comparaison["Réel"].iloc[i]+3, comparaison["Prédit"].iloc[i]+3,
                 comparaison["NomComplet"].iloc[i], fontsize=7)
    max_val = max(comparaison["Réel"].max(), comparaison["Prédit"].max())
    plt.plot([0, max_val], [0, max_val], linestyle='--', color='grey')
    plt.xlabel("Points réels")
    plt.ylabel("Points prédits")
    plt.title("Régression : " + titre)
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def afficher_classement_2023(resultats, pilotes):
    resultats_2023 = resultats[resultats["raceId"] > 1070].copy()
    resultats_2023["positionOrder"] = pd.to_numeric(resultats_2023["positionOrder"], errors="coerce")
    points = resultats_2023.groupby("driverId")["points"].sum().reset_index(name="points_totaux")
    podiums = resultats_2023[resultats_2023["positionOrder"].isin([1, 2, 3])]
    nb_courses = resultats_2023.groupby("driverId").size().reset_index(name="nb_courses")
    nb_podiums = podiums.groupby("driverId").size().reset_index(name="nb_podiums")
    pos_moy = resultats_2023.groupby("driverId")["positionOrder"].mean().reset_index(name="pos_moy")
    pos_counts = resultats_2023[resultats_2023["positionOrder"].isin([1,2,3])].pivot_table(
        index="driverId", columns="positionOrder", aggfunc="size", fill_value=0
    )
    pos_counts.columns = [f"pos_{int(c)}" for c in pos_counts.columns]
    fusion = points.merge(nb_courses, on="driverId", how="left")                    .merge(nb_podiums, on="driverId", how="left")                    .merge(pos_moy, on="driverId", how="left")                    .merge(pos_counts, on="driverId", how="left")                    .merge(pilotes[["driverId", "forename", "surname"]], on="driverId", how="left")
    fusion["FullName"] = fusion["forename"] + " " + fusion["surname"]
    fusion = fusion.sort_values(by="points_totaux", ascending=False)
    df_affichage = fusion[["FullName", "points_totaux", "nb_courses", "nb_podiums", "pos_1", "pos_2", "pos_3", "pos_moy"]]
    print(df_affichage.head(10))
    sauvegarder_tableau_csv(df_affichage.head(10), prefix="classement_2023")
    top10 = fusion.head(10).copy()
    cmap = LinearSegmentedColormap.from_list("red_orange", ["#ff4c4c", "#ffae42"])
    plt.figure(figsize=(10, 6))
    sns.barplot(data=top10, y="FullName", x="points_totaux", palette=cmap(np.linspace(0, 1, len(top10))))
    plt.xlabel("Points totaux")
    plt.ylabel("Pilote")
    plt.title("Top 10 pilotes F1 - Saison 2023")
    plt.grid(axis="x", linestyle="--", alpha=0.5)
    plt.tight_layout()
    sauvegarder_figure(prefix="classement_2023")


def afficher_abandons_pilotes_ecuries(resultats, pilotes, ecuries, statuts):
    resultats_2023 = resultats[resultats["raceId"] > 1070].copy()
    abandon_keywords = ["Accident", "Collision", "Engine", "Gearbox", "Retired", "Fire", "Electrical",
                        "Suspension", "Overheating", "Clutch", "Hydraulics", "Brake", "Oil", "Fuel"]
    abandon_ids = statuts[statuts["status"].str.contains('|'.join(abandon_keywords), case=False)]["statusId"].astype(str).tolist()
    abandons_pilotes = resultats_2023[resultats_2023["statusId"].astype(str).isin(abandon_ids)]         .groupby("driverId").size().reset_index(name="nb_abandons")         .merge(pilotes[["driverId", "forename", "surname"]], on="driverId")         .assign(FullName=lambda df: df["forename"] + " " + df["surname"])         .sort_values(by="nb_abandons", ascending=False)
    df_pilotes = abandons_pilotes[["FullName", "nb_abandons"]].head(10)
    print(df_pilotes)
    sauvegarder_tableau_csv(df_pilotes, prefix="abandons_pilotes")
    top_pilotes = abandons_pilotes.head(10)
    cmap = LinearSegmentedColormap.from_list("red_orange", ["#ff4c4c", "#ffae42"])
    plt.figure(figsize=(10,6))
    sns.barplot(data=top_pilotes, y="FullName", x="nb_abandons", palette=cmap(np.linspace(0, 1, len(top_pilotes))))
    plt.title("Top 10 pilotes avec le plus d'abandons (2023)")
    plt.xlabel("Nombre d'abandons")
    plt.ylabel("Pilote")
    plt.grid(axis="x", linestyle="--", alpha=0.5)
    plt.tight_layout()
    sauvegarder_figure(prefix="abandons_pilotes")
    abandons_ecuries = resultats_2023[resultats_2023["statusId"].astype(str).isin(abandon_ids)]         .groupby("constructorId").size().reset_index(name="nb_abandons")         .merge(ecuries[["constructorId", "name"]], on="constructorId")         .sort_values(by="nb_abandons", ascending=False)
    df_ecuries = abandons_ecuries[["name", "nb_abandons"]].head(10)
    print(df_ecuries)
    sauvegarder_tableau_csv(df_ecuries, prefix="abandons_ecuries")
    top_ecuries = abandons_ecuries.head(10)
    plt.figure(figsize=(10,6))
    sns.barplot(data=top_ecuries, y="name", x="nb_abandons", palette=cmap(np.linspace(0, 1, len(top_ecuries))))
    plt.title("Top 10 écuries avec le plus d'abandons (2023)")
    plt.xlabel("Nombre d'abandons")
    plt.ylabel("Écurie")
    plt.grid(axis="x", linestyle="--", alpha=0.5)
    plt.tight_layout()
    sauvegarder_figure(prefix="abandons_ecuries")


def afficher_regression_multi(races, resultats, pilotes, statuts):
    base = construire_base_multi_saisons(races, resultats, pilotes, statuts)
    modele, mse, r2, comparaison = entrainer_modele_regression(base)
    print(f"RMSE : {mse:.2f}")
    print(f"R² : {r2:.3f}")
    afficher_scatter_comparaison(comparaison, titre="Multi-saisons (> 20 pts)", seuil_filtrage=20)
    sauvegarder_figure(prefix="regression_multi")