import pandas as pd


results = pd.read_csv("results.csv")
drivers = pd.read_csv("drivers.csv")

victoires = results[results["positionOrder"] == 1]

nombre_victoire = victoires["driverId"].value_counts()

nombre_victoire = nombre_victoire[nombre_victoire >= 30]

top_winners = pd.DataFrame({
    "driverId": nombre_victoire.index,
    "wins": nombre_victoire.values
}).merge(drivers, on="driverId")

top_winners["name"] = top_winners["forename"] + " " + top_winners["surname"]
print(top_winners[["name", "wins"]].sort_values(by="wins", ascending=False))