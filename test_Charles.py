#test Charles.py#
#Comme ce qui a été dit en TP, un seul test sera élaboré sur une seule fonction#
#Si j'ai le temps j'essaierai de le faire sur une seconde fonction#

import pandas as pd
import pytest
from Charles import min_max_pit_stop_drivers

drivers_df = pd.DataFrame({
    "driverId": [1, 2],
    "driverRef": ["hamilton", "verstappen"]
})

pit_stops_df = pd.DataFrame({
    "raceId": [101, 101, 102, 102, 103, 103],
    "driverId": [1, 1, 1, 1, 2, 2],
    "milliseconds": [25000, 24000, 23000, 26000, 22000, 28000]
})

# Jeu de tests paramétrés
@pytest.mark.parametrize(
    "nom_pilote, pit_stops, drivers, param, Error",
    [
        # Cas valides
        ("hamilton", pit_stops_df, drivers_df, "min", None),
        ("hamilton", pit_stops_df, drivers_df, "max", None),
        ("verstappen", pit_stops_df, drivers_df, "min", None),
        ("verstappen", pit_stops_df, drivers_df, "max", None),

        # Param incorrect
        ("hamilton", pit_stops_df, drivers_df, "moyenne", ValueError),

        # Colonnes manquantes dans pit_stops
        ("hamilton", pit_stops_df.drop(columns=["milliseconds"]), drivers_df, "min", ValueError),
        ("hamilton", pit_stops_df.drop(columns=["driverId"]), drivers_df, "min", ValueError),

        # Colonnes manquantes dans drivers
        ("hamilton", pit_stops_df, drivers_df.drop(columns=["driverRef"]), "min", ValueError),

        # milliseconds non convertibles
        ("hamilton", 
         pd.DataFrame({
             "raceId": [101, 101],
             "driverId": [1, 1],
             "milliseconds": ["non_numeric", "invalid"]
         }), 
         drivers_df, 
         "min", 
         ValueError
        ),
    ]
)
def test_min_max_pit_stop_drivers(nom_pilote, pit_stops, drivers, param, Error):
    if Error is not None:
        with pytest.raises(Error):
            min_max_pit_stop_drivers(nom_pilote, pit_stops, drivers, param)
    else:
        result = min_max_pit_stop_drivers(nom_pilote, pit_stops, drivers, param)
        assert isinstance(result, pd.DataFrame)
        assert not result.empty
