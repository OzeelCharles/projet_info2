#test Charles.py#
#Comme ce qui a été dit en TP, un seul test sera élaboré sur une seule fonction#
#Si j'ai le temps j'essaierai de le faire sur une seconde fonction#

import pytest
import pandas as pd
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


@pytest.mark.parametrize("nom_pilote, pit_stops, drivers, param, Error",
                         []
)
def test_min_max_pit_stop_drivers(nom_pilote, pit_stops, drivers, param, Error):
    pass