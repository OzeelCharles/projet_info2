# test Charles.py#
# Comme ce qui a été dit en TP, un seul test sera élaboré sur une seule fonction#
# Si j'ai le temps j'essaierai de le faire sur une seconde fonction#

import pandas as pd
import pytest
from modules.Charles import ttal_vict_pts_table, ttal_vict_pts_pilote

@pytest.mark.parametrize(
    "results, drivers, expected",
    [
        (
            pd.DataFrame([
                {"driverId": 1, "raceId": 101, "positionText": "1", "points": 25},
                {"driverId": 1, "raceId": 102, "positionText": "2", "points": 18},
                {"driverId": 2, "raceId": 101, "positionText": "1", "points": 25},
                {"driverId": 2, "raceId": 102, "positionText": "1", "points": 25},
            ]),
            pd.DataFrame([
                {"driverId": 1, "driverRef": "hamilton"},
                {"driverId": 2, "driverRef": "verstappen"},
            ]),
            pd.DataFrame([
                {"driverRef": "hamilton", "points": 43, "win": 1, "raceId": 2, "victory_rate": 0.5},
                {"driverRef": "verstappen", "points": 50, "win": 2, "raceId": 2, "victory_rate": 1.0},
            ])
        )
    ]
)
def test_ttal_vict_pts_table(results, drivers, expected):
    result_df = ttal_vict_pts_table(results, drivers)
    pd.testing.assert_frame_equal(result_df.sort_values("driverRef").reset_index(drop=True),
                                  expected.sort_values("driverRef").reset_index(drop=True))


@pytest.mark.parametrize(
    "pilote, results, drivers, expected",
    [
        (
            "hamilton",
            pd.DataFrame([
                {"driverId": 1, "raceId": 101, "positionText": "1", "points": 25},
                {"driverId": 1, "raceId": 102, "positionText": "2", "points": 18},
            ]),
            pd.DataFrame([
                {"driverId": 1, "driverRef": "hamilton"},
            ]),
            [43.0, 1, 2, 0.5]
        ),
        (
            "verstappen",
            pd.DataFrame([
                {"driverId": 2, "raceId": 101, "positionText": "1", "points": 25},
                {"driverId": 2, "raceId": 102, "positionText": "1", "points": 25},
            ]),
            pd.DataFrame([
                {"driverId": 2, "driverRef": "verstappen"},
            ]),
            [50.0, 2, 2, 1.0]
        ),
    ]
)
def test_ttal_vict_pts_pilote(pilote, results, drivers, expected):
    assert ttal_vict_pts_pilote(pilote, results, drivers) == expected