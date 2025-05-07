import pandas as pd
import pytest
from Jules import compter_victoires

@pytest.mark.parametrize(
    "winners, drivers, expected",
    [
        (
            pd.DataFrame([
                {"driverId": 1},
                {"driverId": 2},
                {"driverId": 2},
                {"driverId": 3},
                {"driverId": 3},
                {"driverId": 3}
            ]),
            pd.DataFrame([
                {"driverId": 1, "surname": "Senna"},
                {"driverId": 2, "surname": "Prost"},
                {"driverId": 3, "surname": "Lauda"},
            ]),
            pd.DataFrame([
                {"driverId": 3, "wins": 3, "surname": "Lauda"},
                {"driverId": 2, "wins": 2, "surname": "Prost"},
            ])
        )
    ]
)
def test_compter_victoires(winners, drivers, expected):
    result = compter_victoires(winners, drivers, seuil=2)
    pd.testing.assert_frame_equal(
        result.sort_values("driverId").reset_index(drop=True),
        expected.sort_values("driverId").reset_index(drop=True)
    )

from Jules import filtrer_pilotes

@pytest.mark.parametrize(
    "victory_counts, drivers_dict, seuil, expected",
    [
        (
            {"1": 5, "2": 3, "3": 1},
            {"1": "Senna", "2": "Prost", "3": "Lauda"},
            2,
            [("Senna", 5), ("Prost", 3)]
        )
    ]
)
def test_filtrer_pilotes(victory_counts, drivers_dict, seuil, expected):
    result = filtrer_pilotes(victory_counts, drivers_dict, seuil)
    assert result == expected