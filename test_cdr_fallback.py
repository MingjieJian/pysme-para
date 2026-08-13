import numpy as np
import pandas as pd

from pysme_para.pysme_abund import _has_valid_line_ranges


def test_has_valid_line_ranges_rejects_missing_columns():
    line_list = pd.DataFrame({"wlcent": [5000.0]})

    assert not _has_valid_line_ranges(line_list)


def test_has_valid_line_ranges_rejects_all_nan_ranges():
    line_list = pd.DataFrame(
        {
            "line_range_s": [np.nan, np.nan],
            "line_range_e": [np.nan, np.nan],
        }
    )

    assert not _has_valid_line_ranges(line_list)


def test_has_valid_line_ranges_accepts_any_finite_ranges():
    line_list = pd.DataFrame(
        {
            "line_range_s": [np.nan, 4999.0],
            "line_range_e": [np.nan, 5001.0],
        }
    )

    assert _has_valid_line_ranges(line_list)
