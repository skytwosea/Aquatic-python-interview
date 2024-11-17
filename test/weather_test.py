import pytest
from pathlib import Path
import polars as pl
from math import isclose
from interview import weather

dne_path = "/is/this/even/real.csv"
exists_dir_path = "/home/vesper/Dropbox/dev/python_interview"
exists_invalid_path = "/home/vesper/Dropbox/dev/python_interview/LICENSE.txt"
test_data_path = "/home/vesper/Dropbox/dev/python_interview/data/test_subset.csv"

staged_obj = weather.ChicagoWeather(test_data_path)


# def test_input_good_and_bad_paths():
#     with pytest.raises(TypeError, match=r"missing 1 required positional argument"):
#         _ = weather.ChicagoWeather()
#     with pytest.raises(weather.InputError, match=r"no path provided"):
#         _ = weather.ChicagoWeather(None)  # default arg in main()
#     with pytest.raises(weather.InputError, match=r"does not exist"):
#         _ = weather.ChicagoWeather(dne_path)
#     with pytest.raises(weather.InputError, match=r"not a file"):
#         _ = weather.ChicagoWeather(exists_dir_path)
#     with pytest.raises(weather.InputError, match=r"is not csv"):
#         _ = weather.ChicagoWeather(exists_invalid_path)
#     assert staged_obj.path == Path(test_data_path).resolve()


def test_input_read_target_columns_only():
    expected_colnames = ["Station Name", "Measurement Timestamp", "Measurement ID", staged_obj.target_col]
    intake_df = staged_obj.ingest_and_process(
        parser_pipe=staged_obj._bypass,
        query_pipe=staged_obj._bypass,
    )
    assert len(intake_df.columns) == 4
    assert all([colname in expected_colnames for colname in intake_df.columns])


def test_processing_isolated_parse_datetime_columns():
    # self.id_col = "Measurement ID"
    # self.timestamp_col = "Measurement Timestamp"
    # self.time12_col = "time_12h"
    # self.time24_col = "time_24h"
    tdf = pl.DataFrame(
        {
            "Measurement ID": ["00001111", "00002222", "00003333", "00004444",],
            "Measurement Timestamp": ["1/1/1 11:11", "2/2/2 22:22", "3/3/3 33:33", "4/4/4 44:44"],
        }
    )
    expected_colnames = ["Date", "time_12h", "time_24h",]
    tdf_time_lazy = staged_obj._process_time_columns(tdf)
    assert isinstance(tdf_time_lazy, pl.LazyFrame)
    tdf_time_processed = tdf_time_lazy.collect()
    assert len(tdf_time_processed.columns) == 3
    assert all([colname in expected_colnames for colname in tdf_time_processed.columns])
    assert (
        tdf_time_processed.lazy()
        .filter(pl.col("time_12h").str.contains("^22:22$"))
        .with_columns(
            pl.when(pl.col("time_24h") == 2222)
            .then(pl.lit(True))
            .otherwise(pl.lit(False))
            .alias("time_check")
        )
        .select("time_check")
        .unique()
    ).collect().height == 1


def test_processing_workflow_parse_datetime_columns():
    expected_colnames = ["Station Name", "Date", "time_12h", "time_24h", staged_obj.target_col]
    reworked_time_df = staged_obj.ingest_and_process(
        parser_pipe=staged_obj._process_time_columns,
        query_pipe=staged_obj._bypass,
    )
    assert len(reworked_time_df.columns) == 5
    assert all([colname in expected_colnames for colname in reworked_time_df.columns])
    assert (
        reworked_time_df.lazy()
        .filter(pl.col("time_12h").str.contains("^10:00:00 PM$"))
        .with_columns(
            pl.when(pl.col("time_24h") == 2200)
            .then(pl.lit(True))
            .otherwise(pl.lit(False))
            .alias("time_check")
        )
        .select("time_check")
        .unique()
    ).collect().height == 1  # if any mismatches registered False, height would be 2


def test_query_workflow_min_max_first_last():
    query_df = staged_obj.ingest_and_process(
        parser_pipe=staged_obj._process_time_columns,
        query_pipe=staged_obj._query_min_max_first_last,
    )
    assert len(query_df.columns) == 6
    tdf_a = (
        query_df
        .filter(pl.col("Date").str.contains("^12/30/2016$"))
        .filter(pl.col("Station Name").str.contains("^Foster"))
    )
    tdf_b = (
        query_df
        .filter(pl.col("Date").str.contains("^12/31/2016$"))
        .filter(pl.col("Station Name").str.contains("^Oak Street"))
    )
    assert isclose(tdf_a.select("First Temp").item(), -0.39, rel_tol=1e-3)
    assert isclose(tdf_a.select("Last Temp").item(), 0.06, rel_tol=1e-3)
    assert isclose(tdf_a.select("Min Temp").item(), -3.5, rel_tol=1e-3)
    assert isclose(tdf_a.select("Max Temp").item(), 3.17, rel_tol=1e-3)
    assert isclose(tdf_b.select("First Temp").item(), 4.9, rel_tol=1e-3)
    assert isclose(tdf_b.select("Last Temp").item(), 4.1, rel_tol=1e-3)
    assert isclose(tdf_b.select("Min Temp").item(), -0.3, rel_tol=1e-3)
    assert isclose(tdf_b.select("Max Temp").item(), 6.3, rel_tol=1e-3)
