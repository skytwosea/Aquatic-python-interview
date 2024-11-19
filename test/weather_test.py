import pytest
from pathlib import Path
import polars as pl
from math import isclose
from interview import weather
import sys
from collections import Counter
import io

dne_path = "/is/this/even/real.csv"
exists_dir_path = "/home/vesper/Dropbox/dev/python_interview"
exists_invalid_path = "/home/vesper/Dropbox/dev/python_interview/LICENSE.txt"
test_reader_path = "/home/vesper/Dropbox/dev/python_interview/data/test_subset.csv"
staged = weather.Weather(test_reader_path, sys.stdout)


def test_io_in_good_and_bad_paths(tmp_path):
    with pytest.raises(TypeError, match=r"missing 2 required"):
        _ = weather.Weather()
    with pytest.raises(weather.InputError, match=r"does not exist"):
        _ = weather.Weather(dne_path, tmp_path/"out")
    with pytest.raises(weather.InputError, match=r"not a file"):
        _ = weather.Weather(exists_dir_path, tmp_path/"out")
    with pytest.raises(weather.InputError, match=r"is not a csv"):
        _ = weather.Weather(exists_invalid_path, tmp_path/"out")
    assert staged._reader == Path(test_reader_path).resolve()

def test_io_out_good_and_bad_paths(tmp_path):
    with pytest.raises(TypeError, match=r"missing 2 required"):
        _ = weather.Weather()
    with pytest.raises(weather.InputError, match=r"does not exist"):
        _ = weather.Weather(test_reader_path, dne_path)
    with pytest.raises(weather.InputError, match=r"not a file"):
        _ = weather.Weather(test_reader_path, exists_dir_path)
    with pytest.raises(weather.InputError, match=r"is not a csv"):
        _ = weather.Weather(test_reader_path, exists_invalid_path)
    assert staged._writer == sys.stdout

def test_io_bad_column_name():
    with pytest.raises(weather.InputError, match=r"column not in file"):
        _ = weather.Weather(test_reader_path, sys.stdout, "aardvaark")
    assert staged._target_col == "Air Temperature"

def test_io_streaming_input():
    with open(test_reader_path, 'r') as f:
        # rawtext = '\n'.join(f.readlines()).strip()
        rawtext = f.read()
    reader = io.StringIO(rawtext)
    writer = io.StringIO()
    stream_obj = weather.Weather(reader, writer)
    stream_obj._read()
    stream_obj._write()
    out = writer.getvalue()
    stream_obj._deinit()
    assert out == "Station Name,Date,Min Temp,Max Temp,First Temp,Last Temp\n63rd Street Weather Station,12/30/2016,-2.8,3.6,0.3,0.6\nFoster Weather Station,12/30/2016,-3.5,3.17,-0.39,0.06\nOak Street Weather Station,12/30/2016,-2.3,4.2,0.7,0.8\n63rd Street Weather Station,12/31/2016,-1.3,5.6,4.4,3.5\nFoster Weather Station,12/31/2016,-1.56,5.17,3.67,3.0\nOak Street Weather Station,12/31/2016,-0.3,6.3,4.9,4.1\n"

def test_io_compare_stream_and_file_inputs():
    # stream first:
    with open(test_reader_path, 'r') as f:
        rawtext = f.read()
    sreader = io.StringIO(rawtext)
    swriter = io.StringIO()
    stream_obj = weather.Weather(sreader, swriter)
    stream_obj._read()
    stream_obj._write()
    sout = swriter.getvalue()
    stream_obj._deinit()
    # file next:
    freader = test_reader_path
    fwriter = io.StringIO()
    file_obj = weather.Weather(freader, fwriter)
    file_obj._read()
    file_obj._write()
    fout = fwriter.getvalue()
    file_obj._deinit()
    assert sout == fout

def test_init_state_label_cols():
    assert len(staged._label_cols) == 4
    assert staged._label_cols[0] == "Min Temp"

def test_init_state_header_indexes_map():
    assert len(list(staged._header_indexes_map.keys())) == 18
    assert staged._header_indexes_map["Station Name"] == 0
    assert staged._header_indexes_map["Measurement ID"] == 17

def test_init_state_schema_overrides_map():
    assert len(list(staged._schema_overrides_map.keys())) == 18
    assert staged._schema_overrides_map["Station Name"] == pl.String
    assert staged._schema_overrides_map["Humidity"] == pl.Float64

def test_init_state_batch_limit():
    assert staged._batch_line_limit == 10000

def test_input_read_from_file():
    expected_colnames = ["Station Name", "Measurement Timestamp", "Measurement ID", staged._target_col]
    staged.set_parser_pipe(staged.__bypass__)
    staged.set_query_pipe(staged.__bypass__)
    _ = staged._process_input_file()
    assert isinstance(staged._processed_dataframes, pl.DataFrame)
    assert len(staged._processed_dataframes.columns) == 4
    assert all([colname in expected_colnames for colname in staged._processed_dataframes.columns])

def test_input_batched_stream_handler():
    pass

def test_processing_iostream_line_handler():
    with open(test_reader_path, 'r') as trp:
        _ = trp.readline()
        line = trp.readline()
    sink, day_tracker = staged._input_stream_line_handler(
        line,
        {
            f"{staged.station_col}": [],
            f"{staged.timestamp_col}": [],
            f"{staged.id_col}": [],
            f"{staged._target_col}": [],
        },
        Counter(),
    )
    assert day_tracker["63rdStreetWeatherStation20161231"] == 1
    assert sink["Station Name"] == ["63rd Street Weather Station",]
    assert sink["Air Temperature"] == ["-1.3",]

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
    tdf_time_lazy = staged._process_time_columns(tdf)
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
    expected_colnames = ["Station Name", "Date", "time_12h", "time_24h", staged._target_col]

    staged.set_parser_pipe(staged._process_time_columns)  # 
    staged.set_query_pipe(staged.__bypass__)
    _ = staged._process_input_file()

    assert len(staged._processed_dataframes.columns) == 5
    assert all([colname in expected_colnames for colname in staged._processed_dataframes.columns])
    assert (
        staged._processed_dataframes.lazy()
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
    staged.set_parser_pipe(staged._process_time_columns)
    staged.set_query_pipe(staged._query_min_max_first_last)
    _ = staged._process_input_file()
    assert len(staged._processed_dataframes.columns) == 6
    tdf_a = (
        staged._processed_dataframes
        .filter(pl.col("Date").str.contains("^12/30/2016$"))
        .filter(pl.col("Station Name").str.contains("^Foster"))
    )
    tdf_b = (
        staged._processed_dataframes
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
