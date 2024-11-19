import io
import sys
import polars as pl
from pathlib import Path
from collections import Counter


class InputError(Exception):
    pass


def process_csv(
    reader: io.TextIOWrapper | io.StringIO | Path | str = sys.stdin,
    writer: io.TextIOWrapper | io.StringIO | Path | str = sys.stdout,
    *,
    target: str | None = None,  # default value set in-class to "Air Temperature",
):
    staged = Weather(reader, writer, target)
    staged.run()


class Weather:
    # input column labels:
    station_col = "Station Name"
    timestamp_col = "Measurement Timestamp"
    timestamp_label_col = "Measurement Timestamp Label"
    id_col = "Measurement ID"
    airtemp_col = "Air Temperature"

    # output column labels:
    date_col = "Date"

    # processing column labels:
    time12_col = "time_12h"
    time24_col = "time_24h"
    timing_struct = "timing_struct"

    def __init__(
        self,
        reader: io.TextIOWrapper | io.StringIO | Path | str,
        writer: io.TextIOWrapper | io.StringIO | Path | str,
        target: str|None = None,
    ):
        self._reader, self._writer = self._validate_io(reader, writer)
        self._header_indexes_map: dict = self._set_header_index_map()
        self._target_col: str = self.set_target_col(target)
        self._label_cols: list = self.set_labels()
        self._schema_overrides_map: dict = self.set_schema_overrides()
        self._parser_pipe: callable = self._process_time_columns
        self._query_pipe: callable = self._query_min_max_first_last
        self._batch_line_limit: int = 10000
        self._processed_dataframes: list | str = []

    def _validate_io(self, reader, writer) -> tuple:
        if not reader or not writer:
            raise InputError(f"Error: missing io path(s).")
        io_dict = {"reader": reader, "writer": writer}
        for label, obj in io_dict.items():
            if not any([isinstance(obj, T) for T in [io.TextIOWrapper, io.StringIO, Path, str]]):
                raise InputError(
                    f"Error: {label} type <{type(obj)}> is not permitted: must be of type [io.TextIOWrapper, io.StringIO, Path, str]"
                )
            if not any([isinstance(obj, T) for T in [io.TextIOWrapper, io.StringIO]]):
                obj = Path(obj).resolve()
                if not obj.exists():
                    raise InputError(f"Error: {label} path does not exist.")
                if not obj.is_file():
                    raise InputError(f"Error: {label} path is not a file.")
                if not obj.suffix == ".csv":
                    raise InputError(f"Error: {label} suffix is not a csv.")
                io_dict[label] = obj
        return (io_dict["reader"], io_dict["writer"])

    def _set_header_index_map(self) -> dict:
        if any([isinstance(self._reader, T) for T in [io.TextIOWrapper, io.StringIO]]):
            header_line = self._reader.readline()
        elif any([isinstance(self._reader, T) for T in [str, Path]]):
            with open(self._reader, "r") as f:
                header_line = f.readline()
        labels = header_line.strip().split(",")
        return {c: i for i, c in enumerate(labels)}

    def set_target_col(self, target: str|None) -> str:
        """Set the column on which to compute. Defaults to 'Air Temperature'.

        param target must be a header label in the target document.
        This parameter is case sensitive.
        """
        if not target:
            target = self.airtemp_col
        if target not in list(self._header_indexes_map.keys()):
            raise InputError("Error: target column not in file header.")
        return target

    def set_labels(self, label: str = None) -> list:
        """Set the output column label tag.

        param label can be any valid string.
        The label will be prefixed with {Min, Max, First, Last}.
        """
        if not label:
            match self._target_col:
                case self.airtemp_col:
                    label = "Temp"
                case _:
                    label = self._target_col
        return [
            f"{prefix} {label}"
            for prefix in [
                "Min",
                "Max",
                "First",
                "Last",
            ]
        ]

    def set_schema_overrides(self, overrides: dict | None = None) -> dict:
        """Set Polars schema overrides.

        param overrides must be a dict as so: {column_label:pl.<T>}
        where column_label is a string and pl.<T> is a valid Polars data type.
        Valid only when reading from file.
        """
        if overrides:
            return overrides
        header_lst = list(self._header_indexes_map.keys())
        string_subset = [
            self.station_col,
            self.timestamp_col,
            self.id_col,
            self.timestamp_label_col,
        ]
        return {
            col: (pl.String if col in string_subset else pl.Float64)
            for col in header_lst
        }

    def set_parser_pipe(self, fn: callable = None) -> None:
        """Set the pre-query processing function.

        Function must return a pl.LazyFrame that adheres to the constraints of
        the input dataframe, and to the constraints of the query function.
        """
        self._parser_pipe = fn

    def set_query_pipe(self, fn: callable = None) -> None:
        """Set the query processing function.

        Function must return a pl.LazyFrame that adheres to the constraints of
        the input dataframe.
        """
        self._query_pipe = fn

    def set_batch_size(self, size) -> None:
        """Set the line limit for batches in streaming mode. Defaults to 10000"""
        self._batch_line_limit = size

    @staticmethod
    def __bypass__(lf: pl.LazyFrame) -> pl.LazyFrame:
        """Utility method, used to isolate processing logic for testing."""
        return lf

    def __breakpoint__(self) -> None:
        """Flush pipe and reconnect tty before dropping into pdb.

        https://stackoverflow.com/questions/9178751/use-pdb-set-trace-in-a-script-that-reads-stdin-via-a-pipe
        (it's an older code, sir, but it checks out)
        """
        if any([isinstance(self._reader, T) for T in [io.TextIOWrapper, io.StringIO]]):
            _ = self._reader.readlines()
            sys.stdin = open("/dev/tty")
            breakpoint()
        else:
            sys.stdin = open("/dev/tty")
            breakpoint()

    def run(self):
        self._read()
        self._write()
        self._deinit()

    def _read(self) -> None:
        if any([isinstance(self._reader, T) for T in [io.TextIOWrapper, io.StringIO]]):
            self._batch_process_input_stream()
        else:
            self._process_input_file()

    def _write(self) -> None:
        if isinstance(self._processed_dataframes, list):
            df = pl.concat(self._processed_dataframes).sort(
                self.date_col, self.station_col
            )
        elif isinstance(self._processed_dataframes, pl.DataFrame):
            df = self._processed_dataframes.sort(
                self.date_col, self.station_col
            )
        df.write_csv(file=self._writer)

    def _deinit(self) -> None:
        # don't close when reading from file; Polars uses an
        # internal context manager
        if any([isinstance(self._reader, T) for T in [io.TextIOWrapper, io.StringIO]]):
            self._reader.close()
        if any([isinstance(self._writer, T) for T in [io.TextIOWrapper, io.StringIO]]):
            self._writer.close()

    def _process_input_file(self) -> None:
        """Extract target columns from file, and process through Polars.

        From Polars documentation:
        > Lazily read from a CSV file or multiple files via glob patterns.
          This allows the query optimizer to push down predicates and projections to
          the scan level, thereby potentially reducing memory overhead.
        https://docs.pola.rs/api/python/dev/reference/api/polars.scan_csv.html
        """
        self._processed_dataframes = (
            pl.scan_csv(self._reader, schema_overrides=self._schema_overrides_map)
            .select(
                [
                    self.station_col,
                    self.timestamp_col,
                    self.id_col,
                    self._target_col,
                ]
            )
            .pipe(self._parser_pipe)
            .pipe(self._query_pipe)
        ).collect()

    def _batch_process_input_stream(self) -> None:
        """Extract target columns from io stream, and process through Polars.

        This function reads lines in serial, and extracts target columns into sink.
        When line limit is reached, the sink is filtered for all components that
        are guaranteed to be complete, ie. 24 measurements in a day for a station.
        These components are processed and stored until write at EOF. The sink is
        then filtered for those components that are not complete prior to next iteration.
        """
        day_tracker = Counter()
        sink = {
            f"{self.station_col}": [],
            f"{self.timestamp_col}": [],
            f"{self.id_col}": [],
            f"{self._target_col}": [],
        }
        done = False
        while not done:
            batch_limit_ctr = 0
            while batch_limit_ctr < self._batch_line_limit:
                if not (line := self._reader.readline()):
                    done = True
                    break
                sink, day_tracker = self._input_stream_line_handler(
                    line, sink, day_tracker
                )
                batch_limit_ctr += 1
            batch_df, sink, day_tracker = self._input_stream_dataframe_handler(
                done, sink, day_tracker
            )
            self._processed_dataframes.append(
                batch_df.lazy().pipe(self._parser_pipe).pipe(self._query_pipe).collect()
            )

    def _input_stream_line_handler(
        self, line: str, sink: dict, day_tracker: Counter
    ) -> tuple:
        line = line.strip().split(",")
        for k, v in sink.items():
            v.append(line[self._header_indexes_map[k]])
            if k == self.id_col:
                day_tracker[line[self._header_indexes_map[self.id_col]][:-4]] += 1
        return (sink, day_tracker)

    def _input_stream_dataframe_handler(
        self, done: bool, sink: dict, day_tracker: Counter
    ) -> tuple:
        completed_days = [key for key, count in day_tracker.items() if count == 24]
        # use completed_days to get subset of sink that is complete:
        if done:
            # don't filter at EOF. Some measurement day spans in the dataset have only 23
            # measurements; let them through at the end:
            batch_df = pl.DataFrame(
                sink,
                schema_overrides={
                    self._target_col: pl.Float64,
                },
                strict=False,
            )
        else:
            # not at EOF: filter for days that are guaranteed to be complete (24 measurements
            # in a day, at a station) so that incomplete ones can continue accummulating if necessary:
            batch_df = pl.DataFrame(
                sink,
                schema_overrides={
                    self._target_col: pl.Float64,
                },
                strict=False,
            ).filter(pl.col(self.id_col).str.contains_any(completed_days))
            sink = (
                pl.DataFrame(sink, infer_schema_length=0).filter(
                    ~pl.col(self.id_col).str.contains_any(completed_days)
                )
            ).to_dict(as_series=False)
            # trim the counter:
            day_tracker = Counter(
                {key: count for key, count in day_tracker.items() if count < 24}
            )
        return (batch_df, sink, day_tracker)

    def _process_time_columns(self, df: pl.DataFrame) -> pl.LazyFrame:
        """Pre-query processing method for time columns.

        _query_min_max_first_last() depends on identifying the first and last hour
        of each day. This function extracts and recasts the last four digits of the
        <id_col> column, which carry 24hr time.
        """
        return (
            df.lazy()
            .with_columns(
                pl.col(self.timestamp_col)
                .str.splitn(by=" ", n=2)
                .struct.rename_fields(
                    [self.date_col, self.time12_col]
                )  # keep time_12h for testing. Is dropped downstream.
                .alias(self.timing_struct)
            )
            .unnest(self.timing_struct)
            .with_columns(
                pl.col(self.id_col).str.tail(n=4).cast(pl.Int64).alias(self.time24_col)
            )
            .drop(self.timestamp_col, self.id_col)
        )

    def _query_min_max_first_last(self, df: pl.DataFrame) -> pl.LazyFrame:
        """Query method for data columns.

        Polars' group_by:agg API accepts expressions; see more here:
        https://labs.quansight.org/blog/dataframe-group-by
        """
        return (
            df.lazy()
            .group_by(self.station_col, self.date_col)
            .agg(
                pl.min(self._target_col).alias(self._label_cols[0]),
                pl.max(self._target_col).alias(self._label_cols[1]),
                (
                    pl.col(self._target_col)
                    .filter(pl.col(self.time24_col) == pl.min(self.time24_col))
                    .first()
                    .alias(self._label_cols[2])
                ),
                (
                    pl.col(self._target_col)
                    .filter(pl.col(self.time24_col) == pl.max(self.time24_col))
                    .first()
                    .alias(self._label_cols[3])
                ),
            )
        )
