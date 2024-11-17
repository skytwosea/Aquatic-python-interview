import io
import sys
import polars as pl
from pathlib import Path
from collections import Counter


class InputError(Exception):
    pass


def process_csv(
    reader: io.TextIOWrapper = sys.stdin,
    writer: io.TextIOWrapper = sys.stdout,
    target_col: str | None = None,  # default value set in-class to "Air Temperature",
):
    """Entry point; initializes Weather class.

    Default io is STDIN/STDOUT. Does not stream output.
    """
    staged = Weather(reader, writer, target_col)
    # staged.read()
    staged.batch_process_input_stream(lines_per_batch=10000)
    staged.write()
    staged.deinit()


class Weather:
    # input column labels:
    station_col = "Station Name"
    timestamp_col = "Measurement Timestamp"
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
        reader: io.TextIOWrapper,
        writer: io.TextIOWrapper,
        provided_target: str = None,
    ):
        self._reader, self._writer = self._validate_data_source(reader, writer)
        self._target_col: str = self._set_target_col(provided_target)
        self._label_cols: list = self._set_labels()
        self._header_indexes_map: dict = self._set_header_index_map()
        self._parser_pipe: callable = self._set_parser_pipe()
        self._query_pipe: callable = self._set_query_pipe()
        self._processed_dataframes: list = []

    def _validate_data_source(self, reader, writer) -> Path | None:
        if not isinstance(reader, io.TextIOWrapper):
            raise InputError("Error: reader is not recognized: require io.TextIOWrapper")
        if not isinstance(writer, io.TextIOWrapper):
            raise InputError("Error: writer is not recognized: require io.TextIOWrapper")
        return (reader, writer)

    def _set_target_col(self, provided_target: str) -> str:
        return self.provided_target if provided_target else self.airtemp_col

    def _set_labels(self) -> list:
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

    def _set_header_index_map(self) -> dict:
        header_line = self._reader.readline()
        labels = header_line.strip().split(',')
        return {c:i for i,c in enumerate(labels)}

    def _set_parser_pipe(self, fn: callable=None) -> callable:
        if not fn:
            return self._process_time_columns
        return fn

    def _set_query_pipe(self, fn: callable=None) -> callable:
        if not fn:
            return self._query_min_max_first_last
        return fn


    def _bypass(self, lf: pl.LazyFrame) -> pl.LazyFrame:
        """Utility method, used to isolate processing logic for testability."""
        return lf

    @staticmethod
    def __breakpoint__() -> None:
        """Flush pipe and reconnect keyboard before dropping into pdb.

        https://stackoverflow.com/questions/9178751/use-pdb-set-trace-in-a-script-that-reads-stdin-via-a-pipe
        (it's an older code, sir, but it checks out)
        """
        _ = sys.stdin.readlines()
        sys.stdin=open("/dev/tty")
        breakpoint()

    def read(self) -> None:
        pass

    def batch_process_input_stream(
        self,
        lines_per_batch: int = 5000,
    ) -> pl.DataFrame:
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
            while batch_limit_ctr < lines_per_batch:
                if not (line := self._reader.readline()):
                    done = True
                    break
                sink, day_tracker = self._input_stream_line_handler(line, sink, day_tracker)
                batch_limit_ctr += 1
            batch_df, sink, day_tracker = self._input_stream_dataframe_handler(done, sink, day_tracker)
            self._processed_dataframes.append(
                batch_df
                .lazy()
                .pipe(self._parser_pipe)
                .pipe(self._query_pipe)
                .collect()
            )

    def _input_stream_line_handler(self, line: str, sink: dict, day_tracker: Counter) -> tuple:
        line = line.strip().split(',')
        for k,v in sink.items():
            v.append(line[self._header_indexes_map[k]])
            if k == self.id_col:
                day_tracker[line[self._header_indexes_map[self.id_col]][:-4]] += 1
        return (sink, day_tracker)

    def _input_stream_dataframe_handler(self, done: bool, sink: dict, day_tracker: Counter) -> tuple:
        completed_days = [key for key, count in day_tracker.items() if count == 24]
        # use completed_days to get subset of sink that is complete:
        if done:
            # don't filter at EOF. Some measurement day spans in the dataset have only 23
            # measurements; let them through at the end:
            batch_df = (
                pl.DataFrame(sink, schema_overrides={self._target_col:pl.Float64,}, strict=False)
            )
        else:
            # not at EOF: filter for days that are guaranteed to be complete (24 measurements
            # in a day, at a station) so that incomplete ones can continue accummulating if necessary:
            batch_df = (
                pl.DataFrame(sink, schema_overrides={self._target_col:pl.Float64,}, strict=False)
                .filter(pl.col(self.id_col).str.contains_any(completed_days))
            )
            sink = (
                pl.DataFrame(sink, infer_schema_length=0)
                .filter(~pl.col(self.id_col).str.contains_any(completed_days))
            ).to_dict(as_series=False)
            # trim the counter:
            day_tracker = Counter({key:count for key, count in day_tracker.items() if count < 24})
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
                )  # keep time_12h for testing
                .alias(self.timing_struct)  # temp name only; can be hardcoded
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

    def write(self):
        df = pl.concat(self._processed_dataframes).sort(self.date_col, self.station_col)
        df.write_csv(self._writer)

    def deinit(self):
        self._reader.close()
        self._writer.close()
