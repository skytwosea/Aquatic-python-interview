import polars as pl
from pathlib import Path
import typer
from typing import Annotated, Optional
import sys

app = typer.Typer(pretty_exceptions_show_locals=False)


class InputError(Exception):
    pass


class ChicagoWeather:
    station_col = "Station Name"
    timestamp_col = "Measurement Timestamp"
    id_col = "Measurement ID"
    date_col = "Date"
    airtemp_col = "Air Temperature"
    schema_overrides_map = {
        airtemp_col: pl.Float64,
        "Wet Bulb Temperature": pl.Float64,
        "Humidity": pl.Float64,
        "Rain Intensity": pl.Float64,
        "Interval Rain": pl.Float64,
        "Total Rain": pl.Float64,
        "Precipitation Type": pl.Float64,
        "Wind Direction": pl.Float64,
        "Wind Speed": pl.Float64,
        "Maximum Wind Speed": pl.Float64,
        "Barometric Pressure": pl.Float64,
        "Solar Radiation": pl.Float64,
        "Heading": pl.Float64,
        "Battery Life": pl.Float64,
    }

    def __init__(
        self,
        rawpath: str | None,
        target_col: str = None,
    ):
        self.path = self._validate_path(rawpath)
        self.processing = None  # repository for processing stages
        self.target_col: str = self.airtemp_col if target_col is None else target_col
        self.temp_cols: list = self._set_labels()

    def ingest_and_process(
        self,
        parser_pipe: callable,
        query_pipe: callable,
    ) -> pl.DataFrame:
        """Scan data from source csv file, and pipe through processing functions.

        Polars.scan_csv() is memory-friendly: it reads only those columns
        specified in .select(). If a filter is included, this is also pushed
        down to the scan, further reducing memory overhead. See the docs:
        https://docs.pola.rs/api/python/stable/reference/api/polars.scan_csv.html#polars-scan-csv

        Pipes are used to segregate logic for testing, as well as to increase modularity.
        """
        return (
            pl.scan_csv(str(self.path), schema_overrides=self.schema_overrides_map)
            .select(
                [
                    self.station_col,
                    self.timestamp_col,
                    self.id_col,
                    self.target_col,
                ]
            )
            .pipe(parser_pipe)
            .pipe(query_pipe)
        ).collect()

    def _validate_path(self, rawpath) -> Path:
        if not rawpath:
            raise InputError("Error: no path provided.")
        candidate = Path(rawpath).resolve()
        if not candidate.exists():
            raise InputError("Error: input path does not exist.")
        if not candidate.is_file():
            raise InputError("Error: input path is not a file.")
        if not candidate.suffix == ".csv":
            raise InputError(f"Error: input suffix is not csv: {candidate.name}")
        return candidate

    def _set_labels(self) -> list:
        match self.target_col:
            case self.airtemp_col:
                label = "Temp"
            case _:
                label = self.target_col
        return [
            f"{prefix} {label}"
            for prefix in [
                "Min",
                "Max",
                "First",
                "Last",
            ]
        ]

    def _bypass(self, df: pl.DataFrame) -> pl.DataFrame:
        """Utility method.

        Used to isolate processing logic for testability.
        Should not impact performance; the query optimizer
        should see that it's chasing pointers in a circle.
        """
        return df

    def _process_time_columns(
        self, df: pl.DataFrame, target_col: str | None = None
    ) -> pl.lazyframe.frame.LazyFrame:
        """Preprocessing method for time columns.

        _query_min_max_first_last() depends on identifying the first and last hour
        of each day. This function extracts and recasts the last four digits of the
        <id_col> column, which carry 24hr time.
        """
        if not target_col:
            target_col = self.timestamp_col
        return (
            df.lazy()
            .with_columns(
                pl.col(target_col)
                .str.splitn(by=" ", n=2)
                .struct.rename_fields(
                    [self.date_col, "time_12h"]
                )  # keep time_12h for testing
                .alias("timing")
            )
            .unnest("timing")
            .with_columns(
                pl.col(self.id_col).str.tail(n=4).cast(pl.Int64).alias("time_24h")
            )
            .drop(target_col, self.id_col)
        )

    def _query_min_max_first_last(
        self, df: pl.DataFrame, target_col: str | None = None
    ) -> pl.lazyframe.frame.LazyFrame:
        """Query method for data columns."""
        if not target_col:
            target_col = self.target_col
        return (
            df.lazy()
            .group_by(self.station_col, self.date_col)
            .agg(
                pl.min(target_col).alias(self.temp_cols[0]),
                pl.max(target_col).alias(self.temp_cols[1]),
                (
                    pl.col(target_col)
                    .filter(pl.col("time_24h") == pl.min("time_24h"))
                    .first()
                    .alias(self.temp_cols[2])
                ),
                (
                    pl.col(target_col)
                    .filter(pl.col("time_24h") == pl.max("time_24h"))
                    .first()
                    .alias(self.temp_cols[3])
                ),
            )
            .sort("Date", self.station_col)
        )


@app.command()
def main(
    file: Annotated[
        str,
        typer.Argument(
            help="path to target pdf file.",
        ),
    ] = None,
    col: Annotated[
        Optional[str], typer.Option("--column", help="name of column to process.")
    ] = "Air Temperature",
    pretty: Annotated[
        Optional[bool], typer.Option("--pretty", help="print a Polars table.")
    ] = False,
):
    staged = ChicagoWeather(file, col)
    df = staged.ingest_and_process(
        parser_pipe=staged._process_time_columns,
        query_pipe=staged._query_min_max_first_last,
    )
    if pretty:
        n = 10
        with pl.Config(
            set_fmt_str_lengths=35,
            set_tbl_rows=n,
            set_tbl_hide_dataframe_shape=True,
            set_tbl_hide_dtype_separator=True,
        ):
            print(df)
            print(f"{df.shape[0]-n} more rows")
    else:
        df.write_csv(file=sys.stdout)


if __name__ == "__main__":
    app()
