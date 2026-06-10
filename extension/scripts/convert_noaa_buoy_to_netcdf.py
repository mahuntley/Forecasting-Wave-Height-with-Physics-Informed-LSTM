"""Convert NOAA buoy text files to the NetCDF format used here."""

import argparse
from pathlib import Path

import pandas as pd
import xarray as xr


NOAA_COLUMNS = [
    "YY",
    "MM",
    "DD",
    "hh",
    "mm",
    "WDIR",
    "WSPD",
    "GST",
    "WVHT",
    "DPD",
    "APD",
    "MWD",
    "PRES",
    "ATMP",
    "WTMP",
    "DEWP",
    "VIS",
    "TIDE",
]

SENTINELS = {
    "WDIR": [999],
    "WSPD": [99.0],
    "GST": [99.0],
    "WVHT": [99.0],
    "DPD": [99.0],
    "APD": [99.0],
    "MWD": [999],
    "PRES": [9999.0],
    "ATMP": [999.0],
    "WTMP": [999.0],
    "DEWP": [999.0],
    "VIS": [99.0],
    "TIDE": [99.0],
}

MODEL_COLUMNS = ["swh", "pp1d", "dwi", "wind"]


def read_noaa_file(path):
    data = pd.read_csv(path, sep=r"\s+", comment="#", header=None, names=NOAA_COLUMNS)
    data["time_original"] = pd.to_datetime(
        {
            "year": data["YY"],
            "month": data["MM"],
            "day": data["DD"],
            "hour": data["hh"],
            "minute": data["mm"],
        },
        errors="coerce",
    )
    data["source_file"] = path.name
    return data


def replace_sentinels(data):
    cleaned = data.copy()
    for column, bad_values in SENTINELS.items():
        cleaned[column] = cleaned[column].replace(bad_values, pd.NA)
    return cleaned


def make_file_report(raw, cleaned, path):
    row = {
        "file": path.name,
        "raw_rows": len(raw),
        "bad_datetimes": int(raw["time_original"].isna().sum()),
        "first_time": raw["time_original"].min(),
        "last_time": raw["time_original"].max(),
    }

    for column, bad_values in SENTINELS.items():
        row[f"{column}_sentinel_count"] = int(raw[column].isin(bad_values).sum())
        row[f"{column}_valid_after_cleaning"] = int(cleaned[column].notna().sum())

    return row


def regularize_hourly(cleaned, max_interpolate_gap):
    needed = cleaned[
        [
            "time_original",
            "WVHT",
            "DPD",
            "WDIR",
            "WSPD",
        ]
    ].dropna(subset=["time_original"])

    hourly = pd.DataFrame(
        {
            "time": needed["time_original"].dt.round("h"),
            "swh": pd.to_numeric(needed["WVHT"], errors="coerce"),
            "pp1d": pd.to_numeric(needed["DPD"], errors="coerce"),
            "dwi": pd.to_numeric(needed["WDIR"], errors="coerce"),
            "wind": pd.to_numeric(needed["WSPD"], errors="coerce"),
        }
    )

    # Rounding can make duplicate hours, so average them.
    hourly = hourly.groupby("time", as_index=True).mean().sort_index()
    full_index = pd.date_range(hourly.index.min(), hourly.index.max(), freq="h")
    hourly = hourly.reindex(full_index)
    hourly.index.name = "time"

    missing_before = hourly[MODEL_COLUMNS].isna().sum().to_dict()
    if max_interpolate_gap > 0:
        hourly[MODEL_COLUMNS] = hourly[MODEL_COLUMNS].interpolate(
            method="time",
            limit=max_interpolate_gap,
            limit_area="inside",
        )
    missing_after = hourly[MODEL_COLUMNS].isna().sum().to_dict()

    return hourly, missing_before, missing_after


def write_netcdf(hourly, output_path, lat, lon, station):
    dataset = xr.Dataset(
        data_vars={
            "swh": (("time", "latitude", "longitude"), hourly["swh"].to_numpy().reshape(-1, 1, 1)),
            "pp1d": (("time", "latitude", "longitude"), hourly["pp1d"].to_numpy().reshape(-1, 1, 1)),
            "dwi": (("time", "latitude", "longitude"), hourly["dwi"].to_numpy().reshape(-1, 1, 1)),
            "wind": (("time", "latitude", "longitude"), hourly["wind"].to_numpy().reshape(-1, 1, 1)),
        },
        coords={
            "time": hourly.index,
            "latitude": [lat],
            "longitude": [lon],
        },
        attrs={
            "source": "NOAA buoy text files",
            "station": station,
            "note": "dwi is wind direction from NOAA WDIR; wind is wind speed from NOAA WSPD.",
        },
    )

    dataset["swh"].attrs.update({"long_name": "significant wave height", "units": "m"})
    dataset["pp1d"].attrs.update({"long_name": "dominant wave period", "units": "s"})
    dataset["dwi"].attrs.update({"long_name": "wind direction", "units": "degrees true"})
    dataset["wind"].attrs.update({"long_name": "wind speed", "units": "m s-1"})

    output_path.parent.mkdir(parents=True, exist_ok=True)
    dataset.to_netcdf(output_path)


def convert(args):
    input_dir = Path(args.input_dir)
    files = sorted(input_dir.glob(args.pattern))
    if args.start_year is not None:
        files = [path for path in files if int(path.stem) >= args.start_year]
    if args.end_year is not None:
        files = [path for path in files if int(path.stem) <= args.end_year]
    if not files:
        raise FileNotFoundError(f"No files matched {input_dir / args.pattern}")

    report_rows = []
    cleaned_parts = []

    for path in files:
        raw = read_noaa_file(path)
        cleaned = replace_sentinels(raw)
        report_rows.append(make_file_report(raw, cleaned, path))
        cleaned_parts.append(cleaned)

    cleaned_all = pd.concat(cleaned_parts, ignore_index=True)
    hourly, missing_before, missing_after = regularize_hourly(cleaned_all, args.max_interpolate_gap)

    final_report = {
        "file": "combined",
        "raw_rows": len(cleaned_all),
        "bad_datetimes": int(cleaned_all["time_original"].isna().sum()),
        "first_time": cleaned_all["time_original"].min(),
        "last_time": cleaned_all["time_original"].max(),
        "hourly_rows": len(hourly),
        "complete_model_rows": int(hourly[MODEL_COLUMNS].dropna().shape[0]),
    }
    for column in MODEL_COLUMNS:
        final_report[f"{column}_missing_before_interpolation"] = int(missing_before[column])
        final_report[f"{column}_missing_after_interpolation"] = int(missing_after[column])

    report = pd.DataFrame(report_rows + [final_report])
    report_path = Path(args.report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report.to_csv(report_path, index=False)

    write_netcdf(hourly, Path(args.output), args.lat, args.lon, args.station)

    print(report.to_string(index=False))
    print(f"Wrote NetCDF: {args.output}")
    print(f"Wrote report: {args.report}")


def parse_args():
    parser = argparse.ArgumentParser(description="Convert NOAA buoy text files to LSTM NetCDF format.")
    parser.add_argument("input_dir", help="Folder containing NOAA text/php files.")
    parser.add_argument("output", help="Output NetCDF path.")
    parser.add_argument("--pattern", default="*.php")
    parser.add_argument("--start-year", type=int)
    parser.add_argument("--end-year", type=int)
    parser.add_argument("--report", default="extension/results/noaa_buoy_cleaning_report.csv")
    parser.add_argument("--lat", type=float, default=32.867)
    parser.add_argument("--lon", type=float, default=-117.257)
    parser.add_argument("--station", default="LJPC1 - La Jolla, CA")
    parser.add_argument(
        "--max-interpolate-gap",
        type=int,
        default=6,
        help="Maximum consecutive hourly gaps to fill. Use 0 to disable interpolation.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    convert(parse_args())
