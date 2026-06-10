"""Download ERA5 wave and wind data in monthly chunks."""

import argparse
from pathlib import Path

import cdsapi
import xarray as xr


DATASET = "reanalysis-era5-single-levels"

WAVE_VARIABLES = [
    "significant_height_of_combined_wind_waves_and_swell",
    "peak_wave_period",
]

WIND_VARIABLES = [
    "10m_u_component_of_wind",
    "10m_v_component_of_wind",
]

MONTHS = [f"{month:02d}" for month in range(1, 13)]
DAYS = [f"{day:02d}" for day in range(1, 32)]
HOURS = [f"{hour:02d}:00" for hour in range(24)]


def parse_area(text):
    """Parse north,west,south,east."""
    values = [float(part.strip()) for part in text.split(",")]
    if len(values) != 4:
        raise argparse.ArgumentTypeError("Use four values: north,west,south,east")
    return values


def make_request(year, month, variables, area):
    return {
        "product_type": ["reanalysis"],
        "variable": variables,
        "year": [str(year)],
        "month": [month],
        "day": DAYS,
        "time": HOURS,
        "data_format": "netcdf",
        "download_format": "unarchived",
        "area": area,
    }


def download_months(client, years, variables, label, area, chunks_dir, overwrite):
    paths = []

    for year in years:
        for month in MONTHS:
            path = chunks_dir / f"era5_{label}_{year}_{month}.nc"
            paths.append(path)

            # This lets us resume after an interrupted download.
            if path.exists() and not overwrite:
                print(f"Skipping existing file: {path}")
                continue

            print(f"Downloading {label} {year}-{month}")
            request = make_request(year, month, variables, area)
            client.retrieve(DATASET, request, str(path))

    return paths


def merge_files(wave_paths, wind_paths, output_path):
    print("Opening wave chunks")
    waves = xr.open_mfdataset([str(path) for path in wave_paths], combine="by_coords")

    print("Opening wind chunks")
    wind = xr.open_mfdataset([str(path) for path in wind_paths], combine="by_coords")

    print(f"Merging into {output_path}")
    merged = xr.merge([waves, wind], compat="override")
    merged.to_netcdf(output_path)
    print(f"Wrote {output_path}")


def parse_args():
    parser = argparse.ArgumentParser(description="Download ERA5 wind and wave data.")
    parser.add_argument("--start-year", type=int, default=2013)
    parser.add_argument("--end-year", type=int, default=2019)
    parser.add_argument(
        "--area",
        type=parse_area,
        default="-25,-52,-34,-46",
        help="Brazil coast box as north,west,south,east.",
    )
    parser.add_argument("--chunks-dir", type=Path, default=Path("era5_chunks"))
    parser.add_argument("--output", type=Path, default=Path("era5_full_2013_2019.nc"))
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--skip-merge", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    args.chunks_dir.mkdir(parents=True, exist_ok=True)

    client = cdsapi.Client()
    years = range(args.start_year, args.end_year + 1)

    wind_paths = download_months(
        client, years, WIND_VARIABLES, "wind", args.area, args.chunks_dir, args.overwrite
    )
    wave_paths = download_months(
        client, years, WAVE_VARIABLES, "waves", args.area, args.chunks_dir, args.overwrite
    )

    if not args.skip_merge:
        merge_files(wave_paths, wind_paths, args.output)


if __name__ == "__main__":
    main()
