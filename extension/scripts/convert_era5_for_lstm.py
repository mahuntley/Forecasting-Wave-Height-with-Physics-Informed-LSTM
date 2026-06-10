"""Convert downloaded ERA5 data to the LSTM input format."""

import argparse
from pathlib import Path

import numpy as np
import xarray as xr


def convert_file(input_path, output_path):
    dataset = xr.open_dataset(input_path)

    # CDS sometimes uses valid_time, while these scripts expect time.
    if "valid_time" in dataset.coords and "time" not in dataset.coords:
        dataset = dataset.rename({"valid_time": "time"})

    needed = {"swh", "pp1d", "u10", "v10"}
    missing = needed - set(dataset.data_vars)
    if missing:
        raise ValueError(f"Missing variables in {input_path}: {sorted(missing)}")

    # Wind speed from the two wind components.
    dataset["wind"] = np.hypot(dataset["u10"], dataset["v10"])
    dataset["wind"].attrs["long_name"] = "10 metre wind speed from u10 and v10"

    # Direction in degrees for compatibility with the original scripts.
    dataset["dwi"] = (180.0 + np.degrees(np.arctan2(dataset["u10"], dataset["v10"]))) % 360.0
    dataset["dwi"].attrs["long_name"] = "10 metre wind direction from u10 and v10"
    dataset["dwi"].attrs["units"] = "degrees"

    output = dataset[["swh", "pp1d", "dwi", "wind"]]
    output.to_netcdf(output_path)
    print(f"Wrote {output_path}")


def parse_args():
    parser = argparse.ArgumentParser(description="Convert ERA5 data for the LSTM scripts.")
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    return parser.parse_args()


def main():
    args = parse_args()
    convert_file(args.input, args.output)


if __name__ == "__main__":
    main()
