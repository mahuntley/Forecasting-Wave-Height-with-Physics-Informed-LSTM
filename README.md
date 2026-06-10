# LSTM Ocean Wave Forecasting Extension

This repo is based on the LSTM wave forecasting code from:

Minuzzi, F. and Farina, L. "A deep learning approach to predict significant
wave height using long short-term memory", Ocean Modelling 181, 2023.

Original paper links:

- <https://arxiv.org/abs/2201.00356>
- <https://doi.org/10.1016/j.ocemod.2022.102151>
- Original code repository: <https://github.com/felipeminuzzi/lstm-ocean>

We kept the original authors' code in `src/`. The code we added is in
`extension/`. Our main goal was to reproduce the basic LSTM setup and then test
a small physics-based regularizer for significant wave height forecasts.

## Project Summary

We forecast significant wave height, written as `Hs`, at 6, 12, 18, and 24 hour
lead times.

We compare four models:

| Model | Meaning |
|---|---|
| Persistence | Simple baseline: future `Hs` equals the latest observed `Hs` |
| Hs-only LSTM | LSTM using only recent wave height history |
| Multivariate wind LSTM | LSTM using `Hs`, peak period, and wind vector components |
| Physics-regularized LSTM | Same multivariate LSTM, but with extra physics-based penalty terms in the loss |

The new ML part is the physics-regularized LSTM. It is still a normal LSTM, but
the loss adds a few penalties for predictions that fail simple wave checks.

## Repository Structure

| Path | Purpose |
|---|---|
| `src/` | Original scripts from the authors' repository |
| `extension/` | Code and notes added for this project |
| `extension/scripts/` | New scripts for data conversion and experiments |
| `extension/final_results/` | Small tracked CSV summaries from the final experiments |
| `extension/results/` | Generated outputs from local runs; ignored by git because it can become large |
| `noaa/` | Optional raw NOAA buoy files; ignored by git |

Large NetCDF files are not tracked in git. Put them in the repo root before
running the experiments.

## What Came From The Original Repo

These files came from the original authors' repo:

| File | Role |
|---|---|
| `src/lstm_v1.py` | Basic LSTM using significant wave height history |
| `src/lstm_multi.py` | Multivariate LSTM script from the original repo |
| `src/lstm_v2.py` | LSTM version using ERA5 inputs with buoy target data |
| `src/lstm_historic.py` | Historical LSTM experiment script |
| `src/lstm_2D.py` | Grid-based LSTM prediction script |
| `src/read_netcdf.py` | Helper script for reading NetCDF data |

We kept these separate from the files we wrote for the extension.

## What We Added

| File or folder | What it does |
|---|---|
| `extension/scripts/evaluate_shared_experiments.py` | Main script used for the final experiments |
| `extension/scripts/convert_era5_for_lstm.py` | Converts downloaded ERA5 variables into the format expected by the LSTM scripts |
| `extension/scripts/download_era5_data.py` | Downloads ERA5 wave and wind data in smaller chunks |
| `extension/scripts/convert_noaa_buoy_to_netcdf.py` | Cleans NOAA buoy files and converts them to hourly NetCDF |
| `extension/final_results/` | Final summary CSV files from Brazil ERA5 and NOAA buoy runs |
| `extension/README.md` | Short guide focused only on the extension folder |

The main script is:

```bash
extension/scripts/evaluate_shared_experiments.py
```

## Data Used

### Brazil ERA5 Dataset

The original repo did not include the full input data, so we downloaded ERA5
single-level reanalysis data for part of the Brazilian coast.

Bounding box used for the ERA5 request:

```text
North: -25
West:  -52
South: -34
East:  -46
```

In CDS API format:

```python
"area": [-25, -52, -34, -46]
```

Years used:

```text
2013-2019
```

ERA5 variables:

| ERA5 variable | Name used in our converted file | Meaning |
|---|---|---|
| `swh` | `swh` | Significant wave height |
| `pp1d` | `pp1d` | Peak wave period |
| `u10` | used to make `wind` and `dwi` | 10 m east-west wind component |
| `v10` | used to make `wind` and `dwi` | 10 m north-south wind component |

The converted file expected by the final experiment is:

```text
era5_full_2013_2019_lstm_v1.nc
```

Main point used in the Brazil experiment:

```text
latitude  = -31.5
longitude = -50.0
```

### La Jolla / Torrey Pines NOAA Buoy Dataset

We also tested the same scripts on NOAA station LJPC1, listed by NDBC as
La Jolla, CA. This gave us a second dataset outside the Brazil ERA5 point. In
our earlier notes and filenames we refer to this as the Torrey Pines NOAA run.

The yearly NOAA files can be downloaded from the station history page:
<https://www.ndbc.noaa.gov/station_history.php?station=ljpc1>

Years used:

```text
2017-2023
```

NOAA columns mapped into our NetCDF file:

| NOAA column | Converted variable | Meaning |
|---|---|---|
| `WVHT` | `swh` | Significant wave height |
| `DPD` | `pp1d` | Dominant wave period, used as the period input |
| `WSPD` | `wind` | Wind speed |
| `WDIR` | `dwi` | Wind direction |

The converted file expected by the final NOAA experiment is:

```text
noaa_torrey_pines_2017_2023_lstm.nc
```

Point metadata used:

```text
latitude  = 32.867
longitude = -117.257
```

## Environment Setup

Create a virtual environment and install dependencies:

```bash
python3 -m venv .venv-tf
source .venv-tf/bin/activate
pip install -r requirements.txt
```

The main packages are TensorFlow, NumPy, pandas, xarray, netCDF4, h5netcdf,
scikit-learn, matplotlib, and cdsapi.

## Preparing Data

### Option 1: Use An Existing Converted ERA5 File

Place this file in the repo root:

```text
era5_full_2013_2019_lstm_v1.nc
```

Then the final Brazil experiment can be run directly.

### Option 2: Download ERA5 Again

First configure the Copernicus CDS API in `~/.cdsapirc`. The file should look
like this:

```text
url: https://cds.climate.copernicus.eu/api
key: YOUR_CDS_API_KEY
```

Then run:

```bash
source .venv-tf/bin/activate

python -u extension/scripts/download_era5_data.py \
  --start-year 2013 \
  --end-year 2019 \
  --area -25 -52 -34 -46 \
  --output-dir era5_chunks \
  --merged-output era5_full_2013_2019.nc

python -u extension/scripts/convert_era5_for_lstm.py \
  era5_full_2013_2019.nc \
  era5_full_2013_2019_lstm_v1.nc
```

The download script uses monthly wave and wind requests because one large ERA5
request can exceed the CDS API size limit.

### Option 3: Convert NOAA Buoy Files

Download the yearly NOAA files from:
<https://www.ndbc.noaa.gov/station_history.php?station=ljpc1>

Place raw NOAA yearly files in:

```text
noaa/
```

For example:

```text
noaa/2017.php
noaa/2018.php
...
noaa/2023.php
```

Then run:

```bash
source .venv-tf/bin/activate

python -u extension/scripts/convert_noaa_buoy_to_netcdf.py \
  noaa \
  noaa_torrey_pines_2017_2023_lstm.nc \
  --start-year 2017 \
  --end-year 2023 \
  --report extension/results/noaa_torrey_pines_2017_2023_cleaning_report.csv \
  --lat 32.867 \
  --lon -117.257 \
  --station "LJPC1 - La Jolla, CA"
```

This script replaces NOAA missing-value codes, rounds observations to hourly
timestamps, averages duplicate hours, fills short gaps, and writes a NetCDF
file for the LSTM scripts.

## Running The Final Experiments

### Quick Smoke Test

The full runs below take longer. To check that the code and environment work,
run a one-lead, one-epoch smoke test after placing a converted NetCDF file in
the repo root:

```bash
source .venv-tf/bin/activate

python -u extension/scripts/evaluate_shared_experiments.py \
  era5_full_2013_2019_lstm_v1.nc \
  --lat -31.5 \
  --lon -50.0 \
  --lead-times 6 \
  --epochs 1 \
  --repeats 1 \
  --test-size 168 \
  --no-save-models \
  --output-dir extension/results/smoke_test
```

### Brazil ERA5 Final Run

```bash
source .venv-tf/bin/activate

python -u extension/scripts/evaluate_shared_experiments.py \
  era5_full_2013_2019_lstm_v1.nc \
  --lat -31.5 \
  --lon -50.0 \
  --epochs 10 \
  --patience 100 \
  --repeats 10 \
  --test-size 744 \
  --no-save-models \
  --output-dir extension/results/shared_eval_wind_brazil_epochs10_repeats10
```

### La Jolla / Torrey Pines NOAA Final Run

```bash
source .venv-tf/bin/activate

python -u extension/scripts/evaluate_shared_experiments.py \
  noaa_torrey_pines_2017_2023_lstm.nc \
  --lat 32.867 \
  --lon -117.257 \
  --epochs 10 \
  --patience 100 \
  --repeats 10 \
  --test-size 744 \
  --no-save-models \
  --output-dir extension/results/torrey_pines_noaa_wind_epochs10_repeats10
```

The `--patience 100` setting keeps the run close to fixed 10-epoch training. We
used fixed epochs so the final comparison was easier to read.

## Output Files

Each run writes:

| File | Meaning |
|---|---|
| `run_config.json` | Settings used for the experiment |
| `shared_eval_runs.csv` | Every individual repeat and model result |
| `shared_eval_summary.csv` | Mean and standard deviation across repeats |
| `shared_eval_best_pinn_by_lead.csv` | Best physics-regularizer weights for each lead time |
| `shared_eval_training_history.csv` | Training and validation loss by epoch |
| `shared_eval_epoch_summary.csv` | Number of epochs trained and best validation epoch |
| `shared_eval_persistence_predictions_*_leadtime.csv` | Persistence baseline predictions |

If `--no-save-models` is not used, the script also saves Keras model files,
scalers, and metadata.

## Physics-Regularized Loss

The physics-regularized model uses the same LSTM as the multivariate wind model,
but trains with:

```text
loss = MAE + lambda1 * R_steep + lambda2 * R_jump + lambda3 * R_bounds
```

The terms are:

| Term | Purpose |
|---|---|
| `MAE` | Main prediction error between predicted and true future `Hs` |
| `R_steep` | Penalizes wave heights that are too large for the peak period |
| `R_jump` | Penalizes sudden jumps from the latest observed wave height |
| `R_bounds` | Penalizes normalized predictions outside the training range |

For `R_steep`, we use the deep-water wavelength approximation:

```text
L = g * Tp^2 / (2 * pi)
```

where `g` is gravity and `Tp` is peak wave period. We then apply a soft penalty
when predicted `Hs` is above approximately:

```text
0.055 * L
```

The regularizer is soft, so the model can still violate the threshold. It just
pays an extra loss penalty when it does.

The lambda values are not learned by the model. We tested this small grid inside
`evaluate_shared_experiments.py`:

```text
(0.02, 0.00, 0.01)
(0.05, 0.01, 0.01)
(0.10, 0.02, 0.01)
(0.20, 0.05, 0.01)
```

## Final Results

MAPE is mean absolute percentage error. Lower is better.

### Brazil ERA5, 2013-2019

Final run:

```text
10 epochs, 10 repeats, test size 744 hours
```

| Lead time | Persistence | Hs-only LSTM | Multi-wind LSTM | Physics-reg LSTM |
|---:|---:|---:|---:|---:|
| 6h | 9.78 | 6.86 | 5.95 | 5.92 |
| 12h | 16.38 | 13.70 | 11.94 | 11.74 |
| 18h | 19.85 | 16.82 | 15.37 | 15.29 |
| 24h | 22.82 | 17.80 | 16.62 | 16.51 |

The physics-regularized LSTM was best at every lead time, but the gap over the
regular multivariate wind LSTM was small. Most of the improvement came from
adding period and wind information.

Tracked result files:

```text
extension/final_results/brazil_era5_summary.csv
extension/final_results/brazil_era5_best_pinn_by_lead.csv
```

### La Jolla / Torrey Pines NOAA Buoy, 2017-2023

Final run:

```text
10 epochs, 10 repeats, test size 744 hours
```

| Lead time | Persistence | Hs-only LSTM | Multi-wind LSTM | Physics-reg LSTM |
|---:|---:|---:|---:|---:|
| 6h | 13.03 | 12.87 | 12.94 | 12.90 |
| 12h | 19.04 | 18.25 | 17.94 | 17.79 |
| 18h | 23.87 | 21.94 | 21.12 | 21.06 |
| 24h | 27.88 | 24.18 | 23.69 | 23.44 |

For the NOAA buoy dataset, the Hs-only model was slightly best at 6 hours. The
physics-regularized LSTM was best at 12, 18, and 24 hours. This suggests that
recent wave history is very strong at short lead times, while wind and period
matter more as the forecast gets farther out.

Tracked result files:

```text
extension/final_results/torrey_pines_noaa_summary.csv
extension/final_results/torrey_pines_noaa_best_pinn_by_lead.csv
```

## Interpretation

The main takeaways were:

1. LSTM models beat the persistence baseline for most settings.
2. Adding physical inputs gave the largest improvement, especially on the Brazil
   ERA5 dataset.
3. The physics-informed regularizer gave a small but mostly consistent
   improvement over the multivariate wind LSTM.

The physics regularizer helped, but the improvement was modest. Some differences
are close to the run-to-run variation, so we should not overstate it. The bigger
gain was from adding physical inputs.

## Why This Is Not A Full PINN

A full PINN would usually include a governing equation residual in the loss. For
ocean waves, that could mean a wave action balance equation or another wave
evolution equation.

We did not do that because our data is a point time series. We do not have the
spatial wave spectrum, bathymetry, currents, boundary conditions, or source
terms needed for a full wave-physics residual.

So our approach is better described as:

```text
physics-regularized LSTM
```

instead of:

```text
full PINN
```

## Limitations

- The physics regularizer is simple and heuristic.
- The improvements over the multivariate LSTM are small.
- Average MAPE may hide poor performance on rare storm or extreme-wave events.
- We tested only one Brazil ERA5 point and one NOAA buoy site.
- We did not validate against operational wave models such as SWAN or WAVEWATCH
  III.
- The Brazil experiment uses ERA5 reanalysis data, not direct buoy observations.
- The NOAA buoy dataset is a different location and data source, so its
  numbers should not be compared one-to-one with the Brazil ERA5 numbers.

## Reproducibility Notes

- Final experiments used 10 repeats to reduce dependence on a single random
  training run.
- Seeds are set inside `evaluate_shared_experiments.py`.
- Each model uses the same train, validation, and test split for a given repeat.
- The test size is 744 hourly samples, matching roughly one month of hourly
  forecasts.
- Generated results are written to `extension/results/`, which is ignored by
  git because it can become large.
- Small final summary CSVs are tracked in `extension/final_results/`.

## License

The base project is licensed under the GNU General Public License. See
`LICENSE` for more information.
