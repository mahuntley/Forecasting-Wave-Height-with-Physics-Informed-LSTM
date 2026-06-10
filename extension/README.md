# Extension Folder

This folder has the code and small result summaries we added. The original
authors' scripts are in `../src/`.

## Main Scripts

| Script | Purpose |
|---|---|
| `scripts/evaluate_shared_experiments.py` | Main experiment runner |
| `scripts/convert_era5_for_lstm.py` | Converts ERA5 `u10` and `v10` into wind speed and wind direction for the LSTM format |
| `scripts/download_era5_data.py` | Downloads ERA5 wave and wind data in smaller monthly chunks |
| `scripts/convert_noaa_buoy_to_netcdf.py` | Cleans NOAA buoy text files and converts them to hourly NetCDF |

## Final Results

The small CSV files in `final_results/` are tracked so the main results can be
checked without rerunning the full experiments.

| File | Contents |
|---|---|
| `final_results/brazil_era5_summary.csv` | Brazil ERA5 final metrics |
| `final_results/brazil_era5_best_pinn_by_lead.csv` | Best regularizer weights for Brazil |
| `final_results/torrey_pines_noaa_summary.csv` | NOAA buoy final metrics |
| `final_results/torrey_pines_noaa_best_pinn_by_lead.csv` | Best regularizer weights for the NOAA buoy run |

Larger generated outputs go in `results/`. That folder is ignored by git.

## Final Commands

Brazil ERA5:

```bash
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

La Jolla / Torrey Pines NOAA:

```bash
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

For setup, data preparation, model explanations, and result tables, see
`../README.md`.
