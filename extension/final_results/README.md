# Final Result Summaries

These are the small CSV summaries from the final runs. They are kept in git so
the main results can be checked without rerunning all of the training.

| File | Meaning |
|---|---|
| `brazil_era5_summary.csv` | Final Brazil ERA5 metrics |
| `brazil_era5_best_pinn_by_lead.csv` | Best physics-regularizer weights for each Brazil lead time |
| `torrey_pines_noaa_summary.csv` | Final Torrey Pines NOAA metrics |
| `torrey_pines_noaa_best_pinn_by_lead.csv` | Best physics-regularizer weights for each Torrey Pines lead time |

The larger per-run outputs are written to `extension/results/`, which is ignored
by git.
